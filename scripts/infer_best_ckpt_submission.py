import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import quick_ctc_smoke as q


def resolve_test_video_path(data_root: Path, rel_path_raw: str) -> Path:
    rel_path = Path(str(rel_path_raw).strip())
    candidates: List[Path] = []

    candidates.append(data_root / rel_path)

    if rel_path.parts and rel_path.parts[0].lower() != "test":
        candidates.append(data_root / "test" / rel_path)
    else:
        if len(rel_path.parts) > 1:
            tail = Path(*rel_path.parts[1:])
            candidates.append(data_root / "test" / tail)
            candidates.append(data_root / tail)

    if rel_path.name:
        candidates.append(data_root / "test" / rel_path.name)
        candidates.append(data_root / rel_path.name)

    seen = set()
    uniq_candidates: List[Path] = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            uniq_candidates.append(c)
            seen.add(key)

    for c in uniq_candidates:
        if c.exists():
            return c

    if rel_path.name:
        search_root = data_root / "test"
        if search_root.exists():
            matches = list(search_root.rglob(rel_path.name))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                preview = ", ".join(str(x) for x in matches[:3])
                raise FileNotFoundError(
                    f"Ambiguous test path '{rel_path_raw}': {len(matches)} matches ({preview} ...)"
                )

    tried = ", ".join(str(x) for x in uniq_candidates)
    raise FileNotFoundError(f"Could not resolve test path '{rel_path_raw}'. Tried: {tried}")


class TestDataset(Dataset):
    def __init__(self, data_root: Path, rel_paths: List[str], n_frames: int, frame_size: int, crop_mode: str):
        self.data_root = data_root
        self.rel_paths = rel_paths
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.crop_mode = crop_mode
        self.full_paths = [resolve_test_video_path(self.data_root, rp) for rp in self.rel_paths]

    def __len__(self) -> int:
        return len(self.full_paths)

    def __getitem__(self, idx: int):
        rel = self.rel_paths[idx]
        full = self.full_paths[idx]
        video = q.read_video_frames(full, self.n_frames, self.frame_size, self.crop_mode)
        return torch.from_numpy(video).unsqueeze(1), rel


def collate_test(batch: List[Tuple[torch.Tensor, str]]):
    videos, rels = zip(*batch)
    videos = torch.stack(videos, dim=0)
    return videos, list(rels)


@torch.inference_mode()
def main(args):
    ckpt_path = Path(args.ckpt).resolve()
    data_root = Path(args.data_root).resolve()
    sample_path = Path(args.sample_submission).resolve()
    out_path = Path(args.output_csv).resolve()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    stoi = ckpt["stoi"]
    itos = {int(k): v for k, v in ckpt["itos"].items()} if isinstance(next(iter(ckpt["itos"].keys())), str) else ckpt["itos"]
    cargs = ckpt.get("args", {})
    n_frames = int(cargs.get("n_frames", args.n_frames))
    frame_size = int(cargs.get("frame_size", args.frame_size))
    crop_mode = str(cargs.get("crop_mode", args.crop_mode))
    token_mode = str(cargs.get("token_mode", "char"))
    hidden = int(cargs.get("hidden", args.hidden))

    model = q.TinyLipCTC(n_frames=n_frames, vocab_size=len(stoi), hidden=hidden)
    model.load_state_dict(ckpt["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    sample = pd.read_csv(sample_path)
    if "path" not in sample.columns:
        raise ValueError("sample_submission.csv must contain 'path' column")
    rel_paths = sample["path"].astype(str).tolist()

    ds = TestDataset(data_root=data_root, rel_paths=rel_paths, n_frames=n_frames, frame_size=frame_size, crop_mode=crop_mode)
    print(f"resolved_paths={len(ds)} first={ds.full_paths[0] if len(ds) else 'n/a'}")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_test,
        persistent_workers=args.num_workers > 0,
    )

    preds = {}
    for step, (videos, rels) in enumerate(loader, 1):
        videos = videos.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(videos)
        log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
        texts = q.greedy_decode(log_probs, itos, token_mode=token_mode)
        for rel, txt in zip(rels, texts):
            txt = q.normalize_text(txt)
            preds[rel] = txt if txt else args.fallback
        if step % 50 == 0:
            print(f"infer_step={step}/{len(loader)}")

    out = sample[["path"]].copy()
    out["transcription"] = out["path"].map(preds).fillna(args.fallback).astype(str).map(q.normalize_text)
    out.loc[out["transcription"].eq(""), "transcription"] = args.fallback

    nulls = int(out["transcription"].isna().sum())
    empties = int(out["transcription"].astype(str).str.strip().eq("").sum())
    if nulls > 0 or empties > 0:
        raise RuntimeError(f"submission still has invalid values: nulls={nulls}, empties={empties}")

    out.to_csv(out_path, index=False)
    print(f"saved={out_path}")
    print(f"rows={len(out)} nulls={nulls} empties={empties}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Infer test submission from best CTC checkpoint")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--sample-submission", type=str, default="sample_submission.csv")
    p.add_argument("--output-csv", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--fallback", type=str, default="i")
    p.add_argument("--n-frames", type=int, default=160)
    p.add_argument("--frame-size", type=int, default=80)
    p.add_argument("--crop-mode", type=str, default="mouth")
    p.add_argument("--hidden", type=int, default=384)
    main(p.parse_args())
