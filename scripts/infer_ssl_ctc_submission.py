import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import quick_ctc_smoke as q
import train_ssl_ctc_curriculum as ssl_ctc


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
    def __init__(self, paths: List[str], root: Path, n_frames: int, frame_size: int, crop_mode: str) -> None:
        self.paths = paths
        self.root = root
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.crop_mode = crop_mode
        self.full_paths = [resolve_test_video_path(self.root, rp) for rp in self.paths]

    def __len__(self) -> int:
        return len(self.full_paths)

    def __getitem__(self, idx: int):
        rel = self.paths[idx]
        mp4 = self.full_paths[idx]
        arr = q.read_video_frames(mp4, self.n_frames, self.frame_size, self.crop_mode)
        return torch.from_numpy(arr).unsqueeze(1), rel


def collate_test(batch):
    vids, rels = zip(*batch)
    vids = torch.stack(vids, 0)
    return vids, list(rels)


def decode_greedy(log_probs_btv: torch.Tensor, itos: Dict[int, str]) -> List[str]:
    pred = log_probs_btv.argmax(dim=-1).cpu().numpy()  # [B,T]
    out: List[str] = []
    for seq in pred:
        ids: List[int] = []
        prev = -1
        for t in seq:
            tid = int(t)
            if tid != 0 and tid != prev:
                ids.append(tid)
            prev = tid
        out.append(q.normalize_text(q.ids_to_text(ids, itos, token_mode="char")))
    return out


def decode_beam(log_probs_btv: torch.Tensor, itos: Dict[int, str], beam_size: int) -> List[str]:
    out: List[str] = []
    arr = log_probs_btv.cpu().numpy()
    for i in range(arr.shape[0]):
        ids = ssl_ctc.ctc_prefix_beam_search(arr[i], beam_size=beam_size, blank=0)
        out.append(q.normalize_text(q.ids_to_text(ids, itos, token_mode="char")))
    return out


@torch.inference_mode()
def main(args):
    root = Path(args.data_root).resolve()
    ckpt = torch.load(Path(args.ckpt).resolve(), map_location="cpu")
    ck_args = ckpt.get("args", {})

    n_frames = int(args.n_frames if args.n_frames > 0 else ck_args.get("n_frames", 96))
    frame_size = int(args.frame_size if args.frame_size > 0 else ck_args.get("frame_size", 72))
    crop_mode = str(args.crop_mode if args.crop_mode else ck_args.get("crop_mode", "mouth"))
    hidden = int(args.hidden if args.hidden > 0 else ck_args.get("hidden", 256))

    stoi: Dict[str, int] = ckpt["stoi"]
    itos = {i: s for s, i in stoi.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ssl_ctc.SSLCTCModel(vocab_size=len(stoi), hidden=hidden).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    sample_path = Path(args.sample_submission)
    if not sample_path.is_absolute():
        sample_path = root / sample_path
    sample = pd.read_csv(sample_path)
    paths = sample["path"].astype(str).tolist()

    ds = TestDataset(paths, root=root, n_frames=n_frames, frame_size=frame_size, crop_mode=crop_mode)
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

    preds: Dict[str, str] = {}
    total = len(loader)
    for step, (videos, rels) in enumerate(loader, 1):
        videos = videos.to(device, non_blocking=True)
        logits = model.forward_ctc(videos)
        log_probs_btv = logits.log_softmax(dim=-1)
        if args.decode == "beam":
            texts = decode_beam(log_probs_btv, itos=itos, beam_size=args.beam_size)
        else:
            texts = decode_greedy(log_probs_btv, itos=itos)
        for r, t in zip(rels, texts):
            preds[r] = t
        if step % max(1, args.log_every) == 0:
            print(f"decode_step={step}/{total}")

    out = sample[["path"]].copy()
    out["transcription"] = out["path"].map(preds).fillna(args.fallback).astype(str).map(q.normalize_text)
    out.loc[out["transcription"].eq(""), "transcription"] = args.fallback

    out_csv = Path(args.output_csv)
    if not out_csv.is_absolute():
        out_csv = root / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"saved: {out_csv}")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser("Infer submission from SSL+CTC checkpoint")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--sample-submission", type=str, default="sample_submission.csv")
    p.add_argument("--output-csv", type=str, required=True)
    p.add_argument("--decode", type=str, default="greedy", choices=["greedy", "beam"])
    p.add_argument("--beam-size", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--n-frames", type=int, default=0)
    p.add_argument("--frame-size", type=int, default=0)
    p.add_argument("--crop-mode", type=str, default="")
    p.add_argument("--hidden", type=int, default=0)
    p.add_argument("--fallback", type=str, default="i")
    p.add_argument("--log-every", type=int, default=40)
    main(p.parse_args())
