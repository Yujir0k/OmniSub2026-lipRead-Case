import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import quick_ctc_smoke as q


def get_path_column_name(df: pd.DataFrame) -> str:
    if "path" in df.columns:
        return "path"
    normalized = {str(c).strip().lower(): c for c in df.columns}
    if "path" in normalized:
        return normalized["path"]
    cols = ", ".join(map(str, df.columns.tolist()))
    raise ValueError(f"sample_submission.csv must contain a 'path' column. Found: [{cols}]")


def resolve_test_video_path(data_root: Path, rel_path_raw: str) -> Path:
    rel_path = Path(str(rel_path_raw).strip())
    candidates: List[Path] = []

    # 1) Old format compatibility: "test/<video_id>/<clip_id>.mp4"
    candidates.append(data_root / rel_path)

    # 2) New final format compatibility: "<clip_id>.mp4" with files under "test/".
    if rel_path.parts and rel_path.parts[0].lower() != "test":
        candidates.append(data_root / "test" / rel_path)
    else:
        # If path already starts with test/, also try dropping the first segment.
        if len(rel_path.parts) > 1:
            tail = Path(*rel_path.parts[1:])
            candidates.append(data_root / "test" / tail)
            candidates.append(data_root / tail)

    # 3) Flat fallback by basename (useful when sample format and folder structure differ).
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

    # Last-resort recursive search by filename.
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


def logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def ctc_prefix_beam_search(log_probs_tv: np.ndarray, beam_size: int = 30, blank: int = 0) -> List[int]:
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, -math.inf)}
    t_max, v_size = log_probs_tv.shape

    for t in range(t_max):
        next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = defaultdict(lambda: (-math.inf, -math.inf))
        lp = log_probs_tv[t]

        for prefix, (pb, pnb) in beams.items():
            nb_pb, nb_pnb = next_beams[prefix]
            nb_pb = logsumexp(nb_pb, pb + float(lp[blank]))
            nb_pb = logsumexp(nb_pb, pnb + float(lp[blank]))
            next_beams[prefix] = (nb_pb, nb_pnb)

            last = prefix[-1] if prefix else None
            for c in range(v_size):
                if c == blank:
                    continue
                p = float(lp[c])
                new_prefix = prefix + (c,)
                if c == last:
                    nb_pb, nb_pnb = next_beams[prefix]
                    nb_pnb = logsumexp(nb_pnb, pnb + p)
                    next_beams[prefix] = (nb_pb, nb_pnb)

                    nb_pb2, nb_pnb2 = next_beams[new_prefix]
                    nb_pnb2 = logsumexp(nb_pnb2, pb + p)
                    next_beams[new_prefix] = (nb_pb2, nb_pnb2)
                else:
                    nb_pb2, nb_pnb2 = next_beams[new_prefix]
                    nb_pnb2 = logsumexp(nb_pnb2, pb + p)
                    nb_pnb2 = logsumexp(nb_pnb2, pnb + p)
                    next_beams[new_prefix] = (nb_pb2, nb_pnb2)

        scored = [(logsumexp(pb, pnb), pr) for pr, (pb, pnb) in next_beams.items()]
        scored.sort(key=lambda x: x[0], reverse=True)
        beams = {pr: next_beams[pr] for _, pr in scored[:beam_size]}

    best_prefix = max(beams.items(), key=lambda kv: logsumexp(kv[1][0], kv[1][1]))[0]
    return list(best_prefix)


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


def collate_test(batch):
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
    itos = ckpt["itos"]
    if isinstance(itos, dict) and itos and isinstance(next(iter(itos.keys())), str):
        itos = {int(k): v for k, v in itos.items()}
    cargs = ckpt.get("args", {})
    n_frames = int(cargs.get("n_frames", args.n_frames))
    frame_size = int(cargs.get("frame_size", args.frame_size))
    crop_mode = str(cargs.get("crop_mode", args.crop_mode))
    hidden = int(cargs.get("hidden", args.hidden))

    model = q.TinyLipCTC(n_frames=n_frames, vocab_size=len(stoi), hidden=hidden)
    model.load_state_dict(ckpt["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    sample = pd.read_csv(sample_path)
    path_col = get_path_column_name(sample)
    raw_paths = sample[path_col]
    if raw_paths.isna().any():
        raise ValueError("sample_submission.csv contains null values in the 'path' column")
    rel_paths = raw_paths.astype(str).str.strip().tolist()
    if any(not p for p in rel_paths):
        raise ValueError("sample_submission.csv contains empty values in the 'path' column")

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
        log_probs_tbc = logits.float().log_softmax(dim=-1).transpose(0, 1)  # [T,B,V]
        lp_btv = log_probs_tbc.transpose(0, 1).detach().cpu().numpy()  # [B,T,V]

        out_texts = []
        for b in range(lp_btv.shape[0]):
            ids = ctc_prefix_beam_search(lp_btv[b], beam_size=args.beam_size, blank=0)
            txt = q.normalize_text(q.ids_to_text(ids, itos, token_mode="char"))
            out_texts.append(txt)

        for rel, txt in zip(rels, out_texts):
            preds[rel] = txt if txt else args.fallback

        if step % 50 == 0:
            print(f"infer_step={step}/{len(loader)}")

    out = pd.DataFrame({"path": rel_paths})
    out["transcription"] = out["path"].map(preds).fillna(args.fallback).astype(str).map(q.normalize_text)
    out.loc[out["transcription"].eq(""), "transcription"] = args.fallback
    out = out[["path", "transcription"]]

    nulls = int(out["transcription"].isna().sum())
    empties = int(out["transcription"].astype(str).str.strip().eq("").sum())
    if nulls > 0 or empties > 0:
        raise RuntimeError(f"submission still has invalid values: nulls={nulls}, empties={empties}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved={out_path}")
    print(f"rows={len(out)} nulls={nulls} empties={empties} path_col={path_col}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Infer test submission with CTC beam decode")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--sample-submission", type=str, default="sample_submission.csv")
    p.add_argument("--output-csv", type=str, required=True)
    p.add_argument("--beam-size", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--fallback", type=str, default="i")
    p.add_argument("--n-frames", type=int, default=96)
    p.add_argument("--frame-size", type=int, default=72)
    p.add_argument("--crop-mode", type=str, default="mouth")
    p.add_argument("--hidden", type=int, default=256)
    main(p.parse_args())
