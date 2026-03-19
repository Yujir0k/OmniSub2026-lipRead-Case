import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


_FACE_CASCADE = None
_MP_FACE_MESH = None


def _get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


def _get_mediapipe_facemesh():
    global _MP_FACE_MESH
    if _MP_FACE_MESH is not None:
        return _MP_FACE_MESH
    try:
        import mediapipe as mp

        _MP_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception:
        _MP_FACE_MESH = None
    return _MP_FACE_MESH


def normalize_text(text: str) -> str:
    text = str(text).lower()
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789 '")
    text = "".join(ch if ch in allowed else " " for ch in text)
    return " ".join(text.split())


def ids_to_text(ids: List[int], itos: Dict[int, str], token_mode: str) -> str:
    if token_mode == "char":
        return "".join(itos[i] for i in ids if i in itos)
    return " ".join(itos[i] for i in ids if i in itos)


def apply_wordnorm(text: str) -> str:
    token_map = {"e": "the", "o": "of"}
    toks = [token_map.get(t, t) for t in str(text).split()]
    out = []
    prev = None
    cnt = 0
    for t in toks:
        if t == prev:
            cnt += 1
        else:
            prev = t
            cnt = 1
        if cnt <= 2:
            out.append(t)
    return " ".join(out).strip()


def crop_frame(gray: np.ndarray, crop_mode: str) -> np.ndarray:
    if crop_mode in {"mouth", "face_mouth"}:
        h, w = gray.shape
        y1, y2 = int(h * 0.45), int(h * 0.95)
        x1, x2 = int(w * 0.2), int(w * 0.8)
        return gray[y1:y2, x1:x2]
    return gray


def read_video_frames(path: Path, n_frames: int, size: int, crop_mode: str) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return np.zeros((n_frames, size, size), dtype=np.float32)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.zeros((n_frames, size, size), dtype=np.float32)

    frame_ids = np.linspace(0, total - 1, n_frames).astype(np.int32)
    frames: List[np.ndarray] = []
    last_face = None
    face_det = _get_face_cascade() if crop_mode == "face_mouth" else None
    mp_mesh = _get_mediapipe_facemesh() if crop_mode == "mp_mouth" else None
    last_mouth = None

    mouth_ids = [
        61,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        308,
        324,
        318,
        402,
        317,
        14,
        87,
        178,
        88,
        95,
        185,
        40,
        39,
        37,
        0,
        267,
        269,
        270,
        409,
        415,
        310,
        311,
        312,
        13,
        82,
        81,
        42,
        183,
        78,
    ]

    for idx in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            frames.append(np.zeros((size, size), dtype=np.float32))
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if crop_mode == "face_mouth":
            h, w = gray.shape
            small = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            faces = face_det.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                x, y, fw, fh = int(x * 2), int(y * 2), int(fw * 2), int(fh * 2)
                last_face = (x, y, fw, fh)
            if last_face is not None:
                x, y, fw, fh = last_face
                mx1 = max(0, int(x + fw * 0.18))
                mx2 = min(w, int(x + fw * 0.82))
                my1 = max(0, int(y + fh * 0.58))
                my2 = min(h, int(y + fh * 0.98))
                if mx2 > mx1 and my2 > my1:
                    gray = gray[my1:my2, mx1:mx2]
                else:
                    gray = crop_frame(gray, crop_mode="mouth")
            else:
                gray = crop_frame(gray, crop_mode="mouth")
        elif crop_mode == "mp_mouth":
            need_detect = (last_mouth is None) or (len(frames) % 4 == 0)
            if need_detect and mp_mesh is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = mp_mesh.process(rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    h, w = gray.shape
                    xs = [int(lm[i].x * w) for i in mouth_ids if i < len(lm)]
                    ys = [int(lm[i].y * h) for i in mouth_ids if i < len(lm)]
                    if xs and ys:
                        x1, x2 = min(xs), max(xs)
                        y1, y2 = min(ys), max(ys)
                        bw = max(1, x2 - x1)
                        bh = max(1, y2 - y1)
                        px = int(bw * 0.9)
                        py = int(bh * 1.2)
                        mx1 = max(0, x1 - px)
                        mx2 = min(w, x2 + px)
                        my1 = max(0, y1 - py)
                        my2 = min(h, y2 + py)
                        if mx2 > mx1 and my2 > my1:
                            last_mouth = (mx1, my1, mx2, my2)
            if last_mouth is not None:
                mx1, my1, mx2, my2 = last_mouth
                gray = gray[my1:my2, mx1:mx2]
            else:
                gray = crop_frame(gray, crop_mode="mouth")
        else:
            gray = crop_frame(gray, crop_mode=crop_mode)

        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(gray.astype(np.float32) / 255.0)

    cap.release()
    return np.stack(frames, axis=0)


def resolve_test_video_path(data_root: Path, rel_path_raw: str) -> Path:
    rel_path = Path(str(rel_path_raw).strip())
    candidates: List[Path] = [data_root / rel_path]

    if rel_path.parts and rel_path.parts[0].lower() != "test":
        candidates.append(data_root / "test" / rel_path)
    elif len(rel_path.parts) > 1:
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


def get_path_column_name(df: pd.DataFrame) -> str:
    if "path" in df.columns:
        return "path"
    normalized = {str(c).strip().lower(): c for c in df.columns}
    if "path" in normalized:
        return normalized["path"]
    cols = ", ".join(map(str, df.columns.tolist()))
    raise ValueError(f"sample_submission.csv must contain a 'path' column. Found: [{cols}]")


class TinyLipCTC(nn.Module):
    def __init__(self, n_frames: int, vocab_size: int, hidden: int = 192, blank_bias: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(24, 48, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(48, 96, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d((n_frames, 1, 1))
        self.rnn = nn.GRU(
            input_size=96,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.head = nn.Linear(hidden * 2, vocab_size + 1)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[0] = float(blank_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        return self.head(x)


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
        video = read_video_frames(full, self.n_frames, self.frame_size, self.crop_mode)
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
    token_mode = str(cargs.get("token_mode", "char"))
    hidden = int(cargs.get("hidden", args.hidden))

    model = TinyLipCTC(n_frames=n_frames, vocab_size=len(stoi), hidden=hidden)
    model.load_state_dict(ckpt["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"device={device} n_frames={n_frames} frame_size={frame_size} crop_mode={crop_mode} hidden={hidden}")

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
        log_probs_tbc = logits.float().log_softmax(dim=-1).transpose(0, 1)
        lp_btv = log_probs_tbc.transpose(0, 1).detach().cpu().numpy()

        out_texts = []
        for b in range(lp_btv.shape[0]):
            ids = ctc_prefix_beam_search(lp_btv[b], beam_size=args.beam_size, blank=0)
            txt = normalize_text(ids_to_text(ids, itos, token_mode=token_mode))
            out_texts.append(txt)

        for rel, txt in zip(rels, out_texts):
            preds[rel] = txt if txt else args.fallback

        if step % 50 == 0 or step == len(loader):
            print(f"infer_step={step}/{len(loader)}")

    out = pd.DataFrame({"path": rel_paths})
    out["transcription"] = out["path"].map(preds).fillna(args.fallback).astype(str).map(normalize_text)
    out.loc[out["transcription"].eq(""), "transcription"] = args.fallback
    if args.apply_wordnorm:
        out["transcription"] = out["transcription"].map(apply_wordnorm).map(normalize_text)

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
    p = argparse.ArgumentParser("Standalone Kaggle inference from best.pt")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data-root", type=str, required=True, help="Root with test/ and sample_submission.csv")
    p.add_argument("--sample-submission", type=str, required=True)
    p.add_argument("--output-csv", type=str, required=True)
    p.add_argument("--beam-size", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--fallback", type=str, default="i")
    p.add_argument("--n-frames", type=int, default=96)
    p.add_argument("--frame-size", type=int, default=72)
    p.add_argument("--crop-mode", type=str, default="mouth")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--apply-wordnorm", action="store_true")
    main(p.parse_args())
