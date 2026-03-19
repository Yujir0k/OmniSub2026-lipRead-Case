import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
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
    text = text.lower()
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789 '")
    text = "".join(ch if ch in allowed else " " for ch in text)
    return " ".join(text.split())


def parse_text_file(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    line = lines[0] if lines else ""
    if ":" in line:
        line = line.split(":", 1)[1]
    return normalize_text(line)


def load_index(root: Path) -> List[Tuple[str, Path, str]]:
    items: List[Tuple[str, Path, str]] = []
    for txt in root.glob("train/*/*.txt"):
        mp4 = txt.with_suffix(".mp4")
        if not mp4.exists():
            continue
        video_id = txt.parent.name
        text = parse_text_file(txt)
        if not text:
            continue
        items.append((video_id, mp4, text))
    return items


def split_by_video_id(
    items: Sequence[Tuple[str, Path, str]], seed: int = 42, train_frac: float = 0.8
) -> Tuple[List[Tuple[str, Path, str]], List[Tuple[str, Path, str]]]:
    video_ids = sorted({video_id for video_id, _, _ in items})
    rng = random.Random(seed)
    rng.shuffle(video_ids)
    cut = int(len(video_ids) * train_frac)
    train_videos = set(video_ids[:cut])
    train_items = [item for item in items if item[0] in train_videos]
    val_items = [item for item in items if item[0] not in train_videos]
    return train_items, val_items


def sample_items(
    items: Sequence[Tuple[str, Path, str]], n: int, seed: int
) -> List[Tuple[str, Path, str]]:
    if n <= 0 or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(list(items), n)


def build_vocab(
    samples: Sequence[Tuple[str, Path, str]], token_mode: str
) -> Tuple[Dict[str, int], Dict[int, str]]:
    if token_mode == "char":
        units = sorted({ch for _, _, text in samples for ch in text})
    else:
        units = sorted({tok for _, _, text in samples for tok in text.split()})
    stoi = {unit: i + 1 for i, unit in enumerate(units)}  # 0 reserved for CTC blank
    itos = {i: unit for unit, i in stoi.items()}
    return stoi, itos


def text_to_ids(text: str, stoi: Dict[str, int], token_mode: str) -> List[int]:
    if token_mode == "char":
        return [stoi[ch] for ch in text if ch in stoi]
    return [stoi[tok] for tok in text.split() if tok in stoi]


def ids_to_text(ids: Sequence[int], itos: Dict[int, str], token_mode: str) -> str:
    if token_mode == "char":
        return "".join(itos[i] for i in ids if i in itos)
    return " ".join(itos[i] for i in ids if i in itos)


def target_len(text: str, token_mode: str) -> int:
    if token_mode == "char":
        return len(text)
    return len(text.split())


def crop_frame(gray: np.ndarray, crop_mode: str) -> np.ndarray:
    if crop_mode == "mouth":
        h, w = gray.shape
        y1, y2 = int(h * 0.45), int(h * 0.95)
        x1, x2 = int(w * 0.2), int(w * 0.8)
        return gray[y1:y2, x1:x2]
    if crop_mode == "face_mouth":
        # Fallback static crop if face detector misses.
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
    # Stable set of lip landmarks (outer + inner contour).
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
            # Detect on downscaled frame for speed; keep last successful bbox as fallback.
            h, w = gray.shape
            small = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            faces = face_det.detectMultiScale(
                small, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
            )
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
            # Run FaceMesh every few frames; reuse last successful mouth bbox between updates.
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
                        # Expand around mouth for robustness to rotation.
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
    arr = np.stack(frames, axis=0)
    return arr


class LipDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[str, Path, str]],
        stoi: Dict[str, int],
        token_mode: str,
        n_frames: int,
        size: int,
        crop_mode: str,
    ) -> None:
        self.samples = list(samples)
        self.stoi = stoi
        self.token_mode = token_mode
        self.n_frames = n_frames
        self.size = size
        self.crop_mode = crop_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        _, path, text = self.samples[idx]
        video = read_video_frames(path, self.n_frames, self.size, crop_mode=self.crop_mode)
        target = text_to_ids(text, self.stoi, self.token_mode)
        if not target:
            target = [1]  # defensive fallback
        parts = list(path.parts)
        rel = path.as_posix()
        if "train" in parts:
            i = parts.index("train")
            rel = "/".join(parts[i:])
        return torch.from_numpy(video).unsqueeze(1), torch.tensor(target, dtype=torch.long), text, rel


def collate_fn(batch):
    videos, targets, raw_texts, rel_paths = zip(*batch)
    videos = pad_sequence(videos, batch_first=True)  # fixed length in practice
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_cat = torch.cat(targets)
    input_lengths = torch.full((len(videos),), videos.shape[1], dtype=torch.long)
    return videos, input_lengths, targets_cat, target_lengths, raw_texts, rel_paths


class TinyLipCTC(nn.Module):
    def __init__(
        self,
        n_frames: int,
        vocab_size: int,
        hidden: int = 192,
        blank_bias: float = 0.0,
    ) -> None:
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
        self.head = nn.Linear(hidden * 2, vocab_size + 1)  # +1 for CTC blank
        with torch.no_grad():
            self.head.bias.zero_()
            # Keep blank prior configurable: smoke runs may use negative bias,
            # full training is typically more stable with zero bias.
            self.head.bias[0] = float(blank_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 1, H, W] -> [B, 1, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # [B, C, T]
        x = x.permute(0, 2, 1)  # [B, T, C]
        x, _ = self.rnn(x)
        logits = self.head(x)  # [B, T, V]
        return logits


def greedy_decode(log_probs: torch.Tensor, itos: Dict[int, str], token_mode: str) -> List[str]:
    # log_probs: [T, B, V]
    pred = log_probs.argmax(dim=-1).transpose(0, 1).cpu().numpy()  # [B, T]
    texts = []
    for seq in pred:
        out = []
        prev = -1
        for token in seq:
            token = int(token)
            if token != 0 and token != prev:
                out.append(token)
            prev = token
        texts.append(normalize_text(ids_to_text(out, itos, token_mode)))
    return texts


def greedy_decode_with_conf(
    log_probs: torch.Tensor, itos: Dict[int, str], token_mode: str
) -> Tuple[List[str], List[float]]:
    probs = log_probs.exp()  # [T, B, V]
    max_probs, max_ids = probs.max(dim=-1)
    max_ids = max_ids.transpose(0, 1).cpu().numpy()  # [B, T]
    max_probs = max_probs.transpose(0, 1).cpu().numpy()  # [B, T]

    texts: List[str] = []
    confs: List[float] = []
    for seq_ids, seq_probs in zip(max_ids, max_probs):
        out = []
        keep_probs = []
        prev = -1
        for token, p in zip(seq_ids, seq_probs):
            token = int(token)
            if token != 0 and token != prev:
                out.append(token)
                keep_probs.append(float(p))
            prev = token
        text = normalize_text(ids_to_text(out, itos, token_mode))
        conf = float(np.mean(keep_probs)) if keep_probs else float(np.mean(seq_probs))
        texts.append(text)
        confs.append(conf)
    return texts, confs


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m]


def wer(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


@dataclass
class EvalResult:
    loss: float
    wer: float
    examples: List[Tuple[str, str]]
    pred_rows: List[Tuple[str, str, str, float]]


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    itos: Dict[int, str],
    token_mode: str,
    max_batches: int,
) -> EvalResult:
    model.eval()
    losses = []
    wers = []
    examples: List[Tuple[str, str]] = []
    pred_rows: List[Tuple[str, str, str, float]] = []
    for step, batch in enumerate(loader, 1):
        if max_batches > 0 and step > max_batches:
            break
        videos, input_lengths, targets_cat, target_lengths, raw_texts, rel_paths = batch
        videos = videos.to(device, non_blocking=True)
        targets_cat = targets_cat.to(device, non_blocking=True)
        input_lengths = input_lengths.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)

        logits = model(videos)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        loss = criterion(log_probs, targets_cat, input_lengths, target_lengths)
        losses.append(float(loss.item()))

        decoded, confs = greedy_decode_with_conf(log_probs, itos, token_mode)
        for gt, pred, conf, rel in zip(raw_texts, decoded, confs, rel_paths):
            wers.append(wer(gt, pred))
            pred_rows.append((rel, gt, pred, conf))
            if len(examples) < 5:
                examples.append((gt, pred))
    return EvalResult(
        loss=float(np.mean(losses)) if losses else 0.0,
        wer=float(np.mean(wers)) if wers else 1.0,
        examples=examples,
        pred_rows=pred_rows,
    )


def save_pred_rows(path: Path, rows: Sequence[Tuple[str, str, str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "reference", "prediction", "confidence"])
        writer.writerows(rows)


def train(args):
    root = Path(args.data_root).resolve()
    items = load_index(root)
    print(f"Loaded {len(items)} train clips.")
    run_dir = Path(args.output_dir).resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    train_items, val_items = split_by_video_id(items, seed=args.seed, train_frac=0.8)
    train_items = sample_items(train_items, args.train_samples, seed=args.seed + 1)
    val_items = sample_items(val_items, args.val_samples, seed=args.seed + 2)
    print(f"Using {len(train_items)} train and {len(val_items)} val samples (video_id split).")

    if args.max_target_len > 0:
        train_items = [x for x in train_items if target_len(x[2], args.token_mode) <= args.max_target_len]
        val_items = [x for x in val_items if target_len(x[2], args.token_mode) <= args.max_target_len]
        print(
            f"Filtered by max_target_len={args.max_target_len}: "
            f"train={len(train_items)} val={len(val_items)}"
        )

    stoi, itos = build_vocab(train_items, token_mode=args.token_mode)
    print(f"Vocab size ({args.token_mode}): {len(stoi)}")

    train_ds = LipDataset(
        train_items,
        stoi,
        token_mode=args.token_mode,
        n_frames=args.n_frames,
        size=args.frame_size,
        crop_mode=args.crop_mode,
    )
    val_ds = LipDataset(
        val_items,
        stoi,
        token_mode=args.token_mode,
        n_frames=args.n_frames,
        size=args.frame_size,
        crop_mode=args.crop_mode,
    )

    num_workers = min(args.num_workers, 8)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyLipCTC(
        n_frames=args.n_frames,
        vocab_size=len(stoi),
        hidden=args.hidden,
        blank_bias=args.blank_bias,
    ).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))

    print(
        f"Device: {device}, batch_size={args.batch_size}, n_frames={args.n_frames}, "
        f"frame_size={args.frame_size}, crop_mode={args.crop_mode}"
    )
    print(f"Starting training. Logs: {run_dir}")
    global_step = 0
    start_time = time.time()
    best_wer = float("inf")
    best_epoch = -1
    start_epoch = 1
    metrics_path = run_dir / "metrics.csv"
    if args.resume_ckpt:
        resume_path = Path(args.resume_ckpt).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        resumed = torch.load(resume_path, map_location="cpu")
        state = resumed["model_state"]
        if args.resume_ignore_head:
            state = {k: v for k, v in state.items() if not k.startswith("head.")}
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(
                f"Resumed backbone from {resume_path}: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        else:
            model.load_state_dict(state, strict=True)
        if "optimizer_state" in resumed and not args.resume_ignore_head:
            optimizer.load_state_dict(resumed["optimizer_state"])
            # Keep CLI learning-rate authoritative for resumed runs.
            for pg in optimizer.param_groups:
                pg["lr"] = float(args.lr)
        if "scaler_state" in resumed and device.type == "cuda" and not args.resume_ignore_head:
            scaler.load_state_dict(resumed["scaler_state"])
        start_epoch = int(resumed.get("epoch", 0)) + 1
        best_wer = float(resumed.get("best_val_wer", resumed.get("val_wer", float("inf"))))
        best_epoch = int(resumed.get("best_epoch", resumed.get("epoch", -1)))
        global_step = int(resumed.get("global_step", 0))
        print(
            f"Resumed from {resume_path}: start_epoch={start_epoch}, "
            f"best_wer={best_wer:.4f}, best_epoch={best_epoch}, global_step={global_step}"
        )

    if (not args.resume_ckpt) or (not metrics_path.exists()):
        with metrics_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_wer",
                    "best_val_wer",
                    "epoch_time_sec",
                    "global_step",
                    "peak_vram_alloc_gb",
                    "peak_vram_reserved_gb",
                ]
            )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(train_loader, 1):
            if args.steps_per_epoch > 0 and step > args.steps_per_epoch:
                break
            videos, input_lengths, targets_cat, target_lengths, _, _ = batch
            videos = videos.to(device, non_blocking=True)
            targets_cat = targets_cat.to(device, non_blocking=True)
            input_lengths = input_lengths.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(videos)
            log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
            loss = criterion(log_probs, targets_cat, input_lengths, target_lengths)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(float(loss.item()))
            global_step += 1

            if step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step}/{args.steps_per_epoch} "
                    f"loss={np.mean(epoch_losses[-args.log_every:]):.4f} "
                    f"time={time.time()-start_time:.1f}s"
                )

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_res = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            itos=itos,
            token_mode=args.token_mode,
            max_batches=args.val_batches,
        )
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_res.loss:.4f} val_wer={val_res.wer:.4f}")
        for gt, pred in val_res.examples[:3]:
            print(f"  gt:   {gt}")
            print(f"  pred: {pred}")

        epoch_time = time.time() - epoch_start
        peak_mem_alloc_gb = 0.0
        peak_mem_reserved_gb = 0.0
        if device.type == "cuda":
            peak_mem_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            peak_mem_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024**3)
            print(
                f"[Epoch {epoch}] peak_vram_alloc={peak_mem_alloc_gb:.2f}GB "
                f"peak_vram_reserved={peak_mem_reserved_gb:.2f}GB"
            )
        with metrics_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_res.loss:.6f}",
                    f"{val_res.wer:.6f}",
                    f"{min(best_wer, val_res.wer):.6f}",
                    f"{epoch_time:.2f}",
                    global_step,
                    f"{peak_mem_alloc_gb:.3f}",
                    f"{peak_mem_reserved_gb:.3f}",
                ]
            )

        ckpt_payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if device.type == "cuda" else None,
            "stoi": stoi,
            "itos": itos,
            "epoch": epoch,
            "global_step": global_step,
            "best_val_wer": float(best_wer),
            "best_epoch": int(best_epoch),
            "val_wer": float(val_res.wer),
            "val_loss": float(val_res.loss),
            "train_loss": float(train_loss),
            "args": vars(args),
        }
        if args.save_last:
            torch.save(ckpt_payload, run_dir / "last.pt")

        if val_res.wer < best_wer:
            best_wer = float(val_res.wer)
            best_epoch = epoch
            ckpt_payload["best_val_wer"] = float(best_wer)
            ckpt_payload["best_epoch"] = int(best_epoch)
            torch.save(ckpt_payload, run_dir / "best.pt")
            save_pred_rows(run_dir / "best_val_predictions.csv", val_res.pred_rows)
            print(f"[Epoch {epoch}] best checkpoint updated: val_wer={best_wer:.4f}")

    elapsed = time.time() - start_time
    print(f"Finished. total_time={elapsed/60:.1f} min, best_epoch={best_epoch}, best_val_wer={best_wer:.4f}")


def parse_args():
    parser = argparse.ArgumentParser("Quick CTC smoke-test for OmniSub data")
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--val-samples", type=int, default=400)
    parser.add_argument("--token-mode", type=str, default="char", choices=["char", "word"])
    parser.add_argument("--max-target-len", type=int, default=0)
    parser.add_argument("--n-frames", type=int, default=40)
    parser.add_argument("--frame-size", type=int, default=64)
    parser.add_argument(
        "--crop-mode",
        type=str,
        default="full",
        choices=["full", "mouth", "face_mouth", "mp_mouth"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=120)
    parser.add_argument("--val-batches", type=int, default=25)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--blank-bias", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default="ctc_run")
    parser.add_argument("--save-last", action="store_true")
    parser.add_argument("--resume-ckpt", type=str, default="")
    parser.add_argument("--resume-ignore-head", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
