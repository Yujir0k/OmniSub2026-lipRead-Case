import argparse
import csv
import json
import math
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import quick_ctc_smoke as q


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_text_conf(path: Path) -> Tuple[str, int]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    m_text = re.search(r"(?im)^\s*Text\s*:\s*(.*)$", raw)
    text = m_text.group(1).strip() if m_text else ""
    text = q.normalize_text(text)
    m_conf = re.search(r"(?im)^\s*Conf\s*:\s*([0-9]+)\s*$", raw)
    conf = int(m_conf.group(1)) if m_conf else 3
    return text, conf


def load_train_items(root: Path) -> List[Tuple[str, Path, str, int]]:
    items: List[Tuple[str, Path, str, int]] = []
    for txt in root.glob("train/*/*.txt"):
        mp4 = txt.with_suffix(".mp4")
        if not mp4.exists():
            continue
        video_id = txt.parent.name
        text, conf = parse_text_conf(txt)
        if not text:
            continue
        items.append((video_id, mp4, text, conf))
    return items


def load_unlabeled_paths(root: Path) -> List[Path]:
    paths = []
    paths.extend(root.glob("train/*/*.mp4"))
    paths.extend(root.glob("test/*/*.mp4"))
    return [p for p in paths if p.exists()]


def split_train_val(
    items: Sequence[Tuple[str, Path, str, int]],
    seed: int = 42,
    train_frac: float = 0.8,
) -> Tuple[List[Tuple[str, Path, str, int]], List[Tuple[str, Path, str, int]]]:
    vids = sorted({x[0] for x in items})
    rng = random.Random(seed)
    rng.shuffle(vids)
    cut = int(len(vids) * train_frac)
    train_vids = set(vids[:cut])
    train_items = [x for x in items if x[0] in train_vids]
    val_items = [x for x in items if x[0] not in train_vids]
    return train_items, val_items


def build_char_vocab(samples: Sequence[Tuple[str, Path, str, int]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted({ch for _, _, text, _ in samples for ch in text})
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}  # 0 = blank
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def text_to_ids(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[ch] for ch in text if ch in stoi]


class SSLVideoDataset(Dataset):
    def __init__(self, paths: Sequence[Path], n_frames: int, frame_size: int, crop_mode: str) -> None:
        self.paths = list(paths)
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.crop_mode = crop_mode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.paths[idx]
        arr = q.read_video_frames(p, self.n_frames, self.frame_size, crop_mode=self.crop_mode)
        return torch.from_numpy(arr).unsqueeze(1)


class CTCVideoDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Tuple[str, Path, str, int]],
        stoi: Dict[str, int],
        n_frames: int,
        frame_size: int,
        crop_mode: str,
    ) -> None:
        self.items = list(items)
        self.stoi = stoi
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.crop_mode = crop_mode

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        _, mp4, text, conf = self.items[idx]
        arr = q.read_video_frames(mp4, self.n_frames, self.frame_size, crop_mode=self.crop_mode)
        tgt = text_to_ids(text, self.stoi)
        if not tgt:
            tgt = [1]
        rel = mp4.as_posix()
        parts = list(mp4.parts)
        if "train" in parts:
            i = parts.index("train")
            rel = "/".join(parts[i:])
        return (
            torch.from_numpy(arr).unsqueeze(1),
            torch.tensor(tgt, dtype=torch.long),
            text,
            rel,
            int(conf),
        )


def collate_ssl(batch: Sequence[torch.Tensor]) -> torch.Tensor:
    return pad_sequence(batch, batch_first=True)


def collate_ctc(batch):
    videos, targets, raw_texts, rel_paths, confs = zip(*batch)
    videos = pad_sequence(videos, batch_first=True)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_cat = torch.cat(targets)
    input_lengths = torch.full((len(videos),), videos.shape[1], dtype=torch.long)
    confs = torch.tensor(confs, dtype=torch.long)
    return videos, input_lengths, targets_cat, target_lengths, raw_texts, rel_paths, confs


class VisualEncoder(nn.Module):
    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        # videos: [B, T, 1, H, W]
        b, t, c, h, w = videos.shape
        x = videos.reshape(b * t, c, h, w)
        x = self.cnn(x).reshape(b, t, 128)
        y, _ = self.rnn(x)
        return y  # [B, T, 2H]


class SSLCTCModel(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 256, proj_dim: int = 128):
        super().__init__()
        self.encoder = VisualEncoder(hidden=hidden)
        self.ssl_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )
        self.ctc_head = nn.Linear(hidden * 2, vocab_size + 1)
        with torch.no_grad():
            self.ctc_head.bias.zero_()
            self.ctc_head.bias[0] = -2.5

    def forward_features(self, videos: torch.Tensor) -> torch.Tensor:
        return self.encoder(videos)

    def forward_ssl(self, videos: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(videos)  # [B,T,C]
        pooled = feat.mean(dim=1)
        z = self.ssl_head(pooled)
        z = nn.functional.normalize(z, dim=-1)
        return z

    def forward_ctc(self, videos: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(videos)
        logits = self.ctc_head(feat)
        return logits


def augment_views(videos: torch.Tensor, noise_std: float = 0.04) -> Tuple[torch.Tensor, torch.Tensor]:
    def _aug(x: torch.Tensor) -> torch.Tensor:
        b, t, _, _, _ = x.shape
        out = x.clone()
        # brightness/contrast per-sample
        alpha = torch.empty((b, 1, 1, 1, 1), device=x.device).uniform_(0.85, 1.15)
        beta = torch.empty((b, 1, 1, 1, 1), device=x.device).uniform_(-0.08, 0.08)
        out = out * alpha + beta
        out = out + torch.randn_like(out) * noise_std
        # random frame dropout
        drop = (torch.rand((b, t, 1, 1, 1), device=x.device) < 0.08).float()
        out = out * (1.0 - drop)
        out = out.clamp(0.0, 1.0)
        return out

    return _aug(videos), _aug(videos)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    b = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B,D]
    sim = (torch.matmul(z, z.T)).float() / temperature
    mask = torch.eye(2 * b, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    targets = torch.arange(2 * b, device=z.device)
    targets = (targets + b) % (2 * b)
    loss = nn.functional.cross_entropy(sim, targets)
    return loss


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


@dataclass
class EvalOut:
    val_loss: float
    val_wer_greedy: float
    examples: List[Tuple[str, str]]
    pred_rows: List[Tuple[str, str, str]]


@torch.inference_mode()
def evaluate_greedy(
    model: SSLCTCModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    itos: Dict[int, str],
    max_batches: int = 0,
) -> EvalOut:
    model.eval()
    losses: List[float] = []
    wers: List[float] = []
    examples: List[Tuple[str, str]] = []
    pred_rows: List[Tuple[str, str, str]] = []
    for step, batch in enumerate(loader, 1):
        if max_batches > 0 and step > max_batches:
            break
        videos, input_lengths, targets_cat, target_lengths, raw_texts, rel_paths, _ = batch
        videos = videos.to(device, non_blocking=True)
        input_lengths = input_lengths.to(device, non_blocking=True)
        targets_cat = targets_cat.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)
        logits = model.forward_ctc(videos)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        loss = criterion(log_probs, targets_cat, input_lengths, target_lengths)
        losses.append(float(loss.item()))
        dec = q.greedy_decode(log_probs, itos, token_mode="char")
        for gt, pr, rel in zip(raw_texts, dec, rel_paths):
            gt = q.normalize_text(gt)
            pr = q.normalize_text(pr)
            wers.append(q.wer(gt, pr))
            pred_rows.append((rel, gt, pr))
            if len(examples) < 4:
                examples.append((gt, pr))
    return EvalOut(
        val_loss=float(np.mean(losses) if losses else 0.0),
        val_wer_greedy=float(np.mean(wers) if wers else 1.0),
        examples=examples,
        pred_rows=pred_rows,
    )


@torch.inference_mode()
def evaluate_beam_sample(
    model: SSLCTCModel,
    items: Sequence[Tuple[str, Path, str, int]],
    itos: Dict[int, str],
    n_frames: int,
    frame_size: int,
    crop_mode: str,
    batch_size: int,
    device: torch.device,
    beam_size: int,
    sample_size: int,
    seed: int,
) -> float:
    if sample_size <= 0:
        return float("nan")
    items = list(items)
    if sample_size < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, sample_size)
    ds = CTCVideoDataset(items, stoi={c: i for i, c in enumerate("a", start=1)}, n_frames=n_frames, frame_size=frame_size, crop_mode=crop_mode)
    # We only need videos/texts here; dummy stoi is replaced by manual unpacking below.
    def coll(batch):
        vids = [x[0] for x in batch]
        txt = [q.normalize_text(x[2]) for x in batch]
        return torch.stack(vids, 0), txt

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=coll)
    model.eval()
    wers = []
    for videos, refs in loader:
        videos = videos.to(device, non_blocking=True)
        logits = model.forward_ctc(videos)
        lp = logits.log_softmax(dim=-1).transpose(0, 1)
        lp_btv = lp.transpose(0, 1).detach().cpu().numpy()
        preds = []
        for b in range(lp_btv.shape[0]):
            ids = ctc_prefix_beam_search(lp_btv[b], beam_size=beam_size, blank=0)
            txt = q.normalize_text(q.ids_to_text(ids, itos, token_mode="char"))
            preds.append(txt)
        for gt, pr in zip(refs, preds):
            wers.append(q.wer(gt, pr))
    return float(np.mean(wers) if wers else 1.0)


def save_pred_rows(path: Path, rows: Sequence[Tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["path", "reference", "prediction"])
        wr.writerows(rows)


def curriculum_filter(
    items: Sequence[Tuple[str, Path, str, int]],
    min_conf: int,
    max_len: int,
    n_frames: int,
) -> List[Tuple[str, Path, str, int]]:
    out = []
    for it in items:
        _, _, text, conf = it
        tlen = len(q.normalize_text(text))
        if conf < min_conf:
            continue
        if max_len > 0 and tlen > max_len:
            continue
        if tlen > n_frames:
            continue
        out.append(it)
    return out


def run_pretrain(args, run_dir: Path, model: SSLCTCModel, device: torch.device) -> Path:
    paths = load_unlabeled_paths(Path(args.data_root).resolve())
    if args.ssl_samples > 0 and args.ssl_samples < len(paths):
        rng = random.Random(args.seed + 7)
        paths = rng.sample(paths, args.ssl_samples)
    print(f"[SSL] unlabeled clips: {len(paths)}")

    ds = SSLVideoDataset(paths, n_frames=args.n_frames, frame_size=args.frame_size, crop_mode=args.crop_mode)
    loader = DataLoader(
        ds,
        batch_size=args.ssl_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_ssl,
        persistent_workers=args.num_workers > 0,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.ssl_lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))

    metrics_path = run_dir / "metrics_ssl.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "ssl_loss", "pos_cos", "neg_cos", "epoch_sec", "peak_vram_alloc_gb"])

    best_loss = float("inf")
    best_ckpt = run_dir / "ssl_best.pt"
    for epoch in range(1, args.ssl_epochs + 1):
        model.train()
        ep_start = time.time()
        losses = []
        pos_sims = []
        neg_sims = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step, videos in enumerate(loader, 1):
            if args.ssl_steps_per_epoch > 0 and step > args.ssl_steps_per_epoch:
                break
            videos = videos.to(device, non_blocking=True)
            v1, v2 = augment_views(videos, noise_std=args.ssl_noise_std)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                z1 = model.forward_ssl(v1)
                z2 = model.forward_ssl(v2)
                loss = nt_xent_loss(z1, z2, temperature=args.ssl_temp)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
            with torch.no_grad():
                pos = (z1 * z2).sum(dim=-1).mean().item()
                z2_shift = torch.roll(z2, shifts=1, dims=0)
                neg = (z1 * z2_shift).sum(dim=-1).mean().item()
                pos_sims.append(float(pos))
                neg_sims.append(float(neg))
            if step % args.log_every == 0:
                print(
                    f"[SSL] epoch={epoch} step={step} "
                    f"loss={np.mean(losses[-args.log_every:]):.4f} "
                    f"pos={np.mean(pos_sims[-args.log_every:]):.4f} "
                    f"neg={np.mean(neg_sims[-args.log_every:]):.4f}"
                )
        ep_loss = float(np.mean(losses) if losses else 0.0)
        ep_pos = float(np.mean(pos_sims) if pos_sims else 0.0)
        ep_neg = float(np.mean(neg_sims) if neg_sims else 0.0)
        peak_alloc = (
            torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        )
        print(
            f"[SSL][Epoch {epoch}] loss={ep_loss:.4f} pos_cos={ep_pos:.4f} "
            f"neg_cos={ep_neg:.4f} epoch_sec={time.time()-ep_start:.1f}"
        )
        with metrics_path.open("a", encoding="utf-8", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(
                [
                    epoch,
                    f"{ep_loss:.6f}",
                    f"{ep_pos:.6f}",
                    f"{ep_neg:.6f}",
                    f"{time.time()-ep_start:.2f}",
                    f"{peak_alloc:.3f}",
                ]
            )
        ck = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "ssl_loss": ep_loss,
            "args": vars(args),
        }
        torch.save(ck, run_dir / "ssl_last.pt")
        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save(ck, best_ckpt)
            print(f"[SSL] best checkpoint updated: ssl_loss={best_loss:.4f}")
    return best_ckpt


def run_finetune(
    args,
    run_dir: Path,
    model: SSLCTCModel,
    device: torch.device,
    ssl_ckpt: Path,
    fixed_stoi: Dict[str, int],
) -> None:
    all_items = load_train_items(Path(args.data_root).resolve())
    train_items, val_items = split_train_val(all_items, seed=args.seed, train_frac=0.8)
    stoi = dict(fixed_stoi)
    itos = {i: ch for ch, i in stoi.items()}
    print(f"[CTC] items train={len(train_items)} val={len(val_items)} vocab={len(stoi)}")

    # load SSL weights if available
    if ssl_ckpt and ssl_ckpt.exists():
        ck = torch.load(ssl_ckpt, map_location="cpu")
        model.load_state_dict(ck["model_state"], strict=False)
        print(f"[CTC] loaded SSL checkpoint: {ssl_ckpt}")

    crit = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.ctc_lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))

    # static val loader
    val_filtered = curriculum_filter(
        val_items,
        min_conf=1,
        max_len=args.max_target_len,
        n_frames=args.n_frames,
    )
    val_ds = CTCVideoDataset(
        val_filtered,
        stoi=stoi,
        n_frames=args.n_frames,
        frame_size=args.frame_size,
        crop_mode=args.crop_mode,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.ctc_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_ctc,
        persistent_workers=args.num_workers > 0,
    )
    print(f"[CTC] val filtered={len(val_filtered)}")

    metrics_path = run_dir / "metrics_ctc.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "epoch",
                "stage",
                "train_items",
                "train_loss",
                "val_loss",
                "val_wer_greedy",
                "val_wer_beam_sample",
                "best_val_wer",
                "epoch_sec",
                "peak_vram_alloc_gb",
            ]
        )

    best_wer = float("inf")
    best_epoch = -1
    best_ckpt_path = run_dir / "ctc_best.pt"

    for epoch in range(1, args.ctc_epochs + 1):
        if epoch <= args.stage1_epochs:
            stage_name = "stage1"
            min_conf = args.stage1_min_conf
            max_len = args.stage1_max_len
        elif epoch <= args.stage1_epochs + args.stage2_epochs:
            stage_name = "stage2"
            min_conf = args.stage2_min_conf
            max_len = args.stage2_max_len
        else:
            stage_name = "stage3"
            min_conf = args.stage3_min_conf
            max_len = args.stage3_max_len

        train_stage = curriculum_filter(train_items, min_conf=min_conf, max_len=max_len, n_frames=args.n_frames)
        if len(train_stage) < 128:
            train_stage = curriculum_filter(
                train_items, min_conf=1, max_len=args.max_target_len, n_frames=args.n_frames
            )
            stage_name = f"{stage_name}_fallback"
        train_ds = CTCVideoDataset(
            train_stage,
            stoi=stoi,
            n_frames=args.n_frames,
            frame_size=args.frame_size,
            crop_mode=args.crop_mode,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.ctc_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_ctc,
            persistent_workers=args.num_workers > 0,
        )

        model.train()
        losses = []
        ep_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        print(
            f"[CTC][Epoch {epoch}] stage={stage_name} "
            f"train_items={len(train_stage)} min_conf={min_conf} max_len={max_len}"
        )
        for step, batch in enumerate(train_loader, 1):
            if args.ctc_steps_per_epoch > 0 and step > args.ctc_steps_per_epoch:
                break
            videos, input_lengths, targets_cat, target_lengths, _, _, _ = batch
            videos = videos.to(device, non_blocking=True)
            input_lengths = input_lengths.to(device, non_blocking=True)
            targets_cat = targets_cat.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model.forward_ctc(videos)
            log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
            loss = crit(log_probs, targets_cat, input_lengths, target_lengths)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

            if step % args.log_every == 0:
                print(
                    f"[CTC] epoch={epoch} step={step} "
                    f"loss={np.mean(losses[-args.log_every:]):.4f}"
                )

        train_loss = float(np.mean(losses) if losses else 0.0)
        ev = evaluate_greedy(
            model=model,
            loader=val_loader,
            criterion=crit,
            device=device,
            itos=itos,
            max_batches=args.val_batches,
        )
        beam_wer = float("nan")
        if args.beam_eval_every > 0 and (epoch % args.beam_eval_every == 0):
            beam_wer = evaluate_beam_sample(
                model=model,
                items=val_filtered,
                itos=itos,
                n_frames=args.n_frames,
                frame_size=args.frame_size,
                crop_mode=args.crop_mode,
                batch_size=args.beam_batch_size,
                device=device,
                beam_size=args.beam_size,
                sample_size=args.beam_val_samples,
                seed=args.seed + epoch,
            )
        peak_alloc = (
            torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        )
        print(
            f"[CTC][Epoch {epoch}] train_loss={train_loss:.4f} val_loss={ev.val_loss:.4f} "
            f"val_wer_greedy={ev.val_wer_greedy:.4f} val_wer_beam_sample={beam_wer:.4f}"
        )
        for gt, pr in ev.examples:
            print(f"  gt:   {gt}")
            print(f"  pred: {pr}")

        with metrics_path.open("a", encoding="utf-8", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(
                [
                    epoch,
                    stage_name,
                    len(train_stage),
                    f"{train_loss:.6f}",
                    f"{ev.val_loss:.6f}",
                    f"{ev.val_wer_greedy:.6f}",
                    f"{beam_wer:.6f}" if not math.isnan(beam_wer) else "",
                    f"{min(best_wer, ev.val_wer_greedy):.6f}",
                    f"{time.time()-ep_start:.2f}",
                    f"{peak_alloc:.3f}",
                ]
            )

        ck = {
            "model_state": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "epoch": epoch,
            "best_val_wer": best_wer,
            "best_epoch": best_epoch,
            "args": vars(args),
        }
        torch.save(ck, run_dir / "ctc_last.pt")
        if ev.val_wer_greedy < best_wer:
            best_wer = float(ev.val_wer_greedy)
            best_epoch = epoch
            ck["best_val_wer"] = best_wer
            ck["best_epoch"] = best_epoch
            torch.save(ck, best_ckpt_path)
            save_pred_rows(run_dir / "ctc_best_val_predictions.csv", ev.pred_rows)
            print(f"[CTC] best checkpoint updated: epoch={best_epoch} val_wer={best_wer:.4f}")

    print(f"[CTC] done. best_epoch={best_epoch} best_val_wer={best_wer:.4f}")


def parse_args():
    p = argparse.ArgumentParser("SSL pretrain + CTC finetune with curriculum")
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default="run_ssl_ctc_curriculum")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-frames", type=int, default=96)
    p.add_argument("--frame-size", type=int, default=72)
    p.add_argument("--crop-mode", type=str, default="mouth", choices=["full", "mouth", "face_mouth", "mp_mouth"])
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)

    # SSL
    p.add_argument("--skip-ssl", action="store_true")
    p.add_argument("--ssl-samples", type=int, default=0)
    p.add_argument("--ssl-epochs", type=int, default=8)
    p.add_argument("--ssl-steps-per-epoch", type=int, default=300)
    p.add_argument("--ssl-batch-size", type=int, default=14)
    p.add_argument("--ssl-lr", type=float, default=1e-3)
    p.add_argument("--ssl-temp", type=float, default=0.1)
    p.add_argument("--ssl-noise-std", type=float, default=0.04)

    # CTC
    p.add_argument("--skip-ctc", action="store_true")
    p.add_argument("--ctc-epochs", type=int, default=12)
    p.add_argument("--ctc-steps-per-epoch", type=int, default=0)
    p.add_argument("--ctc-batch-size", type=int, default=10)
    p.add_argument("--ctc-lr", type=float, default=6e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--max-target-len", type=int, default=110)
    p.add_argument("--val-batches", type=int, default=0)

    # curriculum
    p.add_argument("--stage1-epochs", type=int, default=4)
    p.add_argument("--stage2-epochs", type=int, default=4)
    p.add_argument("--stage1-min-conf", type=int, default=5)
    p.add_argument("--stage2-min-conf", type=int, default=3)
    p.add_argument("--stage3-min-conf", type=int, default=1)
    p.add_argument("--stage1-max-len", type=int, default=55)
    p.add_argument("--stage2-max-len", type=int, default=80)
    p.add_argument("--stage3-max-len", type=int, default=110)

    # beam eval
    p.add_argument("--beam-eval-every", type=int, default=2)
    p.add_argument("--beam-size", type=int, default=30)
    p.add_argument("--beam-batch-size", type=int, default=8)
    p.add_argument("--beam-val-samples", type=int, default=280)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    run_dir = Path(args.output_dir).resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # Vocabulary for model head size during init:
    # initialize with full train vocab to keep head size stable.
    all_items = load_train_items(Path(args.data_root).resolve())
    stoi, _ = build_char_vocab(all_items)
    model = SSLCTCModel(vocab_size=len(stoi), hidden=args.hidden).to(device)

    ssl_ckpt = run_dir / "ssl_best.pt"
    if not args.skip_ssl:
        ssl_ckpt = run_pretrain(args, run_dir, model, device)
    if not args.skip_ctc:
        run_finetune(args, run_dir, model, device, ssl_ckpt, fixed_stoi=stoi)


if __name__ == "__main__":
    main()
