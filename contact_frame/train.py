"""
train.py  –  Train ContactDetector to predict ball-bat contact frames.

Usage
-----
    cd contact_frame
    python train.py                          # uses defaults below
    python train.py --epochs 50 --lr 5e-5   # custom config

Outputs
-------
    checkpoints/best.pt   – lowest val MAE checkpoint
    checkpoints/last.pt   – final epoch checkpoint
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from dataset import ContactFrameDataset
from model import ContactDetector

# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Full-video evaluation ─────────────────────────────────────────────────────

_MEAN = [0.43216, 0.394666, 0.37645]
_STD  = [0.22803, 0.22145,  0.216989]
_BASE_TFM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


@torch.no_grad()
def score_video(model, ann, frames_dir, device, clip_length, frame_size, batch_size=32):
    """
    Score every frame in a single video and return the argmax as the
    predicted contact frame, plus the per-frame score array.
    """
    stem  = Path(ann["video"]).stem
    fd    = Path(frames_dir) / stem
    total = ann["total_frames"]
    half  = clip_length // 2

    tfm = transforms.Compose([transforms.Resize(frame_size)] + _BASE_TFM.transforms)

    def load(idx):
        idx = max(0, min(total - 1, idx))
        return tfm(Image.open(fd / f"frame_{idx:04d}.jpg").convert("RGB"))

    # Load all frames once, then build clips on-the-fly
    all_frames = [load(i) for i in range(total)]

    def make_clip(center):
        indices = [max(0, min(total - 1, center - half + i)) for i in range(clip_length)]
        return torch.stack([all_frames[i] for i in indices], dim=1)

    scores = []
    for start in range(0, total, batch_size):
        batch = torch.stack([make_clip(c) for c in range(start, min(start + batch_size, total))])
        s = torch.sigmoid(model(batch.to(device))).cpu().numpy()
        scores.extend(s.tolist())

    return int(np.argmax(scores)), scores


def evaluate_full(model, val_anns, frames_dir, device, clip_length, frame_size):
    """
    Run full-video evaluation on the val set.
    Returns metrics dict and per-video result list.
    """
    model.eval()
    errors, results = [], []

    for ann in tqdm(val_anns, desc="  full-video eval", leave=False):
        pred, scores = score_video(model, ann, frames_dir, device, clip_length, frame_size)
        err = abs(pred - ann["contact_frame"])
        errors.append(err)
        results.append({
            "video":      ann["video"],
            "gt":         ann["contact_frame"],
            "pred":       pred,
            "error":      err,
            "confidence": scores[pred],
        })

    e = np.array(errors)
    return {
        "mean_error":   float(e.mean()),
        "median_error": float(np.median(e)),
        "within_1":     float((e <= 1).mean() * 100),
        "within_3":     float((e <= 3).mean() * 100),
        "within_5":     float((e <= 5).mean() * 100),
        "within_10":    float((e <= 10).mean() * 100),
    }, results


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations",    default="../annotations.json")
    p.add_argument("--video_dir",      default="../Batting/6s_trimmed_videos")
    p.add_argument("--frames_dir",     default="../Batting/cached_frames")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--clip_length",    type=int,   default=8,
                   help="Frames per clip fed to the 3D CNN")
    p.add_argument("--frame_size",     type=int,   default=112,
                   help="Resize each frame to this square size")
    p.add_argument("--sigma",          type=float, default=3.0,
                   help="Gaussian soft-label width in frames")
    p.add_argument("--stride",         type=int,   default=2,
                   help="Frame stride for training samples (1 = all frames)")
    p.add_argument("--val_frac",       type=float, default=0.15,
                   help="Fraction of videos held out for validation")
    p.add_argument("--eval_every",     type=int,   default=5,
                   help="Run full-video eval every N epochs")
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"\nDevice: {device}")
    if device.type == "mps":
        print("  (MPS detected – set --num_workers 0 if DataLoader hangs)")

    # ── Load annotations ──────────────────────────────────────────────────
    with open(args.annotations) as f:
        all_anns = json.load(f)
    print(f"Annotations: {len(all_anns)} videos\n")

    # ── Train / val split by video ────────────────────────────────────────
    shuffled = all_anns.copy()
    random.shuffle(shuffled)
    n_val      = max(2, int(len(shuffled) * args.val_frac))
    val_anns   = shuffled[:n_val]
    train_anns = shuffled[n_val:]
    print(f"Split: {len(train_anns)} train / {len(val_anns)} val")
    print(f"Val videos: {[a['video'] for a in val_anns]}\n")

    frame_size = (args.frame_size, args.frame_size)

    # ── Datasets ──────────────────────────────────────────────────────────
    print("Building datasets (extracting frames if not cached)...")
    train_ds = ContactFrameDataset(
        train_anns, args.video_dir, args.frames_dir,
        clip_length=args.clip_length,
        frame_size=frame_size,
        sigma=args.sigma,
        stride=args.stride,
        augment=True,
    )
    val_ds = ContactFrameDataset(
        val_anns, args.video_dir, args.frames_dir,
        clip_length=args.clip_length,
        frame_size=frame_size,
        sigma=args.sigma,
        stride=1,           # use every frame for validation clips
        augment=False,
    )
    print(f"Train samples: {len(train_ds):,}   Val samples: {len(val_ds):,}\n")

    # WeightedRandomSampler over-samples the contact region
    sampler = WeightedRandomSampler(
        weights=train_ds.sample_weights(),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=(device.type != "cpu"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type != "cpu"),
    )

    # ── Model / optimizer / loss ──────────────────────────────────────────
    model     = ContactDetector(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()   # soft-label regression: MSE(sigmoid(logit), gaussian_label)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(args.checkpoint_dir, "best.pt")
    last_ckpt = os.path.join(args.checkpoint_dir, "last.pt")
    best_mae  = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for clips, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs} train", leave=False):
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = torch.sigmoid(model(clips))
            loss  = criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        # --- val clip-level loss ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for clips, labels in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{args.epochs} val  ", leave=False):
                clips, labels = clips.to(device), labels.to(device)
                preds  = torch.sigmoid(model(clips))
                val_loss += criterion(preds, labels).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # --- full-video evaluation (expensive, run periodically) ---
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics, per_vid = evaluate_full(
                model, val_anns, args.frames_dir, device,
                args.clip_length, frame_size,
            )
            mae = metrics["mean_error"]
            print(
                f"         video-eval  MAE={mae:.1f}f  "
                f"±1={metrics['within_1']:.0f}%  "
                f"±3={metrics['within_3']:.0f}%  "
                f"±5={metrics['within_5']:.0f}%  "
                f"±10={metrics['within_10']:.0f}%"
            )

            if mae < best_mae:
                best_mae = mae
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_mae":          mae,
                    "metrics":          metrics,
                    "config": {
                        "clip_length": args.clip_length,
                        "frame_size":  args.frame_size,
                        "sigma":       args.sigma,
                    },
                }, best_ckpt)
                print(f"         ✓ Saved best checkpoint  (MAE={mae:.1f}f)")

        # Always save the latest checkpoint
        torch.save({
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "clip_length": args.clip_length,
                "frame_size":  args.frame_size,
                "sigma":       args.sigma,
            },
        }, last_ckpt)

    print(f"\nTraining complete.  Best MAE: {best_mae:.1f} frames")
    print(f"Best checkpoint : {best_ckpt}")
    print(f"Last checkpoint : {last_ckpt}")


if __name__ == "__main__":
    main()
