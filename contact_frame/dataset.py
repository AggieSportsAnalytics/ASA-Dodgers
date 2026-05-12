"""
dataset.py  –  ContactFrameDataset

Each training sample is a short video clip (T frames) centered on a frame index,
paired with a soft Gaussian label:  label = exp(-0.5 * ((i - contact) / sigma)^2)

This gives the model smooth gradient signal near the contact frame rather than a
single hard spike, and naturally encodes the idea that frames close to contact are
"almost" positives.

Frame extraction
----------------
Frames are extracted to `frames_dir` once and reused on subsequent runs.
The directory layout is:
    frames_dir/
        Aaron Judge/
            frame_0000.jpg
            frame_0001.jpg
            ...
        Adam Duvall/
            ...
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

# Kinetics-400 stats used by R(2+1)D-18
_MEAN = [0.43216, 0.394666, 0.37645]
_STD  = [0.22803, 0.22145,  0.216989]


def _build_transform(frame_size: tuple, augment: bool) -> transforms.Compose:
    ops = [transforms.Resize(frame_size)]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
    return transforms.Compose(ops)


def extract_frames(annotations: list, video_dir: str, frames_dir: str):
    """
    Extract every frame from each video to JPEG.  Skips videos already cached.
    Only needs to run once; subsequent dataset instantiations are fast.
    """
    os.makedirs(frames_dir, exist_ok=True)
    for ann in annotations:
        stem    = Path(ann["video"]).stem
        out_dir = Path(frames_dir) / stem
        expected = ann["total_frames"]

        if out_dir.is_dir():
            n_cached = len(list(out_dir.glob("frame_*.jpg")))
            if n_cached >= expected - 2:   # allow tiny rounding differences
                continue

        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = Path(video_dir) / ann["video"]
        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        with tqdm(total=expected, desc=f"  extracting {stem[:45]}", leave=False) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(
                    str(out_dir / f"frame_{idx:04d}.jpg"),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                )
                idx += 1
                pbar.update(1)
        cap.release()


class ContactFrameDataset(Dataset):
    """
    Parameters
    ----------
    annotations : list of dicts  (from annotations.json)
    video_dir   : folder containing the .mp4 files
    frames_dir  : folder where extracted frames are (or will be) cached
    clip_length : number of frames per clip (T)
    frame_size  : (H, W) to resize frames to
    sigma       : Gaussian half-width in frames for soft labels
    stride      : step between sampled frame indices (use 1 for val/test)
    augment     : apply random flip + color jitter
    """

    def __init__(
        self,
        annotations: list,
        video_dir: str,
        frames_dir: str,
        clip_length: int = 8,
        frame_size: tuple = (112, 112),
        sigma: float = 3.0,
        stride: int = 2,
        augment: bool = True,
    ):
        self.frames_dir  = frames_dir
        self.clip_length = clip_length
        self.frame_size  = frame_size
        self.sigma       = sigma
        self.transform   = _build_transform(frame_size, augment)

        extract_frames(annotations, video_dir, frames_dir)

        # Build flat sample list: (frames_dir_path, total_frames, frame_idx, soft_label)
        self.samples: list[tuple] = []
        for ann in annotations:
            stem    = Path(ann["video"]).stem
            total   = ann["total_frames"]
            contact = ann["contact_frame"]
            fd      = str(Path(frames_dir) / stem)

            for fi in range(0, total, stride):
                label = float(np.exp(-0.5 * ((fi - contact) / sigma) ** 2))
                self.samples.append((fd, total, fi, label))

            # Always include the exact contact frame (may be skipped by stride)
            if contact % stride != 0:
                self.samples.append((fd, total, contact, 1.0))

    # ------------------------------------------------------------------
    def sample_weights(self) -> list[float]:
        """
        Per-sample weights for WeightedRandomSampler.

        Maps soft label [0,1] → weight [0.05, 1.0] so negatives are still
        drawn but contact-region frames are heavily oversampled.
        """
        return [0.05 + 0.95 * label for (_, _, _, label) in self.samples]

    # ------------------------------------------------------------------
    def _load_frame(self, fd: str, total: int, idx: int) -> torch.Tensor:
        idx  = max(0, min(total - 1, idx))
        path = os.path.join(fd, f"frame_{idx:04d}.jpg")
        return self.transform(Image.open(path).convert("RGB"))

    def _make_clip(self, fd: str, total: int, center: int) -> torch.Tensor:
        half    = self.clip_length // 2
        indices = range(center - half, center - half + self.clip_length)
        frames  = [self._load_frame(fd, total, i) for i in indices]
        return torch.stack(frames, dim=1)   # (C, T, H, W)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fd, total, fi, label = self.samples[idx]
        clip  = self._make_clip(fd, total, fi)
        label = torch.tensor(label, dtype=torch.float32)
        return clip, label
