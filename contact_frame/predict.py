"""
predict.py  –  Run contact-frame prediction on one video or evaluate a full set.

Usage
-----
    cd contact_frame

    # Predict on a single video
    python predict.py --checkpoint checkpoints/best.pt \
                      --video "../Batting/6s_trimmed_videos/Aaron Judge.mp4"

    # Evaluate all annotated videos and print metrics
    python predict.py --checkpoint checkpoints/best.pt \
                      --annotations ../annotations.json \
                      --video_dir "../Batting/6s_trimmed_videos"

    # Write per-video predictions to a JSON file
    python predict.py --checkpoint checkpoints/best.pt \
                      --annotations ../annotations.json \
                      --video_dir "../Batting/6s_trimmed_videos" \
                      --output predictions.json
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import ContactDetector

_MEAN = [0.43216, 0.394666, 0.37645]
_STD  = [0.22803, 0.22145,  0.216989]


def get_device(override: str | None = None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Predictor class ────────────────────────────────────────────────────────────

class ContactPredictor:
    """
    Load a trained ContactDetector checkpoint and run inference on videos.
    Works directly from video files (no pre-extracted frames needed).
    """

    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = get_device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg  = ckpt.get("config", {})

        self.clip_length = cfg.get("clip_length", 8)
        fs = cfg.get("frame_size", 112)
        self.frame_size = (fs, fs) if isinstance(fs, int) else tuple(fs)

        self.model = ContactDetector(pretrained=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize(self.frame_size),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])

        print(f"Loaded  : {checkpoint_path}")
        print(f"Device  : {self.device}")
        print(f"Config  : clip_length={self.clip_length}, frame_size={self.frame_size}")
        if "val_mae" in ckpt:
            print(f"Val MAE : {ckpt['val_mae']:.2f} frames  (epoch {ckpt.get('epoch', '?')})")

    @torch.no_grad()
    def predict(
        self,
        video_path: str,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> dict:
        """
        Score every frame in `video_path` and return the predicted contact frame.

        Returns
        -------
        {
            "video"                   : filename,
            "predicted_contact_frame" : int,
            "confidence"              : float  (sigmoid score at predicted frame),
            "timestamp_seconds"       : float,
            "fps"                     : float,
            "total_frames"            : int,
            "all_scores"              : list[float],
            "top_5"                   : [(frame_idx, score), ...],
        }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Decode all frames into RAM (short ~6 s clips are small)
        raw: list[torch.Tensor] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            raw.append(self.transform(img))
        cap.release()

        total = len(raw)    # actual frame count (may differ from metadata)
        half  = self.clip_length // 2

        def make_clip(center: int) -> torch.Tensor:
            indices = [max(0, min(total - 1, center - half + i))
                       for i in range(self.clip_length)]
            return torch.stack([raw[i] for i in indices], dim=1)  # (C, T, H, W)

        # Score all frames in batches
        scores: list[float] = []
        centers = range(total)
        if show_progress:
            centers = tqdm(range(0, total, batch_size), desc="Scoring")
        else:
            centers = range(0, total, batch_size)

        for start in centers:
            batch = torch.stack([make_clip(c) for c in range(start, min(start + batch_size, total))])
            s = torch.sigmoid(self.model(batch.to(self.device))).cpu().numpy()
            scores.extend(s.tolist())

        best = int(np.argmax(scores))
        top5 = sorted(enumerate(scores), key=lambda x: -x[1])[:5]

        return {
            "video":                   os.path.basename(video_path),
            "predicted_contact_frame": best,
            "confidence":              scores[best],
            "timestamp_seconds":       best / fps,
            "fps":                     fps,
            "total_frames":            total,
            "all_scores":              scores,
            "top_5":                   top5,
        }


# ── Batch evaluation ───────────────────────────────────────────────────────────

def evaluate(
    predictor: ContactPredictor,
    annotations_path: str,
    video_dir: str,
    output_path: str | None = None,
) -> list[dict]:
    with open(annotations_path) as f:
        anns = json.load(f)

    results = []
    for ann in tqdm(anns, desc="Evaluating"):
        vp = os.path.join(video_dir, ann["video"])
        if not os.path.exists(vp):
            print(f"  [skip] {ann['video']} – file not found")
            continue

        result = predictor.predict(vp, show_progress=False)
        err    = abs(result["predicted_contact_frame"] - ann["contact_frame"])
        results.append({
            "video":      ann["video"],
            "gt_frame":   ann["contact_frame"],
            "pred_frame": result["predicted_contact_frame"],
            "error":      err,
            "confidence": result["confidence"],
        })

    # ── Print metrics ──────────────────────────────────────────────────────
    errors = np.array([r["error"] for r in results])

    print("\n" + "=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    print(f"Videos evaluated  : {len(errors)}")
    print(f"Mean error        : {errors.mean():.2f} frames")
    print(f"Median error      : {np.median(errors):.2f} frames")
    print(f"Within ±1 frame   : {(errors <= 1).mean() * 100:.1f}%")
    print(f"Within ±3 frames  : {(errors <= 3).mean() * 100:.1f}%")
    print(f"Within ±5 frames  : {(errors <= 5).mean() * 100:.1f}%")
    print(f"Within ±10 frames : {(errors <= 10).mean() * 100:.1f}%")

    # Worst predictions
    worst = sorted(results, key=lambda r: -r["error"])[:5]
    print("\nWorst predictions:")
    print(f"  {'Video':<45} {'GT':>5} {'Pred':>5} {'Err':>5}")
    print("  " + "-" * 62)
    for r in worst:
        print(f"  {r['video']:<45} {r['gt_frame']:>5} {r['pred_frame']:>5} {r['error']:>5}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nPredictions saved to: {output_path}")

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True, help="Path to best.pt")
    p.add_argument("--video",       help="Single video file to predict")
    p.add_argument("--annotations", help="annotations.json for batch evaluation")
    p.add_argument("--video_dir",   help="Folder with videos (for batch eval)")
    p.add_argument("--output",      help="Save per-video results to this JSON file")
    p.add_argument("--device",      help="cuda / mps / cpu  (auto-detected if omitted)")
    args = p.parse_args()

    predictor = ContactPredictor(args.checkpoint, device=args.device)

    if args.video:
        result = predictor.predict(args.video)
        print(f"\nVideo             : {result['video']}")
        print(f"Contact frame     : {result['predicted_contact_frame']}")
        print(f"Timestamp         : {result['timestamp_seconds']:.3f} s")
        print(f"Confidence        : {result['confidence']:.3f}")
        print(f"Top-5 candidates  : {result['top_5']}")

    elif args.annotations and args.video_dir:
        evaluate(predictor, args.annotations, args.video_dir, output_path=args.output)

    else:
        p.print_help()
        print("\nProvide --video  OR  (--annotations + --video_dir)")


if __name__ == "__main__":
    main()
