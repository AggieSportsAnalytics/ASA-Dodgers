"""
annotate.py  –  Ball-bat contact frame annotator
=================================================
Keyboard controls
-----------------
  ← / → (arrow keys)   :  ±1 frame
  , / .                 :  ±1 frame  (reliable cross-platform alternative)
  [ / ]                 :  ±10 frames
  { / }                 :  ±100 frames
  Space or Enter        :  MARK contact frame → save → next video
  U                     :  Clear annotation for current video
  N                     :  Next video  (without marking)
  P                     :  Previous video
  Q  or  Esc            :  Save and quit

Annotations saved to:  annotations.json  (same folder as this script)
"""

import cv2
import json
import os
import sys
import glob
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "Batting", "6s_trimmed_videos")
ANNOT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "annotations.json")
DISPLAY_W  = 1280
DISPLAY_H  = 720
WINDOW     = "Annotator  |  Space=mark  N=next  P=prev  Q=quit"

# ── Arrow key codes (vary by platform / OpenCV build) ─────────────────────────
LEFT_KEYS  = {2, 81, 65361}   # macOS, Linux, Windows variants
RIGHT_KEYS = {3, 83, 65363}

# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_annotations() -> dict:
    if os.path.exists(ANNOT_FILE):
        with open(ANNOT_FILE) as f:
            data = json.load(f)
        # Support both list-of-dicts and dict-of-dicts from previous runs
        if isinstance(data, list):
            return {item["video"]: item for item in data}
        return data
    return {}


def save_annotations(annots: dict):
    records = sorted(annots.values(), key=lambda r: r["video"])
    with open(ANNOT_FILE, "w") as f:
        json.dump(records, f, indent=2)


def find_videos() -> list[str]:
    paths = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) +
                   glob.glob(os.path.join(VIDEO_DIR, "*.mkv")) +
                   glob.glob(os.path.join(VIDEO_DIR, "*.avi")))
    return paths


def video_key(path: str) -> str:
    return os.path.basename(path)

# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_frame(raw: "cv2.Mat", idx: int, total: int, fps: float,
               contact, video_path: str, vid_num: int, vid_total: int):
    h, w = raw.shape[:2]
    scale = min(DISPLAY_W / w, DISPLAY_H / h)
    frame = cv2.resize(raw, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA)
    dh, dw = frame.shape[:2]

    # --- top bar ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (dw, 56), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    name = os.path.splitext(os.path.basename(video_path))[0]
    ts   = idx / fps if fps else 0

    cv2.putText(frame,
                f"[{vid_num}/{vid_total}]  {name}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"Frame {idx} / {total - 1}     {ts:.3f}s @ {int(fps)}fps",
                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                (180, 180, 180), 1, cv2.LINE_AA)

    if contact is not None:
        badge_txt = f"CONTACT  frame {contact}"
        badge_col = (0, 255, 80) if contact == idx else (0, 210, 255)
        cv2.putText(frame, badge_txt,
                    (dw - 270, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    badge_col, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "not annotated",
                    (dw - 185, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                    (80, 80, 220), 1, cv2.LINE_AA)

    # --- bottom progress bar ---
    bar_h  = 8
    prog   = int(dw * idx / max(total - 1, 1))
    cv2.rectangle(frame, (0, dh - bar_h), (dw, dh), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, dh - bar_h), (prog, dh), (50, 180, 50), -1)
    if contact is not None:
        cx = int(dw * contact / max(total - 1, 1))
        cv2.rectangle(frame, (cx - 2, dh - bar_h - 2),
                      (cx + 2, dh), (0, 255, 80), -1)

    return frame


# ── Per-video annotation loop ─────────────────────────────────────────────────

def annotate_one(cap, video_path: str, vid_num: int, vid_total: int,
                 annots: dict) -> tuple[str, dict]:
    """
    Returns (action, annots) where action is 'next' | 'prev' | 'quit'.
    """
    key_name  = video_key(video_path)
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    contact   = annots.get(key_name, {}).get("contact_frame")

    # Start at the already-annotated frame, or frame 0
    idx = contact if contact is not None else 0
    idx = max(0, min(total - 1, idx))

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, raw_frame = cap.read()
    if not ret:
        print(f"[warn] Cannot read {key_name}")
        return "next", annots

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, DISPLAY_W, DISPLAY_H + 32)
    cv2.createTrackbar("Frame", WINDOW, idx, total - 1, lambda _: None)

    last_tb = idx
    need_draw = True

    while True:
        # ── trackbar drag ──────────────────────────────────────────────────
        tb = cv2.getTrackbarPos("Frame", WINDOW)
        if tb != last_tb:
            idx = tb
            last_tb = tb
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, raw_frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, raw_frame = cap.read()
            need_draw = True

        if need_draw:
            display = draw_frame(raw_frame, idx, total, fps,
                                 contact, video_path, vid_num, vid_total)
            cv2.imshow(WINDOW, display)
            need_draw = False

        key = cv2.waitKey(30) & 0xFFFF   # 16-bit to catch extended codes

        if key == 0xFFFF:
            continue

        # ── frame navigation ───────────────────────────────────────────────
        step = None

        if key in LEFT_KEYS or key == ord(','):
            step = -1
        elif key in RIGHT_KEYS or key == ord('.'):
            step = 1
        elif key == ord('['):
            step = -10
        elif key == ord(']'):
            step = 10
        elif key == ord('{'):
            step = -100
        elif key == ord('}'):
            step = 100

        # ── actions ────────────────────────────────────────────────────────
        elif key in (ord(' '), 13):          # Space / Enter → mark
            contact = idx
            annots[key_name] = {
                "video":         key_name,
                "total_frames":  total,
                "fps":           fps,
                "contact_frame": contact,
                "annotated_at":  datetime.now().isoformat(timespec="seconds"),
            }
            save_annotations(annots)
            print(f"  ✓  {key_name}  →  frame {contact}")
            cv2.destroyWindow(WINDOW)
            return "next", annots

        elif key in (ord('u'), ord('U')):    # unmark
            contact = None
            annots.pop(key_name, None)
            save_annotations(annots)
            print(f"  ✗  cleared  {key_name}")
            need_draw = True
            continue

        elif key in (ord('n'), ord('N')):    # skip
            cv2.destroyWindow(WINDOW)
            return "next", annots

        elif key in (ord('p'), ord('P')):    # previous
            cv2.destroyWindow(WINDOW)
            return "prev", annots

        elif key in (ord('q'), ord('Q'), 27):  # quit
            cv2.destroyWindow(WINDOW)
            return "quit", annots

        else:
            continue

        # ── apply step ─────────────────────────────────────────────────────
        if step is not None:
            new_idx = max(0, min(total - 1, idx + step))
            if new_idx != idx:
                idx = new_idx
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, raw_frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, raw_frame = cap.read()
                cv2.setTrackbarPos("Frame", WINDOW, idx)
                last_tb = idx
                need_draw = True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    videos = find_videos()
    if not videos:
        print(f"No videos found in:\n  {VIDEO_DIR}")
        sys.exit(1)

    annots = load_annotations()
    done   = sum(1 for v in videos if video_key(v) in annots)

    # ── Print summary table ────────────────────────────────────────────────
    print(f"\nVideo dir : {VIDEO_DIR}")
    print(f"Videos    : {len(videos)}  ({done} annotated, {len(videos)-done} remaining)\n")
    print(f"{'#':<4} {'Frames':>7} {'FPS':>5}  {'Contact':>8}  Name")
    print("─" * 75)
    for i, vp in enumerate(videos):
        cap  = cv2.VideoCapture(vp)
        n    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps  = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        rec    = annots.get(video_key(vp), {})
        marked = str(rec["contact_frame"]) if "contact_frame" in rec else "—"
        name   = os.path.splitext(os.path.basename(vp))[0]
        print(f"{i+1:<4} {n:>7} {fps:>5.0f}  {marked:>8}  {name}")
    print()

    # ── Find first unannotated video ───────────────────────────────────────
    start = 0
    for i, vp in enumerate(videos):
        if video_key(vp) not in annots:
            start = i
            break

    vid_idx = start
    while 0 <= vid_idx < len(videos):
        vp  = videos[vid_idx]
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"[skip] Cannot open: {vp}")
            vid_idx += 1
            continue

        action, annots = annotate_one(cap, vp, vid_idx + 1, len(videos), annots)
        cap.release()

        if action == "next":
            vid_idx += 1
        elif action == "prev":
            vid_idx = max(0, vid_idx - 1)
        elif action == "quit":
            break

    # ── Final stats ────────────────────────────────────────────────────────
    done = sum(1 for v in videos if video_key(v) in annots)
    print(f"\nDone.  {done}/{len(videos)} annotated.")
    print(f"Saved → {ANNOT_FILE}\n")


if __name__ == "__main__":
    main()
