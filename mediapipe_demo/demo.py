# save as pose_right_elbow_overlay.py
import cv2
import math
import mediapipe as mp

VIDEO_IN = "./pitching.mp4"
VIDEO_OUT = "./pitching_annotated.mp4"   # set to None if you don't want to save
PLAYBACK_SPEED = 0.35  # 1.0 = real-time-ish, 0.5 = half speed, 0.35 ~ slo-mo

mp_pose = mp.solutions.pose

def angle_deg(a, b, c):
    """Angle ABC at point b in degrees. a,b,c are (x,y,z) floats."""
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c

    bax, bay, baz = ax - bx, ay - by, az - bz
    bcx, bcy, bcz = cx - bx, cy - by, cz - bz

    dot = bax * bcx + bay * bcy + baz * bcz
    nba = math.sqrt(bax*bax + bay*bay + baz*baz)
    nbc = math.sqrt(bcx*bcx + bcy*bcy + bcz*bcz)
    if nba == 0 or nbc == 0:
        return None
    cos_t = max(-1.0, min(1.0, dot / (nba * nbc)))
    return math.degrees(math.acos(cos_t))

def to_px(landmark, w, h):
    """Convert normalized landmark to pixel coordinates."""
    return int(landmark.x * w), int(landmark.y * h)

def draw_angle_arc(img, A, B, C, color=(0, 255, 255)):
    """
    Draw an angle arc at B between BA and BC.
    A,B,C are 2D pixel tuples (x,y). Returns the angle in degrees.
    """
    # vectors from B
    import numpy as np
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])

    # angle
    dot = BA[0]*BC[0] + BA[1]*BC[1]
    nba = math.hypot(*BA)
    nbc = math.hypot(*BC)
    if nba == 0 or nbc == 0:
        return None
    cos_t = max(-1.0, min(1.0, dot/(nba*nbc)))
    theta = math.degrees(math.acos(cos_t))

    # arc radius ~ 20% of the shorter arm
    r = int(0.2 * min(nba, nbc))
    r = max(r, 15)  # minimum so it's visible

    # angles for ellipse need degrees w.r.t. +x axis; use atan2 to get directions
    angA = math.degrees(math.atan2(-BA[1], BA[0]))  # invert y because image coords
    angC = math.degrees(math.atan2(-BC[1], BC[0]))

    # Compute sweep direction the shorter way
    def shortest_sweep(a1, a2):
        da = (a2 - a1) % 360
        if da > 180:
            da -= 360
        return da
    sweep = shortest_sweep(angA, angC)

    cv2.ellipse(img, B, (r, r), 0, angA, angA + sweep, color, 2, cv2.LINE_AA)
    return theta

def main():
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_IN}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Slow-motion display delay (larger delay = slower)
    delay_ms = max(1, int((1000.0 / fps) / max(1e-6, PLAYBACK_SPEED)))

    writer = None
    if VIDEO_OUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps * PLAYBACK_SPEED if PLAYBACK_SPEED < 1.0 else fps
        # Keep file smooth-ish even if very slow; floor at 10 fps
        out_fps = max(10.0, out_fps)
        writer = cv2.VideoWriter(VIDEO_OUT, fourcc, out_fps, (width, height))

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1
            t_sec = frame_idx / fps

            # Inference
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            angle_text = "None"
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                r_el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
                r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

                vis_ok = (r_sh.visibility > 0.5 and r_el.visibility > 0.5 and r_wr.visibility > 0.5)
                if vis_ok:
                    # 3D angle for numeric readout
                    angle3d = angle_deg(
                        (r_sh.x, r_sh.y, r_sh.z),
                        (r_el.x, r_el.y, r_el.z),
                        (r_wr.x, r_wr.y, r_wr.z),
                    )
                    if angle3d is not None and not math.isnan(angle3d):
                        angle_text = f"{angle3d:6.2f}°"

                    # Draw arm lines + angle arc in 2D pixels
                    A = to_px(r_sh, width, height)
                    B = to_px(r_el, width, height)
                    C = to_px(r_wr, width, height)

                    # Segments (shoulder->elbow, elbow->wrist)
                    cv2.line(frame_bgr, A, B, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.line(frame_bgr, B, C, (0, 255, 0), 4, cv2.LINE_AA)

                    # Joints
                    for p in (A, B, C):
                        cv2.circle(frame_bgr, p, 6, (255, 255, 255), -1, cv2.LINE_AA)

                    # Angle arc + on-arc label
                    theta2d = draw_angle_arc(frame_bgr, A, B, C, color=(0, 255, 255))
                    if theta2d is not None:
                        cv2.putText(frame_bgr, f"{theta2d:.1f}°",
                                    (B[0] + 10, B[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # Overlay HUD text
            cv2.putText(frame_bgr, f"Right Elbow: {angle_text}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"t={t_sec:6.3f}s  slow={PLAYBACK_SPEED}x",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

            # Terminal print
            print(f"[{t_sec:8.3f}s] right_elbow_angle: {angle_text}")

            # Show + (optional) write
            cv2.imshow("Pitching - Right Elbow Angle", frame_bgr)
            if writer:
                writer.write(frame_bgr)

            # quit with q / ESC
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord('q'), 27):
                break

    if writer:
        writer.release()
        print(f"Saved annotated video to: {VIDEO_OUT}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
