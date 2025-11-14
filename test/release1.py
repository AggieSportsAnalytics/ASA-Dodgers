import cv2
import math
import mediapipe as mp
import numpy as np

VIDEO_IN = "pitch2.mov"
VIDEO_OUT = "./dodgers_annotated.mp4"   # set to None if you don't want to save
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
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    dot = BA[0]*BC[0] + BA[1]*BC[1]
    nba = math.hypot(*BA)
    nbc = math.hypot(*BC)
    if nba == 0 or nbc == 0:
        return None
    cos_t = max(-1.0, min(1.0, dot/(nba*nbc)))
    theta = math.degrees(math.acos(cos_t))
    r = int(0.2 * min(nba, nbc))
    r = max(r, 15)
    angA = math.degrees(math.atan2(-BA[1], BA[0]))
    angC = math.degrees(math.atan2(-BC[1], BC[0]))
    def shortest_sweep(a1, a2):
        da = (a2 - a1) % 360
        if da > 180:
            da -= 360
        return da
    sweep = shortest_sweep(angA, angC)
    cv2.ellipse(img, B, (r, r), 0, angA, angA + sweep, color, 2, cv2.LINE_AA)
    return theta

def arm_angle_to_horizontal(A, B):
    vx = B[0] - A[0]
    vy = B[1] - A[1]
    mag = math.hypot(vx, vy)
    if mag == 0:
        return None
    cos_t = vx / mag
    cos_t = max(-1.0, min(1.0, cos_t))
    return math.degrees(math.acos(cos_t))

def main():
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_IN}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay_ms = max(1, int((1000.0 / fps) / max(1e-6, PLAYBACK_SPEED)))

    writer = None
    if VIDEO_OUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps * PLAYBACK_SPEED if PLAYBACK_SPEED < 1.0 else fps
        out_fps = max(10.0, out_fps)
        writer = cv2.VideoWriter(VIDEO_OUT, fourcc, out_fps, (width, height))

    # --- Storage for release-point analysis ---
    elbow_angles = []
    wrist_heights = []
    shoulder_wrist_dists = []

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

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            angle_text = "None"
            arm_angle_text = "None"

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                r_el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
                r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

                vis_ok = (r_sh.visibility > 0.5 and r_el.visibility > 0.5 and r_wr.visibility > 0.5)
                if vis_ok:
                    # 3D angle + 2D features
                    angle3d = angle_deg(
                        (r_sh.x, r_sh.y, r_sh.z),
                        (r_el.x, r_el.y, r_el.z),
                        (r_wr.x, r_wr.y, r_wr.z),
                    )

                    if angle3d is not None:
                        elbow_angles.append(angle3d)
                        wrist_heights.append(r_wr.y)  # normalized y (smaller = higher)
                        dist = math.sqrt((r_wr.x - r_sh.x)**2 + (r_wr.y - r_sh.y)**2)
                        shoulder_wrist_dists.append(dist)
                    else:
                        elbow_angles.append(np.nan)
                        wrist_heights.append(np.nan)
                        shoulder_wrist_dists.append(np.nan)

                    A = to_px(r_sh, width, height)
                    B = to_px(r_el, width, height)
                    C = to_px(r_wr, width, height)
                    cv2.line(frame_bgr, A, B, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.line(frame_bgr, B, C, (0, 255, 0), 4, cv2.LINE_AA)
                    for p in (A, B, C):
                        cv2.circle(frame_bgr, p, 6, (255, 255, 255), -1, cv2.LINE_AA)
                    draw_angle_arc(frame_bgr, A, B, C)

            # Overlay HUD
            cv2.putText(frame_bgr, f"Frame {frame_idx}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Pitching Analysis", frame_bgr)
            if writer:
                writer.write(frame_bgr)

            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord('q'), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    if writer:
        writer.release()

    # --- Analyze for release frame ---
    elbow_angles = np.array(elbow_angles)
    wrist_heights = np.array(wrist_heights)
    shoulder_wrist_dists = np.array(shoulder_wrist_dists)
    valid = ~np.isnan(elbow_angles)

    if np.sum(valid) < 10:
        print("Not enough valid frames for release analysis.")
        return

    elbow_angles = elbow_angles[valid]
    wrist_heights = wrist_heights[valid]
    shoulder_wrist_dists = shoulder_wrist_dists[valid]
    indices = np.arange(len(elbow_angles))

    # Normalize
    norm_elbow = (elbow_angles - elbow_angles.min()) / (elbow_angles.max() - elbow_angles.min())
    norm_dist = (shoulder_wrist_dists - shoulder_wrist_dists.min()) / (shoulder_wrist_dists.max() - shoulder_wrist_dists.min())
    norm_wrist_height = (wrist_heights.max() - wrist_heights) / (wrist_heights.max() - wrist_heights.min())

    # Filter frames: skip first few, require arm nearly straight
    start_frame = 5
    straight_mask = elbow_angles > 160
    candidate_mask = straight_mask & (indices >= start_frame)

    if np.any(candidate_mask):
        cand_idx = indices[candidate_mask]
        score = (0.6 * norm_wrist_height[candidate_mask]
                 + 0.3 * norm_elbow[candidate_mask]
                 + 0.1 * norm_dist[candidate_mask])
        release_rel = cand_idx[np.argmax(score)]
    else:
        release_rel = np.argmax(norm_wrist_height)

    release_frame = int(release_rel)
    print(f"Estimated RELEASE FRAME ≈ {release_frame}")

if __name__ == "__main__":
    main()
