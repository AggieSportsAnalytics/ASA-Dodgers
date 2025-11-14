import cv2
import math
import mediapipe as mp

# ==== CONFIGURATION ====
VIDEO_IN = "./dodgers01.mp4"
VIDEO_OUT = "./dodgers_annotated.mp4"   # set to None if you don't want to save
PLAYBACK_SPEED = 0.35  # 1.0 = real-time-ish, 0.5 = half speed, 0.35 ~ slo-mo

# True  = analyze right arm (right-handed pitcher)
# False = analyze left arm (left-handed pitcher)
USE_RIGHT_ARM = False
# =======================

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


# 2D angle between upper arm (shoulder->elbow) and a horizontal line
def arm_angle_to_horizontal(A, B):
    """
    Angle (in degrees) between the segment A->B (shoulder->elbow)
    and a rightward horizontal line. A and B are (x,y) pixel coords.
    Returns angle in [0, 180].
    """
    vx = B[0] - A[0]
    vy = B[1] - A[1]
    mag = math.hypot(vx, vy)
    if mag == 0:
        return None

    # horizontal vector is (1, 0), so dot = vx
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

    # Slow-motion display delay (larger delay = slower)
    delay_ms = max(1, int((1000.0 / fps) / max(1e-6, PLAYBACK_SPEED)))

    writer = None
    if VIDEO_OUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps * PLAYBACK_SPEED if PLAYBACK_SPEED < 1.0 else fps
        # Keep file smooth-ish even if very slow; floor at 10 fps
        out_fps = max(10.0, out_fps)
        writer = cv2.VideoWriter(VIDEO_OUT, fourcc, out_fps, (width, height))

    # Label for HUD/prints based on boolean
    ARM_LABEL = "Right" if USE_RIGHT_ARM else "Left"

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
            arm_angle_text = "None"

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # === Choose which arm to use based on USE_RIGHT_ARM ===
                if USE_RIGHT_ARM:
                    sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
                    wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
                else:
                    sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    el = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
                    wr = lm[mp_pose.PoseLandmark.LEFT_WRIST]
                # =====================================================

                vis_ok = (sh.visibility > 0.5 and el.visibility > 0.5 and wr.visibility > 0.5)
                if vis_ok:
                    # 3D elbow angle for numeric readout
                    angle3d = angle_deg(
                        (sh.x, sh.y, sh.z),
                        (el.x, el.y, el.z),
                        (wr.x, wr.y, wr.z),
                    )
                    if angle3d is not None and not math.isnan(angle3d):
                        angle_text = f"{angle3d:6.2f}"

                    # 2D pixel coordinates
                    A = to_px(sh, width, height)  # shoulder
                    B = to_px(el, width, height)  # elbow
                    C = to_px(wr, width, height)  # wrist

                    # Segments (shoulder->elbow, elbow->wrist)
                    cv2.line(frame_bgr, A, B, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.line(frame_bgr, B, C, (0, 255, 0), 4, cv2.LINE_AA)

                    # Joints
                    for p in (A, B, C):
                        cv2.circle(frame_bgr, p, 6, (255, 255, 255), -1, cv2.LINE_AA)

                    # Angle arc at the elbow
                    theta2d = draw_angle_arc(frame_bgr, A, B, C, color=(0, 255, 255))
                    # If you want the 2D elbow angle label on the arc, uncomment:
                    # if theta2d is not None:
                    #     cv2.putText(frame_bgr, f"{theta2d:.1f}°",
                    #                 (B[0] + 10, B[1] - 10),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                    # Horizontal reference line from shoulder
                    horiz_len = 150  # pixels; tweak to taste
                    H = (A[0] + horiz_len, A[1])
                    cv2.line(frame_bgr, A, H, (255, 0, 0), 2, cv2.LINE_AA)

                    # Angle between upper arm and horizontal
                    arm_angle = arm_angle_to_horizontal(A, B)
                    if arm_angle is not None:
                        arm_angle_text = f"{arm_angle:6.2f}"
                        # If you want it near the shoulder, uncomment:
                        # cv2.putText(frame_bgr, f"Arm:{arm_angle:.1f}°",
                        #             (A[0] + 10, A[1] - 10),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            # Overlay HUD text
            cv2.putText(frame_bgr, f"{ARM_LABEL} Elbow: {angle_text}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 220, 255), 2, cv2.LINE_AA)

            cv2.putText(frame_bgr, f"{ARM_LABEL} Arm vs Horiz: {arm_angle_text}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2, cv2.LINE_AA)

            cv2.putText(frame_bgr, f"t={t_sec:6.3f}s  slow={PLAYBACK_SPEED}x",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

            # Terminal print
            print(f"[{t_sec:8.3f}s] {ARM_LABEL.lower()}_elbow_angle: {angle_text}  arm_vs_horiz: {arm_angle_text}")

            # Show + (optional) write
            cv2.imshow("Pitching - Elbow Angle + Arm Angle", frame_bgr)
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
