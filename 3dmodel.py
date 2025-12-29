import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend on macOS worker threads

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import os
import time
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt

# ========================================
# MediaPipe initialization
# ========================================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ========================================
# Video input
# ========================================

cap = cv2.VideoCapture('pitch13.mov')

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

arm_vertical_angles = []
elbow_extension_angles = []
wrist_heights = []
shoulder_to_wrist_distances = []
wrist_velocities = []
frames_data = []
all_landmarks = []

frame_count = 0
prev_wrist_pos = None
target_person_id = None
initial_shoulder_center = None
body_parts = {
    'torso': [], 'l_upper_arm': [], 'l_forearm': [], 'r_upper_arm': [],
    'r_forearm': [], 'l_thigh': [], 'l_shin': [], 'r_thigh': [], 'r_shin': [],
    'shoulder_width': []
}

print("Processing video - Pass 1: Extracting metrics...")

consecutive_detections = []
lock_on_threshold = 3
max_frames_to_wait = 50

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frames_data.append(frame.copy())
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        current_shoulder_center = np.array([
            (lm[11].x + lm[11].y) / 2,
            (lm[12].y + lm[12].y) / 2
        ])

        avg_visibility = np.mean(
            [lm[i].visibility for i in [11, 12, 13, 14, 15, 16, 23, 24]]
        )
        shoulder_width = np.linalg.norm(
            np.array([lm[11].x, lm[11].y]) -
            np.array([lm[12].x, lm[12].y])
        )
        closeness_score = shoulder_width * avg_visibility

        # Lock onto pitcher
        if target_person_id is None:
            left_shoulder = np.array([lm[11].x, lm[11].y])
            left_wrist = np.array([lm[15].x, lm[15].y])
            right_shoulder = np.array([lm[12].x, lm[12].y])
            right_wrist = np.array([lm[16].x, lm[16].y])

            left_ext = np.linalg.norm(left_wrist - left_shoulder)
            right_ext = np.linalg.norm(right_wrist - right_shoulder)
            max_ext = max(left_ext, right_ext)

            if max_ext > 0.2 and avg_visibility > 0.4:
                consecutive_detections.append({
                    'frame': frame_count,
                    'position': current_shoulder_center.copy(),
                    'extension': max_ext,
                    'closeness': closeness_score
                })

                if len(consecutive_detections) >= lock_on_threshold:
                    positions = np.array(
                        [d['position'] for d in consecutive_detections[-lock_on_threshold:]]
                    )
                    max_distance = np.max([
                        np.linalg.norm(positions[i] - positions[i + 1])
                        for i in range(len(positions) - 1)
                    ])

                    if max_distance < 0.1:
                        target_person_id = consecutive_detections[0]['frame']
                        initial_shoulder_center = consecutive_detections[0]['position'].copy()
                        print(f"  Locked onto pitcher at frame {target_person_id} (arm movement detected)")
                        consecutive_detections = []
            else:
                consecutive_detections = []

            if frame_count >= max_frames_to_wait and target_person_id is None:
                if avg_visibility > 0.4 and shoulder_width > 0.05:
                    target_person_id = frame_count
                    initial_shoulder_center = current_shoulder_center.copy()
                    print(f"  Locked onto closest person at frame {frame_count} (fallback)")

        if target_person_id is not None:
            if initial_shoulder_center is not None:
                distance_from_initial = np.linalg.norm(
                    current_shoulder_center - initial_shoulder_center
                )
                is_same_person = distance_from_initial < 0.25 and avg_visibility > 0.3
            else:
                is_same_person = True

            if is_same_person:
                frame_lm = np.zeros((33, 3), dtype=np.float32)
                for i in range(33):
                    frame_lm[i, 0] = lm[i].x
                    frame_lm[i, 1] = lm[i].y
                    frame_lm[i, 2] = lm[i].z

                all_landmarks.append(frame_lm)
                left_shoulder = np.array([lm[11].x, lm[11].y])
                left_elbow = np.array([lm[13].x, lm[13].y])
                left_wrist = np.array([lm[15].x, lm[15].y])
                left_hip = np.array([lm[23].x, lm[23].y])

                right_shoulder = np.array([lm[12].x, lm[12].y])
                right_elbow = np.array([lm[14].x, lm[14].y])
                right_wrist = np.array([lm[16].x, lm[16].y])
                right_hip = np.array([lm[24].x, lm[24].y])

                initial_shoulder_center = 0.95 * initial_shoulder_center + 0.05 * current_shoulder_center

                left_ext = np.linalg.norm(left_wrist - left_shoulder)
                right_ext = np.linalg.norm(right_wrist - right_shoulder)

                if left_ext > right_ext:
                    shoulder, elbow, wrist = left_shoulder, left_elbow, left_wrist
                    hip = left_hip
                else:
                    shoulder, elbow, wrist = right_shoulder, right_elbow, right_wrist
                    hip = right_hip

                torso_size = np.linalg.norm(shoulder - hip)
                if torso_size == 0:
                    torso_size = 1.0

                torso_vector = hip - shoulder
                arm_vector = wrist - shoulder
                dot_product = np.dot(arm_vector, torso_vector)
                magnitudes = np.linalg.norm(arm_vector) * np.linalg.norm(torso_vector)
                arm_vertical_angle = np.degrees(
                    np.arccos(
                        np.clip(dot_product / (magnitudes + 1e-10), -1.0, 1.0)
                    )
                )
                arm_vertical_angles.append(arm_vertical_angle)

                v1 = shoulder - elbow
                v2 = wrist - elbow
                elbow_angle = np.degrees(
                    np.arccos(
                        np.clip(
                            np.dot(v1, v2) /
                            ((np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-10),
                            -1.0, 1.0
                        )
                    )
                )
                elbow_extension_angles.append(elbow_angle)

                wrist_heights.append(wrist[1])
                shoulder_to_wrist_distances.append(np.linalg.norm(shoulder - wrist))

                if prev_wrist_pos is not None:
                    raw_displacement = np.linalg.norm(wrist - prev_wrist_pos)
                    norm_vel = raw_displacement / torso_size
                    wrist_velocities.append(norm_vel)
                else:
                    wrist_velocities.append(0.0)
                prev_wrist_pos = wrist.copy()
            else:
                lm = None
        else:
            lm = None
    else:
        lm = None

    if lm is None:
        arm_vertical_angles.append(None)
        all_landmarks.append(None)
        elbow_extension_angles.append(None)
        wrist_heights.append(None)
        shoulder_to_wrist_distances.append(None)
        wrist_velocities.append(0.0)
        prev_wrist_pos = None

cap.release()

def clean_data(data):
    arr = np.array(data, dtype=float)
    mask = np.isnan(arr)
    if mask.all():
        return np.zeros_like(arr)
    if not mask.any():
        return arr
    valid_indices = np.flatnonzero(~mask)
    if len(valid_indices) > 0:
        arr[mask] = np.interp(np.flatnonzero(mask), valid_indices, arr[~mask])
    return arr

arm_vertical_clean = clean_data(arm_vertical_angles)
elbow_extension_clean = clean_data(elbow_extension_angles)
wrist_heights_clean = clean_data(wrist_heights)
distances_clean = clean_data(shoulder_to_wrist_distances)
velocities_clean = np.array(wrist_velocities)

if len(arm_vertical_clean) == 0 or target_person_id is None:
    print("\n" + "=" * 60)
    print("WARNING: No person detected with arm movement in the video!")
    print("=" * 60)
    num_frames = len(frames_data)
    arm_vertical_clean = np.zeros(num_frames)
    elbow_extension_clean = np.zeros(num_frames)
    wrist_heights_clean = np.zeros(num_frames)
    distances_clean = np.zeros(num_frames)
    velocities_clean = np.zeros(num_frames)
    release_frame = num_frames // 2 if num_frames > 0 else 0
else:
    print(f"  Successfully tracked pitcher from frame {target_person_id} onwards")

start_frame = 5

print("\n" + "=" * 60)
print("BODY PROPORTIONS")
print("=" * 60)
avg_body = {k: np.median(v) if v else 0.15 for k, v in body_parts.items()}
for k, v in avg_body.items():
    print(f"  {k}: {v:.4f}")

# ========================================
# Release detection
# ========================================

print("\n" + "=" * 60)
print("ANALYZING PITCH PHASES (SCALE INVARIANT)")
print("=" * 60)

window_len = 7
if len(velocities_clean) <= window_len:
    window_len = 3 if len(velocities_clean) > 3 else 1

if window_len > 1:
    smooth_velocity = savgol_filter(velocities_clean, window_length=window_len, polyorder=2)
    smooth_elbow = savgol_filter(elbow_extension_clean, window_length=window_len, polyorder=2)
else:
    smooth_velocity = velocities_clean
    smooth_elbow = elbow_extension_clean

max_vel_ref = np.max(smooth_velocity) if np.max(smooth_velocity) > 0 else 1.0
candidate_peaks, _ = find_peaks(smooth_velocity, height=max_vel_ref * 0.3, distance=5)

best_score = -1.0
release_frame = 0

print(f"Candidates found at frames: {candidate_peaks}")

for p in candidate_peaks:
    if p < 5 or p >= len(frames_data) - 5:
        continue

    vel_score = smooth_velocity[p] / max_vel_ref
    current_ext = smooth_elbow[p]
    ext_score = 0.0
    if current_ext > 130:
        ext_score = (current_ext - 130) / 40.0
    ext_score = np.clip(ext_score, 0, 1)

    relative_time = p / len(frames_data)
    time_score = 1.0 if relative_time > 0.4 else 0.5

    accel_score = 0.5
    if p > 2 and p < len(smooth_velocity) - 2:
        vel_before = smooth_velocity[p - 2:p].mean()
        vel_after = smooth_velocity[p + 1:p + 3].mean()
        if smooth_velocity[p] >= vel_before:
            accel_score = 1.0
        elif vel_after > smooth_velocity[p]:
            accel_score = 0.3

    total_score = (
        vel_score * 0.40 +
        ext_score * 0.40 +
        time_score * 0.05 +
        accel_score * 0.15
    )

    print(f"  Frame {p}: Vel={vel_score:.2f}, Ext={current_ext:.1f}deg, Accel={accel_score:.2f} -> Score: {total_score:.3f}")

    if total_score > best_score:
        best_score = total_score
        release_frame = p

if release_frame > 0:
    peak_vel = smooth_velocity[release_frame]
    for offset in range(0, min(4, release_frame)):
        check_frame = release_frame - offset
        if smooth_velocity[check_frame] >= 0.95 * peak_vel and smooth_elbow[check_frame] > 140:
            release_frame = check_frame
            print(
                f"  Adjusted release earlier to frame {release_frame} "
                f"(velocity still at {smooth_velocity[check_frame]/peak_vel*100:.1f}% of peak)"
            )
            break

if best_score < 0.3 or len(candidate_peaks) == 0:
    print("  ⚠ Low confidence in peaks. Using fallback logic.")
    valid_mask = smooth_elbow > 140
    if np.any(valid_mask):
        masked_vel = smooth_velocity.copy()
        masked_vel[~valid_mask] = 0
        release_frame = np.argmax(masked_vel)
        print(f"  Fallback: Selected max velocity frame {release_frame} where arm > 140deg")
    else:
        release_frame = np.argmax(smooth_velocity)
        print(f"  Fallback: Selected global max velocity frame {release_frame}")

print("=" * 60 + "\n")

norm_elbow = (elbow_extension_clean - elbow_extension_clean.min()) / (
    elbow_extension_clean.max() - elbow_extension_clean.min() + 1e-6
)
norm_velocity = (velocities_clean - velocities_clean.min()) / (
    velocities_clean.max() - velocities_clean.min() + 1e-6
)

# ========================================
# Pass 2: annotated video
# ========================================

print("\nProcessing video - Pass 2: Creating annotated video...")

metrics_panel_height = 300
output_height = height + metrics_panel_height
output_width = width

output_path = 'pitch_analysis_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
pose2 = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

for idx, frame in enumerate(frames_data):
    extended_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    extended_frame[0:height, 0:width] = frame

    results = pose2.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        current_shoulder_center = np.array([
            (lm[11].x + lm[12].x) / 2,
            (lm[11].y + lm[12].y) / 2
        ])

        avg_visibility = np.mean(
            [lm[i].visibility for i in [11, 12, 13, 14, 15, 16, 23, 24]]
        )

        is_target_person = False
        if initial_shoulder_center is not None and target_person_id is not None:
            distance_from_initial = np.linalg.norm(
                current_shoulder_center - initial_shoulder_center
            )
            is_target_person = (distance_from_initial < 0.25 and avg_visibility > 0.3)

        if is_target_person:
            mp_drawing.draw_landmarks(
                extended_frame[0:height, 0:width],
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            h, w = height, width

            left_shoulder_px = (int(lm[11].x * w), int(lm[11].y * h))
            left_elbow_px = (int(lm[13].x * w), int(lm[13].y * h))
            left_wrist_px = (int(lm[15].x * w), int(lm[15].y * h))
            left_hip_px = (int(lm[23].x * w), int(lm[23].y * h))

            right_shoulder_px = (int(lm[12].x * w), int(lm[12].y * h))
            right_elbow_px = (int(lm[14].x * w), int(lm[14].y * h))
            right_wrist_px = (int(lm[16].x * w), int(lm[16].y * h))
            right_hip_px = (int(lm[24].x * w), int(lm[24].y * h))

            left_ext = np.linalg.norm(
                np.array(left_wrist_px) - np.array(left_shoulder_px)
            )
            right_ext = np.linalg.norm(
                np.array(right_wrist_px) - np.array(right_shoulder_px)
            )

            if left_ext > right_ext:
                shoulder_px, elbow_px, wrist_px = left_shoulder_px, left_elbow_px, left_wrist_px
                hip_px = left_hip_px
            else:
                shoulder_px, elbow_px, wrist_px = right_shoulder_px, right_elbow_px, right_wrist_px
                hip_px = right_hip_px

            cv2.line(extended_frame, shoulder_px, elbow_px, (255, 255, 0), 4)
            cv2.line(extended_frame, elbow_px, wrist_px, (255, 255, 0), 4)
            cv2.circle(extended_frame, wrist_px, 8, (0, 255, 255), -1)

            torso_vec = np.array(
                [hip_px[0] - shoulder_px[0], hip_px[1] - shoulder_px[1]]
            )
            torso_length = np.linalg.norm(torso_vec)
            if torso_length > 0:
                torso_unit = torso_vec / torso_length
                vertical_start = (
                    int(shoulder_px[0] - torso_unit[0] * 100),
                    int(shoulder_px[1] - torso_unit[1] * 100)
                )
                vertical_end = (
                    int(shoulder_px[0] + torso_unit[0] * 200),
                    int(shoulder_px[1] + torso_unit[1] * 200)
                )
            else:
                vertical_start = (shoulder_px[0], shoulder_px[1] - 100)
            vertical_end = (shoulder_px[0], shoulder_px[1] + 200)

            cv2.line(extended_frame, vertical_start, vertical_end, (0, 165, 255), 3)
            cv2.line(extended_frame, shoulder_px, hip_px, (0, 165, 255), 4)
            cv2.circle(extended_frame, shoulder_px, 6, (0, 165, 255), -1)
            cv2.circle(extended_frame, hip_px, 6, (0, 165, 255), -1)

    overlay = extended_frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, extended_frame, 0.5, 0, extended_frame)

    cv2.putText(
        extended_frame, f"Frame: {idx}/{len(frames_data) - 1}",
        (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 255), 2
    )

    if idx == release_frame:
        cv2.rectangle(
            extended_frame,
            (0, height // 2 - 60),
            (width, height // 2 + 60),
            (0, 255, 0), -1
        )
        cv2.putText(
            extended_frame, "*** RELEASE POINT ***",
            (width // 2 - 280, height // 2 + 15),
            cv2.FONT_HERSHEY_DUPLEX, 1.8,
            (0, 0, 0), 5
        )

    cv2.rectangle(
        extended_frame, (0, height),
        (width, output_height), (20, 20, 20), -1
    )

    cv2.rectangle(
        extended_frame, (0, height),
        (width, height + 50), (40, 40, 60), -1
    )
    title = "PITCH BIOMECHANICS ANALYSIS"
    cv2.putText(
        extended_frame, title, (20, height + 35),
        cv2.FONT_HERSHEY_DUPLEX, 1.0,
        (255, 255, 255), 2
    )

    progress_text = f"Progress: {(idx + 1) / len(frames_data) * 100:.0f}%"
    cv2.putText(
        extended_frame, progress_text, (width - 200, height + 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (100, 255, 100), 2
    )

    y_start = height + 70
    line_height = 35

    metrics = [
        (f"Arm Angle from Torso:    {arm_vertical_clean[idx]:.1f}", (100, 200, 255)),
        (f"Elbow Extension:         {elbow_extension_clean[idx]:.1f}", (100, 255, 255)),
        (f"Wrist Height:            {wrist_heights_clean[idx]:.4f}", (255, 220, 150)),
        (f"Arm Extension:           {distances_clean[idx]:.4f}", (150, 255, 150)),
        (f"Rel Velocity (Body Len): {velocities_clean[idx]:.4f}", (255, 150, 255)),
        (f"Velocity Score:          {norm_velocity[idx]:.3f}", (255, 255, 255))
    ]

    for i, (text, color) in enumerate(metrics):
        cv2.putText(
            extended_frame, text,
            (30, y_start + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    frames_to_release = release_frame - idx
    if -5 <= frames_to_release <= 15:
        if frames_to_release > 0:
            msg = f"RELEASE IN {frames_to_release} FRAMES"
            color = (100, 255, 255)
        elif frames_to_release == 0:
            msg = "RELEASE NOW!"
            color = (100, 255, 100)
        else:
            msg = f"{-frames_to_release} frames past release"
            color = (200, 200, 255)

        cv2.rectangle(
            extended_frame,
            (width - 420, height + 60),
            (width - 20, height + 110),
            (60, 60, 80), -1
        )
        cv2.putText(
            extended_frame, msg,
            (width - 400, height + 95),
            cv2.FONT_HERSHEY_DUPLEX, 0.8,
            color, 2
        )

    bar_width = width - 60
    bar_x, bar_y, bar_height = 30, output_height - 40, 20

    cv2.rectangle(
        extended_frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (60, 60, 60), -1
    )

    progress = int((idx / (len(frames_data) - 1)) * bar_width)
    cv2.rectangle(
        extended_frame,
        (bar_x, bar_y),
        (bar_x + progress, bar_y + bar_height),
        (100, 255, 100), -1
    )

    release_x = bar_x + int((release_frame / (len(frames_data) - 1)) * bar_width)
    cv2.line(
        extended_frame,
        (release_x, bar_y - 5),
        (release_x, bar_y + bar_height + 5),
        (255, 255, 0), 4
    )
    cv2.circle(
        extended_frame,
        (release_x, bar_y + bar_height // 2),
        8, (255, 200, 0), -1
    )

    out.write(extended_frame)

    if idx % 30 == 0:
        print(f"  Processing frame {idx}/{len(frames_data) - 1}")

out.release()
pose2.close()
print(f"\n✓ Annotated video saved as: {output_path}")

# ========================================
# MuJoCo: smoothed landmarks, root pinned above ground
# ========================================

# ========================================
# MuJoCo: smoothed landmarks, root pinned above ground
# ========================================

# ========================================
# MuJoCo: smoothed landmarks, root pinned above ground
# ========================================

print("\n" + "=" * 60)
print("MUJOCO KINEMATIC RECONSTRUCTION (MEDIA PIPE DRIVEN, SMOOTHED)")
print("=" * 60)

# Remove invalid frames
filtered_landmarks = [lm for lm in all_landmarks if lm is not None]

if len(filtered_landmarks) == 0:
    raise RuntimeError("No valid landmarks for MuJoCo playback")

landmarks = np.array(filtered_landmarks)   # shape (T, 33, 3)
T, J, C = landmarks.shape

# ---- 1) Low-pass filter landmarks over time ----
def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

# Sampling rate ~ video fps
fs = max(fps, 1)
cutoff = min(8.0, 0.45 * fs)  # 6–8 Hz typical for fast sports motion[web:129][web:131]

smoothed_landmarks = landmarks.copy()
for j in range(J):
    for c in range(3):
        series = landmarks[:, j, c]
        smoothed_landmarks[:, j, c] = butter_lowpass_filter(series, cutoff=cutoff, fs=fs, order=2)

# Optionally blend raw and smoothed to keep sharpness
SMOOTH_BLEND = 0.7
smoothed_landmarks = SMOOTH_BLEND * smoothed_landmarks + (1.0 - SMOOTH_BLEND) * landmarks

# -------------------------
# MJCF (PURE KINEMATIC)
# -------------------------

mjcf = """
<mujoco model="pitcher">
    compiler angle="radian" coordinate="local"/>
    <option timestep="0.01" gravity="0 0 0"/>

    <worldbody>
        <!-- Pelvis at about hip height -->
        <body name="pelvis" pos="0 0 1.0">
            <joint name="root_x"   type="slide" axis="1 0 0"/>
            <joint name="root_y"   type="slide" axis="0 1 0"/>
            <joint name="root_z"   type="slide" axis="0 0 1"/>
            <joint name="root_yaw" type="hinge" axis="0 0 1"/>

            <!-- Pelvis sphere -->
            <geom type="sphere" size="0.09" rgba="0.7 0.6 0.6 1"/>

            <!-- Torso: vertical capsule, no pitch joint -->
            <body name="torso" pos="0 0 0.0">
                <!-- remove torso_pitch joint so torso stays upright -->
                <!-- <joint name="torso_pitch" type="hinge" axis="1 0 0"/> -->
                <!-- vertical torso from pelvis up -->
                <geom type="capsule" fromto="0 0 0 0 0 0.55" size="0.09"/>

                <!-- Head directly above torso, also vertical -->
                <body name="head" pos="0 0 0.65">
                    <!-- remove neck_pitch or keep at 0 -->
                    <!-- <joint name="neck_pitch" type="hinge" axis="1 0 0"/> -->
                    <geom type="sphere" size="0.10" rgba="0.8 0.7 0.7 1"/>
                </body>

                <!-- Right arm anchor: small lateral offset at shoulder height -->
                <body name="upper_arm_r" pos="0 0.18 0.45">
                    <joint name="shoulder_r_yaw"   type="hinge" axis="0 0 1"/>
                    <joint name="shoulder_r_pitch" type="hinge" axis="1 0 0"/>
                    <geom type="capsule" fromto="0 0 0 0.30 0 0" size="0.045"/>
                    <site name="site_shoulder_r" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>

                    <body name="forearm_r" pos="0.30 0 0">
                        <joint name="elbow_r" type="hinge" axis="0 1 0"/>
                        <geom type="capsule" fromto="0 0 0 0.28 0 0" size="0.04"/>
                        <site name="site_elbow_r" pos="0 0 0" size="0.01" rgba="0 1 0 1"/>

                        <body name="hand_r" pos="0.28 0 0">
                            <geom type="sphere" size="0.035"/>
                            <site name="site_wrist_r" pos="0 0 0" size="0.01" rgba="0 0 1 1"/>
                        </body>
                    </body>
                </body>

                <!-- Left arm: mirror on -y -->
                <body name="upper_arm_l" pos="0 -0.18 0.45">
                    <joint name="shoulder_l_yaw"   type="hinge" axis="0 0 1"/>
                    <joint name="shoulder_l_pitch" type="hinge" axis="1 0 0"/>
                    <geom type="capsule" fromto="0 0 0 -0.30 0 0" size="0.045"/>

                    <body name="forearm_l" pos="-0.30 0 0">
                        <joint name="elbow_l" type="hinge" axis="0 1 0"/>
                        <geom type="capsule" fromto="0 0 0 -0.28 0 0" size="0.04"/>
                    </body>
                </body>
            </body>

            <!-- Legs: keep as you already have (they are vertical and look correct) -->
            <body name="thigh_r" pos="0 0.10 -0.05">
                <joint name="hip_r" type="hinge" axis="1 0 0"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.45" size="0.06"/>
                <site name="site_hip_r" pos="0 0 0" size="0.01" rgba="1 0 1 1"/>

                <body name="shin_r" pos="0 0 -0.45">
                    <joint name="knee_r" type="hinge" axis="1 0 0"/>
                    <geom type="capsule" fromto="0 0 0 0 0 -0.45" size="0.05"/>
                    <site name="site_knee_r" pos="0 0 0" size="0.01" rgba="0 1 1 1"/>

                    <body name="foot_r" pos="0 0 -0.45">
                        <geom type="sphere" size="0.04"/>
                        <site name="site_ankle_r" pos="0 0 0" size="0.01" rgba="1 1 0 1"/>
                    </body>
                </body>
            </body>

            <body name="thigh_l" pos="0 -0.10 -0.05">
                <joint name="hip_l" type="hinge" axis="1 0 0"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.45" size="0.06"/>

                <body name="shin_l" pos="0 0 -0.45">
                    <joint name="knee_l" type="hinge" axis="1 0 0"/>
                    <geom type="capsule" fromto="0 0 0 0 0 -0.45" size="0.05"/>
                </body>
            </body>
        </body>

    </worldbody>
</mujoco>

"""

xml_path = "kinematic_pitcher.xml"
with open(xml_path, "w") as f:
    f.write(mjcf)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Joint index lookup
qadr = {model.joint(i).name: model.jnt_qposadr[i] for i in range(model.njnt)}

# Site ids
site_ids = {
    "shoulder_r": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "site_shoulder_r"),
    "elbow_r":    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "site_elbow_r"),
    "wrist_r":    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "site_wrist_r"),
    "hip_r":      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "site_hip_r"),
    "knee_r":     mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "site_knee_r"),
    "ankle_r":    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "site_ankle_r"),
}

# Which DOFs we allow IK to adjust (throwing arm + right leg)
ik_dofs = [
    #qadr["torso_pitch"],
    qadr["shoulder_r_yaw"],
    qadr["shoulder_r_pitch"],
    qadr["elbow_r"],
    qadr["hip_r"],
    qadr["knee_r"],
]

# Helper
def angle_between(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))

def mp_to_mj(pt):
    return np.array([pt[0] * 2.0, -pt[2] * 2.0, (1.0 - pt[1]) * 1.8])

viewer = mujoco.viewer.launch_passive(model, data)
viewer.cam.azimuth = 90
viewer.cam.elevation = -10
viewer.cam.distance = 4.5
viewer.cam.lookat[:] = [0, 0, 1.2]

T = smoothed_landmarks.shape[0]
frame_idx = 0
dt = 1.0 / max(fps, 1)

print(f"Playing {T} frames with IK on throwing arm + right leg...")

# IK parameters
IK_ITERS = 10          # iterations per frame
ALPHA = 0.5            # step size
TARGET_WEIGHT_ARM = 1.0
TARGET_WEIGHT_LEG = 0.6

while viewer.is_running():
    lm = smoothed_landmarks[frame_idx]

    # Target world positions from MediaPipe
    LS = mp_to_mj(lm[11])
    RS = mp_to_mj(lm[12])
    LE = mp_to_mj(lm[13])
    RE = mp_to_mj(lm[14])
    LW = mp_to_mj(lm[15])
    RW = mp_to_mj(lm[16])
    LH = mp_to_mj(lm[23])
    RH = mp_to_mj(lm[24])
    LK = mp_to_mj(lm[25])
    RK = mp_to_mj(lm[26])
    LA = mp_to_mj(lm[27])
    RA = mp_to_mj(lm[28])

    pelvis = (LH + RH) / 2
    pelvis[2] = max(0.3, pelvis[2])

    data.qpos[qadr["root_x"]] = pelvis[0]
    data.qpos[qadr["root_y"]] = pelvis[1]
    data.qpos[qadr["root_z"]] = pelvis[2]

    # IK targets (world frame)
    targets = {
        "shoulder_r": RS,
        "elbow_r":    RE,
        "wrist_r":    RW,
        "hip_r":      RH,
        "knee_r":     RK,
        "ankle_r":    RA,
    }
    weights = {
        "shoulder_r": TARGET_WEIGHT_ARM,
        "elbow_r":    TARGET_WEIGHT_ARM,
        "wrist_r":    TARGET_WEIGHT_ARM,
        "hip_r":      TARGET_WEIGHT_LEG,
        "knee_r":     TARGET_WEIGHT_LEG,
        "ankle_r":    TARGET_WEIGHT_LEG,
    }

    # Run a few IK iterations
    for _ in range(IK_ITERS):
        mujoco.mj_forward(model, data)

        # Small regularization toward previous pose can be added if you store qprev

        for name, target in targets.items():
            sid = site_ids[name]

            # Current site position
            site_pos = data.site_xpos[sid].copy()
            err = target - site_pos

            if np.linalg.norm(err) < 1e-4:
                continue

            # Site Jacobian (world) wrt all DOFs
            Jpos = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, Jpos, None, sid)

            # Restrict to selected DOFs
            J_reduced = Jpos[:, ik_dofs]   # 3 x ndof
            # Jacobian transpose control: dq = alpha * J^T * err
            dq = ALPHA * J_reduced.T @ err * weights[name]

            # Apply to qpos (note qpos index == dof index here for 1-DoF joints)
            for i, dof in enumerate(ik_dofs):
                data.qpos[dof] += dq[i]

    mujoco.mj_forward(model, data)
    viewer.sync()

    frame_idx = (frame_idx + 1) % T
    time.sleep(dt)

print("\nViewer closed.")
if os.path.exists(xml_path):
    os.remove(xml_path)


# ========================================
# Plots
# ========================================

fig, axes = plt.subplots(6, 1, figsize=(12, 12))

max_vertical_angle_idx = np.argmax(arm_vertical_clean)
max_elbow_extension_idx = np.argmax(elbow_extension_clean)
min_wrist_height_idx = np.argmin(wrist_heights_clean)
max_distance_idx = np.argmax(distances_clean)

axes[0].plot(arm_vertical_clean, label='Arm Angle from Torso', color='orange')
axes[0].axvline(max_vertical_angle_idx, color='red', linestyle='--', alpha=0.5, label=f'Max Angle ({max_vertical_angle_idx})')
axes[0].axvline(release_frame, color='green', linestyle='--', linewidth=2, label=f'Release ({release_frame})')
axes[0].set_ylabel('Angle (°)')
axes[0].set_title('Arm Angle from Torso (Vertical Reference)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(elbow_extension_clean, label='Elbow Extension', color='cyan')
axes[1].axhline(180, color='gray', linestyle=':', alpha=0.5, label='Fully Straight')
axes[1].axvline(max_elbow_extension_idx, color='red', linestyle='--', alpha=0.5, label=f'Max Extension ({max_elbow_extension_idx})')
axes[1].axvline(release_frame, color='green', linestyle='--', linewidth=2, label=f'Release ({release_frame})')
axes[1].set_ylabel('Angle (°)')
axes[1].set_title('Elbow Extension')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(wrist_heights_clean, label='Wrist Height', color='purple')
axes[2].axvline(min_wrist_height_idx, color='red', linestyle='--', alpha=0.5, label=f'Highest ({min_wrist_height_idx})')
axes[2].axvline(release_frame, color='green', linestyle='--', linewidth=2, label=f'Release ({release_frame})')
axes[2].set_ylabel('Y Position')
axes[2].set_title('Wrist Height (Lower = Higher Position)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

axes[3].plot(distances_clean, label='Shoulder-Wrist Distance', color='lime')
axes[3].axvline(max_distance_idx, color='red', linestyle='--', alpha=0.5, label=f'Max ({max_distance_idx})')
axes[3].axvline(release_frame, color='green', linestyle='--', linewidth=2, label=f'Release ({release_frame})')
axes[3].set_ylabel('Distance')
axes[3].set_title('Arm Extension')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

axes[4].plot(velocities_clean, label='Wrist Velocity', color='brown')
axes[4].axvline(release_frame, color='green', linestyle='--', linewidth=3, label=f'Release (Peak Velocity) ({release_frame})')
axes[4].set_ylabel('Velocity')
axes[4].set_title('Wrist Velocity')
axes[4].legend()
axes[4].grid(True, alpha=0.3)

axes[5].plot(norm_velocity, label='Normalized Wrist Velocity', color='green')
axes[5].axvline(release_frame, color='green', linestyle='--', linewidth=2, label=f'Release (Peak) ({release_frame})')
axes[5].set_ylabel('Normalized Score')
axes[5].set_xlabel('Frame')
axes[5].set_title('Normalized Wrist Velocity (Release Indicator)')
axes[5].legend()
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pitch_analysis_plots.png', dpi=150, bbox_inches='tight')
print(f"✓ Analysis plots saved as: pitch_analysis_plots.png")

print("\n" + "=" * 60)
print("PITCH RELEASE FRAME DETECTION RESULTS")
print("=" * 60)
print(f"\nPredicted Release Frame: {release_frame}")
print(f"  - Wrist Velocity: {velocities_clean[release_frame]:.4f}")
print(f"  - Arm Angle from Torso: {arm_vertical_clean[release_frame]:.2f}°")
print(f"  - Elbow Extension: {elbow_extension_clean[release_frame]:.2f}°")
print(f"  - Wrist Height: {wrist_heights_clean[release_frame]:.4f}")
print(f"  - Arm Extension: {distances_clean[release_frame]:.4f}")
print("=" * 60)
