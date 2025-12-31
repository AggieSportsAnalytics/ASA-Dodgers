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
from scipy.spatial.transform import Rotation as R

# ========================================
# MediaPipe initialization (enhanced for 3D)
# ========================================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    smooth_segmentation=True,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
mp_drawing = mp.solutions.drawing_utils

# Enhanced temporal smoothing with Kalman-like filtering
class LandmarkSmoother3D:
    def __init__(self, alpha_pos=0.6, alpha_vel=0.8, dt=1/30):
        self.alpha_pos = alpha_pos
        self.alpha_vel = alpha_vel
        self.prev_pos = None
        self.prev_vel = None
        self.dt = dt
        
    def __call__(self, lm_array):
        if self.prev_pos is None:
            self.prev_pos = lm_array.copy()
            self.prev_vel = np.zeros_like(lm_array)
            return self.prev_pos
        
        # Predict
        predicted_pos = self.prev_pos + self.prev_vel * self.dt
        
        # Update
        pos_error = lm_array - predicted_pos
        smoothed_pos = predicted_pos + self.alpha_pos * pos_error
        
        # Update velocity
        vel = (smoothed_pos - self.prev_pos) / self.dt
        self.prev_vel = self.alpha_vel * vel + (1 - self.alpha_vel) * self.prev_vel
        self.prev_pos = smoothed_pos.copy()
        
        return smoothed_pos

lm_smoother = LandmarkSmoother3D(alpha_pos=0.7, alpha_vel=0.8)

# ========================================
# Video input
# ========================================

cap = cv2.VideoCapture('pitch4.mov')

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
lock_on_threshold = 2
max_frames_to_wait = 30

# NEW: Store 3D landmarks with better depth estimation
landmarks_3d_history = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frames_data.append(frame.copy())
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # Get 3D landmarks with depth
        current_shoulder_center_3d = np.array([
            (lm[11].x + lm[12].x) / 2,
            (lm[11].y + lm[12].y) / 2,
            (lm[11].z + lm[12].z) / 2  # Use z for depth
        ])
        
        current_shoulder_center = current_shoulder_center_3d[:2]  # 2D for compatibility

        avg_visibility = np.mean(
            [lm[i].visibility for i in [11, 12, 13, 14, 15, 16, 23, 24]]
        )
        shoulder_width = np.linalg.norm(
            np.array([lm[11].x, lm[11].y]) -
            np.array([lm[12].x, lm[12].y])
        )
        closeness_score = shoulder_width * avg_visibility

        # Lock onto pitcher (more lenient)
        if target_person_id is None:
            left_shoulder = np.array([lm[11].x, lm[11].y, lm[11].z])
            left_wrist = np.array([lm[15].x, lm[15].y, lm[15].z])
            right_shoulder = np.array([lm[12].x, lm[12].y, lm[12].z])
            right_wrist = np.array([lm[16].x, lm[16].y, lm[16].z])

            left_ext = np.linalg.norm(left_wrist - left_shoulder)
            right_ext = np.linalg.norm(right_wrist - right_shoulder)
            max_ext = max(left_ext, right_ext)

            if max_ext > 0.15 and avg_visibility > 0.3:
                consecutive_detections.append({
                    'frame': frame_count,
                    'position': current_shoulder_center_3d.copy(),
                    'extension': max_ext,
                    'closeness': closeness_score
                })

                if len(consecutive_detections) >= lock_on_threshold:
                    positions = np.array(
                        [d['position'] for d in consecutive_detections[-lock_on_threshold:]]
                    )
                    if len(positions) > 1:
                        max_distance = np.max([
                            np.linalg.norm(positions[i] - positions[i + 1])
                            for i in range(len(positions) - 1)
                        ])
                    else:
                        max_distance = 0

                    if max_distance < 0.15:
                        target_person_id = consecutive_detections[0]['frame']
                        initial_shoulder_center = consecutive_detections[0]['position'][:2].copy()
                        print(f"  Locked onto pitcher at frame {target_person_id} (arm movement detected)")
                        consecutive_detections = []
            else:
                consecutive_detections = []

            if frame_count >= max_frames_to_wait and target_person_id is None:
                if avg_visibility > 0.3 and shoulder_width > 0.04:
                    target_person_id = frame_count
                    initial_shoulder_center = current_shoulder_center.copy()
                    print(f"  Locked onto closest person at frame {frame_count} (fallback)")

        if target_person_id is not None:
            if initial_shoulder_center is not None:
                distance_from_initial = np.linalg.norm(
                    current_shoulder_center - initial_shoulder_center
                )
                is_same_person = distance_from_initial < 0.5 and avg_visibility > 0.15
            else:
                is_same_person = True

            if is_same_person:
                # Store full 3D landmarks with depth
                frame_lm = np.zeros((33, 4), dtype=np.float32)  # x, y, z, visibility
                for i in range(33):
                    frame_lm[i, 0] = lm[i].x
                    frame_lm[i, 1] = lm[i].y
                    frame_lm[i, 2] = lm[i].z  # Depth information
                    frame_lm[i, 3] = lm[i].visibility

                # Apply 3D smoothing
                frame_lm_smoothed = lm_smoother(frame_lm[:, :3])  # Only smooth x,y,z
                frame_lm[:, :3] = frame_lm_smoothed
                
                all_landmarks.append(frame_lm)
                landmarks_3d_history.append(frame_lm[:, :3].copy())

                # Update initial position (2D only for compatibility)
                initial_shoulder_center = 0.8 * initial_shoulder_center + 0.2 * current_shoulder_center
                
                # Get 3D positions for analysis
                left_shoulder = np.array([lm[11].x, lm[11].y, lm[11].z])
                left_elbow = np.array([lm[13].x, lm[13].y, lm[13].z])
                left_wrist = np.array([lm[15].x, lm[15].y, lm[15].z])
                left_hip = np.array([lm[23].x, lm[23].y, lm[23].z])

                right_shoulder = np.array([lm[12].x, lm[12].y, lm[12].z])
                right_elbow = np.array([lm[14].x, lm[14].y, lm[14].z])
                right_wrist = np.array([lm[16].x, lm[16].y, lm[16].z])
                right_hip = np.array([lm[24].x, lm[24].y, lm[24].z])

                # Calculate 3D distances
                left_ext = np.linalg.norm(left_wrist - left_shoulder)
                right_ext = np.linalg.norm(right_wrist - right_shoulder)

                if left_ext > right_ext:
                    shoulder, elbow, wrist = left_shoulder, left_elbow, left_wrist
                    hip = left_hip
                else:
                    shoulder, elbow, wrist = right_shoulder, right_elbow, right_wrist
                    hip = right_hip

                # 3D torso vector
                torso_vector = hip - shoulder
                arm_vector = wrist - shoulder
                
                # Calculate 3D angle between arm and torso
                dot_product = np.dot(arm_vector, torso_vector)
                magnitudes = np.linalg.norm(arm_vector) * np.linalg.norm(torso_vector)
                arm_vertical_angle = np.degrees(
                    np.arccos(
                        np.clip(dot_product / (magnitudes + 1e-10), -1.0, 1.0)
                    )
                )
                arm_vertical_angles.append(arm_vertical_angle)

                # 3D elbow angle
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

                # 3D velocity calculation
                if prev_wrist_pos is not None:
                    raw_displacement = np.linalg.norm(wrist - prev_wrist_pos)
                    torso_size = np.linalg.norm(torso_vector)
                    if torso_size > 0:
                        norm_vel = raw_displacement / torso_size
                    else:
                        norm_vel = raw_displacement
                    wrist_velocities.append(norm_vel)
                else:
                    wrist_velocities.append(0.0)
                prev_wrist_pos = wrist.copy()
            else:
                all_landmarks.append(None)
                arm_vertical_angles.append(None)
                elbow_extension_angles.append(None)
                wrist_heights.append(None)
                shoulder_to_wrist_distances.append(None)
                wrist_velocities.append(0.0)
                prev_wrist_pos = None
                landmarks_3d_history.append(None)
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
        landmarks_3d_history.append(None)

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

# Process 3D landmarks for body proportions
if landmarks_3d_history and any(lm is not None for lm in landmarks_3d_history):
    valid_landmarks = [lm for lm in landmarks_3d_history if lm is not None]
    if valid_landmarks:
        avg_landmarks = np.mean(valid_landmarks, axis=0)
        
        # Calculate 3D limb lengths
        torso_len = np.linalg.norm(avg_landmarks[11] - avg_landmarks[23])  # Shoulder to hip
        upper_arm_len = np.linalg.norm(avg_landmarks[11] - avg_landmarks[13])  # Shoulder to elbow
        forearm_len = np.linalg.norm(avg_landmarks[13] - avg_landmarks[15])  # Elbow to wrist
        thigh_len = np.linalg.norm(avg_landmarks[23] - avg_landmarks[25])  # Hip to knee
        shin_len = np.linalg.norm(avg_landmarks[25] - avg_landmarks[27])  # Knee to ankle
        
        print(f"  Torso length: {torso_len:.4f}")
        print(f"  Upper arm length: {upper_arm_len:.4f}")
        print(f"  Forearm length: {forearm_len:.4f}")
        print(f"  Thigh length: {thigh_len:.4f}")
        print(f"  Shin length: {shin_len:.4f}")

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
# Pass 2: annotated video (same as before)
# ========================================

print("\nProcessing video - Pass 2: Creating annotated video...")

metrics_panel_height = 300
output_height = height + metrics_panel_height
output_width = width

output_path = 'pitch_analysis_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
pose2 = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                     smooth_landmarks=True, min_detection_confidence=0.6,
                     min_tracking_confidence=0.6)

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
# MuJoCo: Enhanced 3D model with better IK
# ========================================

# ========================================
# MuJoCo: Enhanced 3D model with better IK - LOOPING ANIMATION
# ========================================

print("\n" + "=" * 60)
print("MUJOCO FULL BODY RECONSTRUCTION (ENHANCED 3D)")
print("=" * 60)

# Filter and prepare landmarks for MuJoCo - INCLUDING ALL FRAMES
print("\n  Preparing landmarks for animation...")

# Create arrays for all frames
T_total = len(frames_data)
landmarks_3d_full = np.zeros((T_total, 33, 3))

# First pass: fill with available landmarks
valid_count = 0
for i, lm in enumerate(all_landmarks):
    if lm is not None:
        landmarks_3d_full[i] = lm[:, :3]  # Extract x, y, z
        valid_count += 1
    else:
        # Mark as invalid with NaN
        landmarks_3d_full[i] = np.nan

print(f"  Valid landmarks: {valid_count}/{T_total} frames")

# Interpolate missing frames
print("  Interpolating missing frames...")
for joint_idx in range(33):
    for coord_idx in range(3):
        series = landmarks_3d_full[:, joint_idx, coord_idx]
        
        # Create mask for valid values
        valid_mask = ~np.isnan(series)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) >= 2:
            # Interpolate missing values
            series_interp = np.interp(
                np.arange(len(series)),
                valid_indices,
                series[valid_indices]
            )
            landmarks_3d_full[:, joint_idx, coord_idx] = series_interp
        elif len(valid_indices) == 1:
            # Fill with single valid value
            landmarks_3d_full[:, joint_idx, coord_idx] = series[valid_indices[0]]
        # If no valid values, leave as is (will be 0)

# Apply 3D smoothing
def smooth_3d_trajectory(trajectory, window_size=7):
    """Smooth trajectory while preserving edges for fast movements"""
    if len(trajectory) < window_size:
        return trajectory
    
    # Use Savitzky-Golay filter for better edge preservation
    try:
        from scipy.signal import savgol_filter
        smoothed = savgol_filter(trajectory, window_size, 2, axis=0)
    except:
        # Fallback to moving average
        smoothed = np.zeros_like(trajectory)
        half_window = window_size // 2
        
        for i in range(len(trajectory)):
            start = max(0, i - half_window)
            end = min(len(trajectory), i + half_window + 1)
            smoothed[i] = np.mean(trajectory[start:end], axis=0)
    
    return smoothed

print("  Smoothing trajectories...")
# Apply more aggressive smoothing for pitching arm joints
for joint_idx in range(33):
    trajectory = landmarks_3d_full[:, joint_idx, :]
    landmarks_3d_full[:, joint_idx, :] = smooth_3d_trajectory(trajectory, window_size=5)

# Special handling for pitching arm (right arm joints: 11, 13, 15)
# Apply less smoothing to preserve the fast pitching motion
pitching_joints = [11, 13, 15]  # Right shoulder, elbow, wrist
print("  Preserving fast motion for pitching arm...")
for joint_idx in pitching_joints:
    trajectory = landmarks_3d_full[:, joint_idx, :]
    landmarks_3d_full[:, joint_idx, :] = smooth_3d_trajectory(trajectory, window_size=3)

# Create mapping from video frames to smoothed landmarks
print("  Creating frame mapping...")
frame_to_landmark_idx = list(range(T_total))  # Now all frames have data

# Enhanced 3D mapping function with better forward motion
def mp_to_mj_3d(pt, frame_idx, total_frames, scale_factor=2.0, depth_scale=2.0):
    """Convert MediaPipe 3D coordinates to MuJoCo world coordinates with forward motion"""
    x, y, z = pt
    
    # MediaPipe coordinates: x,y in [0,1], z is relative depth
    # Convert to MuJoCo coordinates
    
    # Horizontal position
    world_x = (x - 0.5) * scale_factor
    
    # Forward/back position - enhanced for pitching motion
    # Add forward lean during release phase
    forward_bias = 0
    if release_frame - 10 <= frame_idx <= release_frame + 10:
        # Lean forward during release
        forward_bias = 0.3 * (1 - abs(frame_idx - release_frame) / 10)
    
    world_y = -z * depth_scale - forward_bias  # More forward movement
    
    # Vertical position with pitching motion enhancement
    base_height = 1.0
    vertical_adjust = 0
    if release_frame - 5 <= frame_idx <= release_frame + 5:
        # Slight dip then rise during release
        if frame_idx <= release_frame:
            vertical_adjust = -0.1 * (release_frame - frame_idx) / 5
        else:
            vertical_adjust = 0.05 * (frame_idx - release_frame) / 5
    
    world_z = base_height + (0.5 - y) * scale_factor + vertical_adjust
    
    return np.array([world_x, world_y, world_z])

# Calculate body proportions from landmarks
def estimate_body_proportions(landmarks):
    """Estimate body segment lengths from landmarks"""
    # Use average of several frames for stable estimates
    sample_size = min(30, len(landmarks))
    avg_lm = np.mean(landmarks[:sample_size], axis=0)
    
    # Calculate lengths
    torso_len = np.linalg.norm(avg_lm[11] - avg_lm[23]) + np.linalg.norm(avg_lm[12] - avg_lm[24])
    torso_len /= 2.0
    
    upper_arm_len = (np.linalg.norm(avg_lm[11] - avg_lm[13]) + np.linalg.norm(avg_lm[12] - avg_lm[14])) / 2.0
    forearm_len = (np.linalg.norm(avg_lm[13] - avg_lm[15]) + np.linalg.norm(avg_lm[14] - avg_lm[16])) / 2.0
    
    thigh_len = (np.linalg.norm(avg_lm[23] - avg_lm[25]) + np.linalg.norm(avg_lm[24] - avg_lm[26])) / 2.0
    shin_len = (np.linalg.norm(avg_lm[25] - avg_lm[27]) + np.linalg.norm(avg_lm[26] - avg_lm[28])) / 2.0
    
    return {
        'torso': float(torso_len),
        'upper_arm': float(upper_arm_len),
        'forearm': float(forearm_len),
        'thigh': float(thigh_len),
        'shin': float(shin_len)
    }

# Get body proportions
proportions = estimate_body_proportions(landmarks_3d_full)
print("\n  Estimated body proportions:")
for part, length in proportions.items():
    print(f"    {part}: {length:.3f}")

# MANUAL ADJUSTMENTS FOR BETTER PROPORTIONS
print("\n  Applying manual adjustments for better aesthetics:")
print("    - Arms: Longer (increased by 25%)")
print("    - Shoulder width: Narrower (reduced by 35%)")
print("    - Torso: Thinner (reduced by 20%)")

# Scale factors based on estimated proportions with MANUAL ADJUSTMENTS
scale_torso = proportions['torso'] / 0.5 if proportions['torso'] > 0 else 1.0

# ARM ADJUSTMENT: Make arms longer (increase by 25%)
arm_length_multiplier = 1.25  # 25% longer arms
scale_limb = ((proportions['upper_arm'] + proportions['forearm']) / 0.5 
              if (proportions['upper_arm'] + proportions['forearm']) > 0 else 1.0) * arm_length_multiplier

# BODY WIDTH ADJUSTMENT: Make body less wide (reduce by 35%)
shoulder_width_multiplier = 0.65  # 35% narrower
torso_width_multiplier = 0.80    # 20% thinner

print(f"\n  Adjusted scaling factors:")
print(f"    - Torso scale: {scale_torso:.3f}")
print(f"    - Limb scale: {scale_limb:.3f} (with {arm_length_multiplier:.1f}x arm length multiplier)")
print(f"    - Shoulder width multiplier: {shoulder_width_multiplier:.2f}")
print(f"    - Torso width multiplier: {torso_width_multiplier:.2f}")

# -------------------------
# Enhanced MJCF model with PITCHING-OPTIMIZED PROPORTIONS
# -------------------------

mjcf = f"""
<mujoco model="pitcher_3d">
    <compiler angle="radian" coordinate="local"/>
    <option timestep="0.002" gravity="0 0 -9.81"/> <!-- Faster physics for quick movements -->
    
    <worldbody>
        <!-- Ground -->
        <geom name="ground" type="plane" size="5 5 0.1" rgba="0.8 0.9 0.8 1" pos="0 0 0"/>
        
        <!-- Pitcher mound visual -->
        <geom name="mound" type="cylinder" size="0.6 0.1" pos="0 -1.5 0" rgba="0.9 0.8 0.6 1"/>
        
        <!-- Pitcher with PITCHING-OPTIMIZED PROPORTIONS -->
        <body name="pelvis" pos="0 0 1.0">
            <joint name="root_x" type="slide" axis="1 0 0" limited="true" range="-1 1" damping="50"/>
            <joint name="root_y" type="slide" axis="0 1 0" limited="true" range="-2 1" damping="50"/> <!-- More forward range -->
            <joint name="root_z" type="slide" axis="0 0 1" limited="true" range="0.5 2" damping="50"/>
            <joint name="root_yaw" type="hinge" axis="0 0 1" limited="true" range="-60 60" damping="8"/>
            <joint name="root_pitch" type="hinge" axis="1 0 0" limited="true" range="-30 30" damping="8"/>
            <joint name="root_roll" type="hinge" axis="0 1 0" limited="true" range="-30 30" damping="8"/>

            <geom type="ellipsoid" size="0.065 0.09 0.1" rgba="0.7 0.6 0.6 1"/> <!-- Narrower, more athletic pelvis -->

            <!-- Torso - THINNER AND MORE FLEXIBLE -->
            <body name="torso" pos="0 0 0.15" euler="0 0 0">
                <joint name="spine_y" type="hinge" axis="0 0 1" limited="true" range="-25 25" damping="4"/>
                <joint name="spine_x" type="hinge" axis="1 0 0" limited="true" range="-30 20" damping="4"/>
                
                <geom type="capsule" fromto="0 0 0 0 0 {0.55*scale_torso:.3f}" size="0.095" rgba="0.6 0.7 0.6 1"/> <!-- Thinner torso -->
                
                <!-- Head -->
                <body name="head" pos="0 0 {0.65*scale_torso:.3f}">
                    <geom type="sphere" size="0.095" rgba="0.8 0.7 0.7 1"/>
                    <site name="head_site" pos="0 0 0.05" size="0.02" rgba="1 0 0 1"/>
                </body>

                <!-- Right Arm (PITCHING ARM) - EXTRA LONG FOR WINDUP -->
                <body name="upper_arm_r" pos="0 {0.16*scale_torso*shoulder_width_multiplier:.3f} {0.45*scale_torso:.3f}">
                    <joint name="shoulder_r_yaw" type="hinge" axis="0 0 1" limited="true" range="-120 60" damping="2"/> <!-- More range for windup -->
                    <joint name="shoulder_r_pitch" type="hinge" axis="1 0 0" limited="true" range="-150 40" damping="2"/> <!-- More overhead range -->
                    <joint name="shoulder_r_roll" type="hinge" axis="0 1 0" limited="true" range="-80 80" damping="2"/>
                    
                    <geom type="capsule" fromto="0 0 0 {0.38*scale_limb:.3f} 0 0" size="0.05" rgba="0.9 0.3 0.3 1"/> <!-- Longer, thinner -->
                    <site name="site_shoulder_r" pos="0 0 0" size="0.02" rgba="1 0 0 1"/>

                    <body name="forearm_r" pos="{0.38*scale_limb:.3f} 0 0">
                        <joint name="elbow_r" type="hinge" axis="0 1 0" limited="true" range="0 165" damping="1.5"/> <!-- More extension -->
                        <joint name="elbow_twist_r" type="hinge" axis="0 0 1" limited="true" range="-60 60" damping="1.5"/>
                        
                        <geom type="capsule" fromto="0 0 0 {0.35*scale_limb:.3f} 0 0" size="0.042" rgba="1.0 0.4 0.4 1"/> <!-- Longer -->
                        <site name="site_elbow_r" pos="0 0 0" size="0.02" rgba="0 1 0 1"/>

                        <body name="hand_r" pos="{0.35*scale_limb:.3f} 0 0">
                            <geom type="sphere" size="0.038" rgba="1.0 0.5 0.5 1"/>
                            <site name="site_wrist_r" pos="0 0 0" size="0.02" rgba="0 0 1 1"/>
                            <!-- Visual baseball in hand -->
                            <geom name="baseball" type="sphere" size="0.03" pos="0.05 0 0" rgba="0.9 0.9 0.2 1" 
                                  contype="0" conaffinity="0"/> <!-- No collisions -->
                        </body>
                    </body>
                </body>

                <!-- Left Arm (GLOVE ARM) - BALANCED -->
                <body name="upper_arm_l" pos="0 {-0.16*scale_torso*shoulder_width_multiplier:.3f} {0.45*scale_torso:.3f}">
                    <joint name="shoulder_l_yaw" type="hinge" axis="0 0 1" limited="true" range="-60 120" damping="2"/>
                    <joint name="shoulder_l_pitch" type="hinge" axis="1 0 0" limited="true" range="-120 30" damping="2"/>
                    <joint name="shoulder_l_roll" type="hinge" axis="0 1 0" limited="true" range="-60 60" damping="2"/>
                    
                    <geom type="capsule" fromto="0 0 0 {-0.35*scale_limb:.3f} 0 0" size="0.05" rgba="0.3 0.3 0.8 1"/>
                    <site name="site_shoulder_l" pos="0 0 0" size="0.02" rgba="1 0 0 1"/>

                    <body name="forearm_l" pos="{-0.35*scale_limb:.3f} 0 0">
                        <joint name="elbow_l" type="hinge" axis="0 1 0" limited="true" range="0 160" damping="1.5"/>
                        <joint name="elbow_twist_l" type="hinge" axis="0 0 1" limited="true" range="-45 45" damping="1.5"/>
                        
                        <geom type="capsule" fromto="0 0 0 {-0.32*scale_limb:.3f} 0 0" size="0.042" rgba="0.4 0.4 0.9 1"/>
                        <site name="site_elbow_l" pos="0 0 0" size="0.02" rgba="0 1 0 1"/>

                        <body name="hand_l" pos="{-0.32*scale_limb:.3f} 0 0">
                            <geom type="sphere" size="0.038" rgba="0.5 0.5 1.0 1"/>
                            <site name="site_wrist_l" pos="0 0 0" size="0.02" rgba="0 0 1 1"/>
                        </body>
                    </body>
                </body>
            </body>

            <!-- Right Leg - DRIVE LEG -->
            <body name="thigh_r" pos="0 {0.07*scale_torso:.3f} -0.05">
                <joint name="hip_r_y" type="hinge" axis="0 0 1" limited="true" range="-40 40" damping="4"/>
                <joint name="hip_r_x" type="hinge" axis="1 0 0" limited="true" range="-70 40" damping="4"/>
                
                <geom type="capsule" fromto="0 0 0 0 0 -{0.45*scale_torso:.3f}" size="0.065" rgba="0.7 0.6 0.6 1"/>
                <site name="site_hip_r" pos="0 0 0" size="0.02" rgba="1 0 1 1"/>

                <body name="shin_r" pos="0 0 -{0.45*scale_torso:.3f}">
                    <joint name="knee_r" type="hinge" axis="1 0 0" limited="true" range="0 130" damping="3"/>
                    
                    <geom type="capsule" fromto="0 0 0 0 0 -{0.45*scale_torso:.3f}" size="0.055" rgba="0.6 0.7 0.6 1"/>
                    <site name="site_knee_r" pos="0 0 0" size="0.02" rgba="0 1 1 1"/>

                    <body name="foot_r" pos="0 0 -{0.45*scale_torso:.3f}">
                        <geom type="sphere" size="0.04" rgba="0.8 0.8 0.6 1"/>
                        <site name="site_ankle_r" pos="0 0 0" size="0.02" rgba="1 1 0 1"/>
                    </body>
                </body>
            </body>

            <!-- Left Leg - PIVOT LEG -->
            <body name="thigh_l" pos="0 {-0.07*scale_torso:.3f} -0.05">
                <joint name="hip_l_y" type="hinge" axis="0 0 1" limited="true" range="-40 40" damping="4"/>
                <joint name="hip_l_x" type="hinge" axis="1 0 0" limited="true" range="-70 40" damping="4"/>
                
                <geom type="capsule" fromto="0 0 0 0 0 -{0.45*scale_torso:.3f}" size="0.065" rgba="0.7 0.6 0.6 1"/>
                <site name="site_hip_l" pos="0 0 0" size="0.02" rgba="1 0 1 1"/>

                <body name="shin_l" pos="0 0 -{0.45*scale_torso:.3f}">
                    <joint name="knee_l" type="hinge" axis="1 0 0" limited="true" range="0 130" damping="3"/>
                    
                    <geom type="capsule" fromto="0 0 0 0 0 -{0.45*scale_torso:.3f}" size="0.055" rgba="0.6 0.7 0.6 1"/>
                    <site name="site_knee_l" pos="0 0 0" size="0.02" rgba="0 1 1 1"/>

                    <body name="foot_l" pos="0 0 -{0.45*scale_torso:.3f}">
                        <geom type="sphere" size="0.04" rgba="0.8 0.8 0.6 1"/>
                        <site name="site_ankle_l" pos="0 0 0" size="0.02" rgba="1 1 0 1"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
</mujoco>
"""

xml_path = "pitcher_optimized.xml"
with open(xml_path, "w") as f:
    f.write(mjcf)

print(f"\n  Created optimized pitcher model with:")
print(f"    - Faster physics timestep (0.002s)")
print(f"    - Enhanced joint ranges for pitching")
print(f"    - Visual baseball in hand")
print(f"    - Pitcher mound visualization")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Create joint mapping
joint_names = [model.joint(i).name for i in range(model.njnt)]
qadr = {name: model.joint(name).qposadr[0] for name in joint_names}
dofadr = {name: model.joint(name).dofadr[0] for name in joint_names}

# Site IDs for IK targets
site_ids = {}
for site_name in ["site_shoulder_r", "site_elbow_r", "site_wrist_r", 
                  "site_shoulder_l", "site_elbow_l", "site_wrist_l",
                  "site_hip_r", "site_knee_r", "site_ankle_r",
                  "site_hip_l", "site_knee_l", "site_ankle_l",
                  "head_site"]:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id != -1:
        site_ids[site_name] = site_id

# Get baseball geom ID for visual effects
baseball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "baseball")

# Enhanced IK function with faster convergence
def enhanced_ik(model, data, targets, max_iter=30, tolerance=0.02):
    """Enhanced IK with damping and joint limits - optimized for speed"""
    nv = model.nv
    
    for iteration in range(max_iter):
        mujoco.mj_forward(model, data)
        
        # Prepare Jacobian and error arrays
        target_count = len(targets)
        jac = np.zeros((3 * target_count, nv))
        error = np.zeros(3 * target_count)
        
        idx = 0
        for site_name, target_pos in targets.items():
            if site_name not in site_ids:
                continue
                
            site_id = site_ids[site_name]
            
            # Site position error
            site_pos = data.site_xpos[site_id]
            error[3*idx:3*idx+3] = target_pos - site_pos
            
            # Jacobian for this site
            jac_site = np.zeros((3, nv))
            mujoco.mj_jacSite(model, data, jac_site, None, site_id)
            jac[3*idx:3*idx+3, :] = jac_site
            
            idx += 1
        
        # Truncate arrays to actual number of processed targets
        jac = jac[:3*idx, :]
        error = error[:3*idx]
        
        if idx == 0:
            return  # No valid targets
        
        # Damped least squares solution with adaptive damping
        jac_t = jac.T
        damping = 1e-5 + 1e-4 * iteration  # Adaptive damping
        hessian = jac @ jac_t + damping * np.eye(jac.shape[0])
        
        try:
            dq = jac_t @ np.linalg.solve(hessian, error)
        except np.linalg.LinAlgError:
            # Fallback to simple gradient descent
            dq = jac_t @ error * 0.02
        
        # Apply with adaptive step size
        max_step = 0.15
        step_norm = np.linalg.norm(dq)
        if step_norm > max_step:
            dq = dq * max_step / step_norm
        
        # Update positions directly with momentum
        momentum = 0.3
        data.qpos[:] += dq * (0.1 + momentum * (1 - iteration/max_iter))
        
        # Check convergence
        if np.linalg.norm(error) < tolerance:
            break
    
    # Zero velocities after IK
    data.qvel[:] = 0.0

# Initialize viewer
try:
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -15
    viewer.cam.distance = 2.8
    viewer.cam.lookat[:] = [0, -0.5, 1.0]  # Look slightly forward
    
    T = T_total
    print(f"\n  Playing ALL {T} frames at FASTER SPEED")
    print(f"  Capture frames around release: {max(0, release_frame-5)} to {min(T, release_frame+15)}")
    print(f"  Animation will loop continuously")
    print("  Press ESC or close window to exit\n")
    
    # FASTER PLAYBACK SETTINGS
    dt = 1.0 / max(fps, 1)
    speed_multiplier = 2.0  # Play 2x faster than real-time
    target_dt = dt / speed_multiplier
    
    # Store for smooth interpolation
    prev_targets = None
    smoothing_factor = 0.4  # Less smoothing for faster movements
    
    # Animation loop with reset capability
    iteration = 0
    max_iterations = 50  # Maximum number of loops
    
    # Statistics
    frame_times = []
    
    while viewer.is_running() and iteration < max_iterations:
        frame_idx = 0
        loop_start_time = time.time()
        loop_frame_count = 0
        
        print(f"\n  Starting FAST loop #{iteration + 1} ({speed_multiplier:.1f}x speed)")
        
        # Reset model to initial state
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        
        # Reset baseball visibility
        if baseball_geom_id != -1:
            model.geom_rgba[baseball_geom_id][3] = 1.0  # Make baseball visible
        
        while frame_idx < T and viewer.is_running():
            frame_start_time = time.time()
            
            # Convert key landmarks to MuJoCo coordinates WITH FRAME-AWARE MAPPING
            targets = {}
            
            # Upper body - with frame-specific adjustments
            targets['site_shoulder_r'] = mp_to_mj_3d(landmarks_3d_full[frame_idx, 11], frame_idx, T)
            targets['site_elbow_r'] = mp_to_mj_3d(landmarks_3d_full[frame_idx, 13], frame_idx, T)
            targets['site_wrist_r'] = mp_to_mj_3d(landmarks_3d_full[frame_idx, 15], frame_idx, T)
            
            targets['site_shoulder_l'] = mp_to_mj_3d(landmarks_3d_full[frame_idx, 12], frame_idx, T)
            targets['site_elbow_l'] = mp_to_mj_3d(landmarks_3d_full[frame_idx, 14], frame_idx, T)
            targets['site_wrist_l'] = mp_to_mj_3d(landmarks_3d_full[frame_idx, 16], frame_idx, T)
            
            # Lower body
            targets['site_hip_r'] = mp_to_mj_3d(landmarks_3d_full[frame_idx, 23], frame_idx, T)
            targets['site_knee_r'] = mp_to_mj_3d(landmarks_3d_full[frame_idx, 25], frame_idx, T)
            targets['site_ankle_r'] = mp_to_mj_3d(landmarks_3d_full[frame_idx, 27], frame_idx, T)
            
            targets['site_hip_l'] = mp_to_mj_3d(landmarks_3d_full[frame_idx, 24], frame_idx, T)
            targets['site_knee_l'] = mp_to_mj_3d(landmarks_3d_full[frame_idx, 26], frame_idx, T)
            targets['site_ankle_l'] = mp_to_mj_3d(landmarks_3d_full[frame_idx, 28], frame_idx, T)
            
            # Head
            targets['head_site'] = mp_to_mj_3d(landmarks_3d_full[frame_idx, 0], frame_idx, T)
            
            # Apply smoothing between frames (but not between loops)
            if prev_targets is not None and frame_idx > 0:
                for key in targets:
                    if key in prev_targets:
                        targets[key] = (1 - smoothing_factor) * prev_targets[key] + smoothing_factor * targets[key]
            
            # Apply IK - faster convergence for speed
            enhanced_ik(model, data, targets, max_iter=15, tolerance=0.03)
            
            prev_targets = targets
            
            # Special visual effects for pitching sequence
            if frame_idx == release_frame:
                print(f"    ⚾ RELEASE at frame {frame_idx}!")
                # Make throwing arm brighter
                model.geom_rgba[6][0] = 1.0  # Right hand bright red
                model.geom_rgba[6][1] = 0.3
                model.geom_rgba[6][2] = 0.3
                
            elif frame_idx == release_frame + 1:
                # Start ball release animation
                if baseball_geom_id != -1:
                    # Move ball forward from hand
                    model.geom_pos[baseball_geom_id][1] -= 0.1  # Forward
                    
            elif release_frame + 1 < frame_idx <= release_frame + 10:
                # Continue ball trajectory
                if baseball_geom_id != -1:
                    forward_speed = 0.15
                    model.geom_pos[baseball_geom_id][1] -= forward_speed
                    model.geom_pos[baseball_geom_id][2] += 0.02  # Slight upward arc
                    
                    # Fade ball out
                    fade = 1.0 - (frame_idx - release_frame) / 15
                    model.geom_rgba[baseball_geom_id][3] = max(0, fade)
                    
            elif frame_idx == release_frame + 11:
                # Reset ball position and visibility
                if baseball_geom_id != -1:
                    model.geom_pos[baseball_geom_id][1] = 0.05
                    model.geom_pos[baseball_geom_id][2] = 0
                    model.geom_rgba[baseball_geom_id][3] = 0  # Hide ball
            
            # Forward dynamics and render
            mujoco.mj_forward(model, data)
            viewer.sync()
            
            # FAST TIMING CONTROL
            elapsed = time.time() - frame_start_time
            frame_times.append(elapsed)
            
            # Calculate sleep time for desired speed
            sleep_time = max(0, target_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            frame_idx += 1
            loop_frame_count += 1
            
            # Show progress within current loop (less frequent updates for speed)
            if frame_idx % 20 == 0 or frame_idx == release_frame or frame_idx == release_frame + 1:
                progress = frame_idx / T * 100
                
                if frame_idx < release_frame:
                    phase = "Wind-up"
                    frames_to_go = release_frame - frame_idx
                elif frame_idx == release_frame:
                    phase = "⚾ RELEASE!"
                    frames_to_go = 0
                elif frame_idx <= release_frame + 5:
                    phase = "Follow-through"
                    frames_to_go = frame_idx - release_frame
                else:
                    phase = "Recovery"
                    frames_to_go = frame_idx - release_frame
                
                fps_actual = 1.0 / (sum(frame_times[-10:]) / min(10, len(frame_times))) if frame_times else 0
                print(f"  Frame: {frame_idx:4d}/{T} ({progress:5.1f}%) | {phase:15} | FPS: {fps_actual:5.1f}", end='\r')
                
                # Reset arm color after release sequence
                if frame_idx == release_frame + 6:
                    model.geom_rgba[6][0] = 1.0  # Reset right hand color
                    model.geom_rgba[6][1] = 0.5
                    model.geom_rgba[6][2] = 0.5
        
        iteration += 1
        loop_time = time.time() - loop_start_time
        avg_fps = loop_frame_count / loop_time if loop_time > 0 else 0
        
        print(f"\n  Completed loop #{iteration} in {loop_time:.1f}s ({avg_fps:.1f} avg FPS)")
        
        # Brief pause between loops
        if viewer.is_running() and iteration < max_iterations:
            print("  Preparing next loop...")
            time.sleep(0.3)
            
except KeyboardInterrupt:
    print("\n  Animation interrupted by user")
except Exception as e:
    print(f"\n  Error during animation: {e}")
    import traceback
    traceback.print_exc()
finally:
    if 'viewer' in locals():
        viewer.close()
    print("\n✓ MuJoCo 3D animation complete")

# Cleanup
if os.path.exists(xml_path):
    os.remove(xml_path)

print("\n" + "=" * 60)
print("PITCH RELEASE FRAME DETECTION RESULTS")
print("=" * 60)
print(f"\nPredicted Release Frame: {release_frame}")
print(f"  Total frames analyzed: {T_total}")
print(f"  Release window: Frames {max(0, release_frame-10)} to {min(T_total, release_frame+15)}")
print(f"  Wrist Velocity: {velocities_clean[release_frame]:.4f}")
print(f"  Arm Angle from Torso: {arm_vertical_clean[release_frame]:.2f}°")
print(f"  Elbow Extension: {elbow_extension_clean[release_frame]:.2f}°")
print(f"  Wrist Height: {wrist_heights_clean[release_frame]:.4f}")
print(f"  Arm Extension: {distances_clean[release_frame]:.4f}")
print("=" * 60)
print("\n✓ Analysis complete! Animation captured all frames at faster speed.")