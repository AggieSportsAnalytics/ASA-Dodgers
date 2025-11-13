import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open video
cap = cv2.VideoCapture('pitch2.mov')

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Storage for tracking metrics
arm_horizontal_angles = []
elbow_extension_angles = []
wrist_heights = []
shoulder_to_wrist_distances = []
wrist_velocities = []
frames_data = []
frame_count = 0
prev_wrist_pos = None

print("Processing video - Pass 1: Extracting metrics...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frames_data.append(frame.copy())
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Left landmarks
        left_shoulder = np.array([lm[11].x, lm[11].y])
        left_elbow = np.array([lm[13].x, lm[13].y])
        left_wrist = np.array([lm[15].x, lm[15].y])

        # Right landmarks
        right_shoulder = np.array([lm[12].x, lm[12].y])
        right_elbow = np.array([lm[14].x, lm[14].y])
        right_wrist = np.array([lm[16].x, lm[16].y])

        # Determine pitching arm by extension
        left_ext = np.linalg.norm(left_wrist - left_shoulder)
        right_ext = np.linalg.norm(right_wrist - right_shoulder)

        if left_ext > right_ext:
            shoulder, elbow, wrist = left_shoulder, left_elbow, left_wrist
        else:
            shoulder, elbow, wrist = right_shoulder, right_elbow, right_wrist

        # Arm angle to horizontal
        arm_vector = wrist - shoulder
        arm_horizontal_angle = np.degrees(np.arctan2(-arm_vector[1], arm_vector[0]))
        arm_horizontal_angles.append(arm_horizontal_angle)

        # Elbow extension (180° = straight)
        v1 = shoulder - elbow
        v2 = wrist - elbow
        elbow_angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
        elbow_extension_angles.append(elbow_angle)

        # Wrist height (y-coordinate)
        wrist_heights.append(wrist[1])

        # Shoulder-to-wrist distance
        shoulder_to_wrist_distances.append(np.linalg.norm(shoulder - wrist))

        # Wrist velocity
        if prev_wrist_pos is not None:
            wrist_velocities.append(np.linalg.norm(wrist - prev_wrist_pos))
        else:
            wrist_velocities.append(0)
        prev_wrist_pos = wrist.copy()
    else:
        arm_horizontal_angles.append(None)
        elbow_extension_angles.append(None)
        wrist_heights.append(None)
        shoulder_to_wrist_distances.append(None)
        wrist_velocities.append(0)
        prev_wrist_pos = None

cap.release()

# Interpolate missing values
def clean_data(data):
    arr = np.array(data, dtype=float)
    mask = np.isnan(arr)
    if mask.any():
        arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    return arr

arm_horizontal_clean = clean_data(arm_horizontal_angles)
elbow_extension_clean = clean_data(elbow_extension_angles)
wrist_heights_clean = clean_data(wrist_heights)
distances_clean = clean_data(shoulder_to_wrist_distances)
velocities_clean = np.array(wrist_velocities)

# --- REFINED RELEASE FRAME DETECTION ---
start_frame = 5  # skip first few frames to avoid initial noise
norm_elbow = (elbow_extension_clean - elbow_extension_clean.min()) / (elbow_extension_clean.max() - elbow_extension_clean.min())
norm_distance = (distances_clean - distances_clean.min()) / (distances_clean.max() - distances_clean.min())
norm_wrist_height = (wrist_heights_clean.max() - wrist_heights_clean) / (wrist_heights_clean.max() - wrist_heights_clean.min())

# Only consider frames with nearly straight arm (>160°) and after start_frame
straight_arm_mask = elbow_extension_clean > 160
candidate_mask = straight_arm_mask & (np.arange(len(arm_horizontal_clean)) >= start_frame)

if np.any(candidate_mask):
    candidate_indices = np.where(candidate_mask)[0]
    candidate_wrist_heights = norm_wrist_height[candidate_indices]
    candidate_scores = candidate_wrist_heights + 0.3*norm_elbow[candidate_indices] + 0.1*norm_distance[candidate_indices]
    release_frame = candidate_indices[np.argmax(candidate_scores)]
else:
    release_frame = np.argmax(norm_wrist_height)

# Additional metrics for plotting
max_elbow_extension_idx = np.argmax(elbow_extension_clean)
min_horizontal_angle_idx = np.argmin(arm_horizontal_clean)
min_wrist_height_idx = np.argmin(wrist_heights_clean)
max_distance_idx = np.argmax(distances_clean)
peaks, _ = find_peaks(velocities_clean, distance=10)
max_velocity_idx = peaks[np.argmax(velocities_clean[peaks])] if len(peaks) > 0 else np.argmax(velocities_clean)

# --- VIDEO OUTPUT WITH OVERLAYS ---
print("\nProcessing video - Pass 2: Creating annotated video...")
output_path = 'pitch_analysis_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
pose2 = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

for idx, frame in enumerate(frames_data):
    results = pose2.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        lm = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        left_shoulder_px = (int(lm[11].x * w), int(lm[11].y * h))
        left_elbow_px = (int(lm[13].x * w), int(lm[13].y * h))
        left_wrist_px = (int(lm[15].x * w), int(lm[15].y * h))

        right_shoulder_px = (int(lm[12].x * w), int(lm[12].y * h))
        right_elbow_px = (int(lm[14].x * w), int(lm[14].y * h))
        right_wrist_px = (int(lm[16].x * w), int(lm[16].y * h))

        left_ext = np.linalg.norm(np.array(left_wrist_px) - np.array(left_shoulder_px))
        right_ext = np.linalg.norm(np.array(right_wrist_px) - np.array(right_shoulder_px))

        if left_ext > right_ext:
            shoulder_px, elbow_px, wrist_px = left_shoulder_px, left_elbow_px, left_wrist_px
        else:
            shoulder_px, elbow_px, wrist_px = right_shoulder_px, right_elbow_px, right_wrist_px

        cv2.line(frame, shoulder_px, elbow_px, (255, 255, 0), 4)
        cv2.line(frame, elbow_px, wrist_px, (255, 255, 0), 4)
        cv2.circle(frame, wrist_px, 8, (0, 255, 255), -1)

        # Horizontal reference
        arm_vec_x = wrist_px[0] - shoulder_px[0]
        horizontal_end = (shoulder_px[0] + 200, shoulder_px[1]) if arm_vec_x > 0 else (shoulder_px[0] - 200, shoulder_px[1])
        cv2.line(frame, shoulder_px, horizontal_end, (0, 165, 255), 3)
        cv2.circle(frame, shoulder_px, 6, (0, 165, 255), -1)
        cv2.putText(frame, "Horizontal", (horizontal_end[0] - 80, horizontal_end[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    # Overlay and metrics panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    panel_height = 220
    cv2.rectangle(overlay, (0, height - panel_height), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    title = "PITCH BIOMECHANICS ANALYSIS"
    cv2.putText(frame, title, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {idx}/{len(frames_data)-1}", (width - 220, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y_start = height - panel_height + 30
    line_height = 32
    metrics = [
        (f"Arm Angle to Horizontal: {arm_horizontal_clean[idx]:.1f}°", (0, 165, 255)),
        (f"Elbow Extension: {elbow_extension_clean[idx]:.1f}°", (0, 255, 255)),
        (f"Wrist Height: {wrist_heights_clean[idx]:.4f}", (255, 200, 100)),
        (f"Arm Extension: {distances_clean[idx]:.4f}", (100, 255, 100)),
        (f"Wrist Velocity: {velocities_clean[idx]:.4f}", (255, 100, 255)),
        (f"Release Score: {norm_wrist_height[idx]:.3f}", (255, 255, 255))
    ]

    for i, (text, color) in enumerate(metrics):
        cv2.putText(frame, text, (20, y_start + i * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    if idx == release_frame:
        cv2.rectangle(frame, (0, height//2 - 60), (width, height//2 + 60), (0, 255, 0), -1)
        cv2.putText(frame, "*** RELEASE POINT ***", (width//2 - 280, height//2 + 15),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 0), 5)

    frames_to_release = release_frame - idx
    if -5 <= frames_to_release <= 10:
        if frames_to_release > 0:
            msg = f"Release in {frames_to_release} frames"
            color = (0, 255, 255)
        elif frames_to_release == 0:
            msg = "RELEASE NOW!"
            color = (0, 255, 0)
        else:
            msg = f"Release {-frames_to_release} frames ago"
            color = (150, 150, 255)
        cv2.putText(frame, msg, (width - 350, height - 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

    # Progress bar
    bar_width = width - 40
    bar_x, bar_y, bar_height = 20, 70, 15
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    progress = int((idx / (len(frames_data) - 1)) * bar_width)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress, bar_y + bar_height), (0, 255, 0), -1)
    release_x = bar_x + int((release_frame / (len(frames_data) - 1)) * bar_width)
    cv2.line(frame, (release_x, bar_y - 5), (release_x, bar_y + bar_height + 5), (255, 255, 0), 3)

    out.write(frame)
    if idx % 30 == 0:
        print(f"  Processing frame {idx}/{len(frames_data)-1}")

out.release()
pose2.close()
print(f"\n✓ Annotated video saved as: {output_path}")

# --- PLOTS ---
fig, axes = plt.subplots(6, 1, figsize=(12, 12))
axes[0].plot(arm_horizontal_clean, label='Arm Angle to Horizontal', color='orange')
axes[0].axvline(min_horizontal_angle_idx, color='red', linestyle='--', alpha=0.5, label=f'Most Horizontal ({min_horizontal_angle_idx})')
axes[0].axvline(release_frame, color='green', linestyle='--', linewidth=2, label=f'Release ({release_frame})')
axes[0].axhline(0, color='gray', linestyle=':', alpha=0.3)
axes[0].set_ylabel('Angle (°)')
axes[0].set_title('Arm Angle to Horizontal')
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
axes[4].axvline(max_velocity_idx, color='red', linestyle='--', alpha=0.5, label=f'Peak ({max_velocity_idx})')
axes[4].axvline(release_frame, color='green', linestyle='--', linewidth=2, label=f'Release ({release_frame})')
axes[4].set_ylabel('Velocity')
axes[4].set_title('Wrist Velocity')
axes[4].legend()
axes[4].grid(True, alpha=0.3)

axes[5].plot(norm_wrist_height, label='Composite Release Score', color='green')
axes[5].axvline(release_frame, color='green', linestyle='--', linewidth=2, label=f'Release ({release_frame})')
axes[5].set_ylabel('Score')
axes[5].set_xlabel('Frame')
axes[5].set_title('Composite Release Score')
axes[5].legend()
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pitch_analysis_plots.png', dpi=150, bbox_inches='tight')
print(f"✓ Analysis plots saved as: pitch_analysis_plots.png")
plt.show()

# --- PRINT RESULTS ---
print("\n" + "="*60)
print("PITCH RELEASE FRAME DETECTION RESULTS")
print("="*60)
print(f"\nPredicted Release Frame: {release_frame}")
print(f"  - Arm Angle to Horizontal: {arm_horizontal_clean[release_frame]:.2f}°")
print(f"  - Elbow Extension: {elbow_extension_clean[release_frame]:.2f}°")
print(f"  - Wrist Height: {wrist_heights_clean[release_frame]:.4f}")
print(f"  - Arm Extension: {distances_clean[release_frame]:.4f}")
print(f"  - Wrist Velocity: {velocities_clean[release_frame]:.4f}")
print("="*60)
