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
cap = cv2.VideoCapture('pitch1.mov')

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Storage for tracking metrics
arm_vertical_angles = []  # Changed from horizontal to vertical
elbow_extension_angles = []
wrist_heights = []
shoulder_to_wrist_distances = []
wrist_velocities = []
frames_data = []
frame_count = 0
prev_wrist_pos = None
target_person_id = None  # Track which person we're following
initial_shoulder_center = None  # Store initial shoulder position for tracking

print("Processing video - Pass 1: Extracting metrics...")

# Track multiple consecutive frames to confirm lock-on
consecutive_detections = []
lock_on_threshold = 3  # Need 3 consecutive detections to confirm
max_frames_to_wait = 5  # After this, lock onto closest person

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frames_data.append(frame.copy())
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # Calculate shoulder center position for this detection
        current_shoulder_center = np.array([
            (lm[11].x + lm[11].y) / 2,
            (lm[12].y + lm[12].y) / 2
        ])
        
        # Check average visibility
        avg_visibility = np.mean([lm[i].visibility for i in [11, 12, 13, 14, 15, 16, 23, 24]])
        
        # Calculate "closeness" score - larger shoulder width and higher visibility = closer to camera
        shoulder_width = np.linalg.norm(np.array([lm[11].x, lm[11].y]) - np.array([lm[12].x, lm[12].y]))
        closeness_score = shoulder_width * avg_visibility
        
        # If this is the first frame, wait for someone with arm movement
        if target_person_id is None:
            # Left landmarks
            left_shoulder = np.array([lm[11].x, lm[11].y])
            left_wrist = np.array([lm[15].x, lm[15].y])
            right_shoulder = np.array([lm[12].x, lm[12].y])
            right_wrist = np.array([lm[16].x, lm[16].y])
            
            # Calculate arm extension
            left_ext = np.linalg.norm(left_wrist - left_shoulder)
            right_ext = np.linalg.norm(right_wrist - right_shoulder)
            max_ext = max(left_ext, right_ext)
            
            # Detect arm movement with lower threshold for initial detection
            if max_ext > 0.2 and avg_visibility > 0.4:  # Lowered thresholds
                consecutive_detections.append({
                    'frame': frame_count,
                    'position': current_shoulder_center.copy(),
                    'extension': max_ext,
                    'closeness': closeness_score
                })
                
                # Confirm lock-on after consecutive detections
                if len(consecutive_detections) >= lock_on_threshold:
                    # Check that detections are for the same person (positions are close)
                    positions = np.array([d['position'] for d in consecutive_detections[-lock_on_threshold:]])
                    max_distance = np.max([np.linalg.norm(positions[i] - positions[i+1]) 
                                          for i in range(len(positions)-1)])
                    
                    if max_distance < 0.1:  # Same person across frames
                        target_person_id = consecutive_detections[0]['frame']
                        initial_shoulder_center = consecutive_detections[0]['position'].copy()
                        print(f"  Locked onto pitcher at frame {target_person_id} (arm movement detected)")
                        consecutive_detections = []
            else:
                consecutive_detections = []  # Reset if movement stops
            
            # Fallback: If we've waited too long without arm movement, lock onto closest person
            if frame_count >= max_frames_to_wait and target_person_id is None:
                if avg_visibility > 0.4 and shoulder_width > 0.05:
                    target_person_id = frame_count
                    initial_shoulder_center = current_shoulder_center.copy()
                    print(f"  Locked onto closest person at frame {frame_count} (fallback: no arm movement detected)")
                    print(f"  Closeness score: {closeness_score:.3f} (shoulder_width={shoulder_width:.3f}, visibility={avg_visibility:.3f})")
        
        # If we've locked onto someone, check if this is still the same person
        if target_person_id is not None:
            # More lenient tracking once locked on
            if initial_shoulder_center is not None:
                distance_from_initial = np.linalg.norm(current_shoulder_center - initial_shoulder_center)
                # Allow larger movement during pitch motion
                is_same_person = distance_from_initial < 0.25 and avg_visibility > 0.3
            else:
                is_same_person = True
            
            if is_same_person:
                # Process this person
                # Left landmarks
                left_shoulder = np.array([lm[11].x, lm[11].y])
                left_elbow = np.array([lm[13].x, lm[13].y])
                left_wrist = np.array([lm[15].x, lm[15].y])
                left_hip = np.array([lm[23].x, lm[23].y])

                # Right landmarks
                right_shoulder = np.array([lm[12].x, lm[12].y])
                right_elbow = np.array([lm[14].x, lm[14].y])
                right_wrist = np.array([lm[16].x, lm[16].y])
                right_hip = np.array([lm[24].x, lm[24].y])
                
                # Update initial position with weighted average for drift compensation
                initial_shoulder_center = 0.95 * initial_shoulder_center + 0.05 * current_shoulder_center
                
                # Determine pitching arm by extension
                left_ext = np.linalg.norm(left_wrist - left_shoulder)
                right_ext = np.linalg.norm(right_wrist - right_shoulder)

                if left_ext > right_ext:
                    shoulder, elbow, wrist = left_shoulder, left_elbow, left_wrist
                    hip = left_hip
                    is_left_arm = True
                else:
                    shoulder, elbow, wrist = right_shoulder, right_elbow, right_wrist
                    hip = right_hip
                    is_left_arm = False

                # Calculate torso vector (shoulder to hip on pitching side)
                torso_vector = hip - shoulder
                
                # Arm angle to vertical (measured from torso line)
                arm_vector = wrist - shoulder
                # Calculate angle between arm and torso vectors
                dot_product = np.dot(arm_vector, torso_vector)
                magnitudes = np.linalg.norm(arm_vector) * np.linalg.norm(torso_vector)
                arm_vertical_angle = np.degrees(np.arccos(np.clip(dot_product / magnitudes, -1.0, 1.0)))
                arm_vertical_angles.append(arm_vertical_angle)

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
                # Different person or low visibility, skip but keep target locked
                lm = None
        else:
            # Haven't locked onto anyone yet
            lm = None
    else:
        lm = None
        
    if lm is None:
        arm_vertical_angles.append(None)
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
    
    # If all values are NaN, return zeros
    if mask.all():
        return np.zeros_like(arr)
    
    # If no NaN values, return as is
    if not mask.any():
        return arr
    
    # If there are some valid values, interpolate
    valid_indices = np.flatnonzero(~mask)
    if len(valid_indices) > 0:
        arr[mask] = np.interp(np.flatnonzero(mask), valid_indices, arr[~mask])
    
    return arr

arm_vertical_clean = clean_data(arm_vertical_angles)
elbow_extension_clean = clean_data(elbow_extension_angles)
wrist_heights_clean = clean_data(wrist_heights)
distances_clean = clean_data(shoulder_to_wrist_distances)
velocities_clean = np.array(wrist_velocities)

# Check if we have valid data
if len(arm_vertical_clean) == 0 or target_person_id is None:
    print("\n" + "="*60)
    print("WARNING: No person detected with arm movement in the video!")
    print("="*60)
    print("The script will continue with empty data.")
    print("Possible reasons:")
    print("  - Pitcher not visible in early frames")
    print("  - Arm movement threshold too high")
    print("  - Video quality or lighting issues")
    print("  - Person too far from camera")
    print("\nTip: Try adjusting the arm extension threshold (currently 0.2)")
    print("     or visibility threshold (currently 0.4) in the code.")
    print("="*60 + "\n")
    
    # Create dummy data to allow video processing to continue
    num_frames = len(frames_data)
    arm_vertical_clean = np.zeros(num_frames)
    elbow_extension_clean = np.zeros(num_frames)
    wrist_heights_clean = np.zeros(num_frames)
    distances_clean = np.zeros(num_frames)
    velocities_clean = np.zeros(num_frames)
    
    # Set a default release frame
    release_frame = num_frames // 2 if num_frames > 0 else 0
else:
    print(f"  Successfully tracked pitcher from frame {target_person_id} onwards")

start_frame = 5  # skip first few frames to avoid initial noise

# Find peak velocity as the primary release indicator
# Use peak detection to find local maxima in velocity with constraints
# Release must have high velocity AND extended arm (elbow nearly straight)

# Create a mask for valid release candidates
# Conditions: high velocity (>0.25) AND extended arm (>100°)
velocity_threshold = 0.25
elbow_threshold = 100.0

valid_release_mask = (velocities_clean > velocity_threshold) & (elbow_extension_clean > elbow_threshold) & (np.arange(len(velocities_clean)) >= start_frame)

if np.any(valid_release_mask):
    # Among valid candidates, find the frame with maximum velocity
    valid_indices = np.where(valid_release_mask)[0]
    valid_velocities = velocities_clean[valid_indices]
    release_frame = valid_indices[np.argmax(valid_velocities)]
    print(f"  Release detected at frame {release_frame} (velocity={velocities_clean[release_frame]:.4f}, elbow={elbow_extension_clean[release_frame]:.1f}°)")
else:
    # Fallback: try lowering thresholds slightly
    velocity_threshold_low = 0.15
    elbow_threshold_low = 80.0
    
    valid_release_mask = (velocities_clean > velocity_threshold_low) & (elbow_extension_clean > elbow_threshold_low) & (np.arange(len(velocities_clean)) >= start_frame)
    
    if np.any(valid_release_mask):
        valid_indices = np.where(valid_release_mask)[0]
        valid_velocities = velocities_clean[valid_indices]
        release_frame = valid_indices[np.argmax(valid_velocities)]
        print(f"  Release detected at frame {release_frame} with relaxed thresholds (velocity={velocities_clean[release_frame]:.4f}, elbow={elbow_extension_clean[release_frame]:.1f}°)")
    else:
        # Last resort: use overall max velocity
        release_frame = np.argmax(velocities_clean) if len(velocities_clean) > 0 else 0
        print(f"  Warning: Using max velocity frame {release_frame} (thresholds not met)")


# Store normalized metrics for visualization
norm_elbow = (elbow_extension_clean - elbow_extension_clean.min()) / (elbow_extension_clean.max() - elbow_extension_clean.min())
norm_distance = (distances_clean - distances_clean.min()) / (distances_clean.max() - distances_clean.min())
norm_wrist_height = (wrist_heights_clean.max() - wrist_heights_clean) / (wrist_heights_clean.max() - wrist_heights_clean.min())
norm_velocity = (velocities_clean - velocities_clean.min()) / (velocities_clean.max() - velocities_clean.min())

# Additional metrics for plotting
max_elbow_extension_idx = np.argmax(elbow_extension_clean)
max_vertical_angle_idx = np.argmax(arm_vertical_clean)
min_wrist_height_idx = np.argmin(wrist_heights_clean)
max_distance_idx = np.argmax(distances_clean)
# Peak velocity is now the release_frame (already calculated above)

# --- VIDEO OUTPUT WITH OVERLAYS ---
print("\nProcessing video - Pass 2: Creating annotated video...")
output_path = 'pitch_analysis_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
pose2 = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

for idx, frame in enumerate(frames_data):
    results = pose2.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # Check if this is the same person we locked onto
        current_shoulder_center = np.array([
            (lm[11].x + lm[12].x) / 2,
            (lm[11].y + lm[12].y) / 2
        ])
        
        avg_visibility = np.mean([lm[i].visibility for i in [11, 12, 13, 14, 15, 16, 23, 24]])
        
        # Check if this matches our target person with more lenient criteria
        is_target_person = False
        if initial_shoulder_center is not None and target_person_id is not None:
            distance_from_initial = np.linalg.norm(current_shoulder_center - initial_shoulder_center)
            is_target_person = (distance_from_initial < 0.25 and avg_visibility > 0.3)
        
        if is_target_person:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            h, w = frame.shape[:2]

            left_shoulder_px = (int(lm[11].x * w), int(lm[11].y * h))
            left_elbow_px = (int(lm[13].x * w), int(lm[13].y * h))
            left_wrist_px = (int(lm[15].x * w), int(lm[15].y * h))
            left_hip_px = (int(lm[23].x * w), int(lm[23].y * h))

            right_shoulder_px = (int(lm[12].x * w), int(lm[12].y * h))
            right_elbow_px = (int(lm[14].x * w), int(lm[14].y * h))
            right_wrist_px = (int(lm[16].x * w), int(lm[16].y * h))
            right_hip_px = (int(lm[24].x * w), int(lm[24].y * h))

            left_ext = np.linalg.norm(np.array(left_wrist_px) - np.array(left_shoulder_px))
            right_ext = np.linalg.norm(np.array(right_wrist_px) - np.array(right_shoulder_px))

            if left_ext > right_ext:
                shoulder_px, elbow_px, wrist_px = left_shoulder_px, left_elbow_px, left_wrist_px
                hip_px = left_hip_px
            else:
                shoulder_px, elbow_px, wrist_px = right_shoulder_px, right_elbow_px, right_wrist_px
                hip_px = right_hip_px

            # Draw arm segments
            cv2.line(frame, shoulder_px, elbow_px, (255, 255, 0), 4)
            cv2.line(frame, elbow_px, wrist_px, (255, 255, 0), 4)
            cv2.circle(frame, wrist_px, 8, (0, 255, 255), -1)

            # Draw vertical reference line (torso line extended)
            # Calculate torso vector and extend it
            torso_vec = np.array([hip_px[0] - shoulder_px[0], hip_px[1] - shoulder_px[1]])
            torso_length = np.linalg.norm(torso_vec)
            if torso_length > 0:
                torso_unit = torso_vec / torso_length
                # Extend the line 200 pixels in both directions along torso
                vertical_start = (int(shoulder_px[0] - torso_unit[0] * 100), 
                                int(shoulder_px[1] - torso_unit[1] * 100))
                vertical_end = (int(shoulder_px[0] + torso_unit[0] * 200), 
                              int(shoulder_px[1] + torso_unit[1] * 200))
            else:
                # Fallback to pure vertical if torso detection fails
                vertical_start = (shoulder_px[0], shoulder_px[1] - 100)
                vertical_end = (shoulder_px[0], shoulder_px[1] + 200)
            
            # Draw torso reference line
            cv2.line(frame, vertical_start, vertical_end, (0, 165, 255), 3)
            cv2.line(frame, shoulder_px, hip_px, (0, 165, 255), 4)  # Draw actual torso segment
            cv2.circle(frame, shoulder_px, 6, (0, 165, 255), -1)
            cv2.circle(frame, hip_px, 6, (0, 165, 255), -1)
            cv2.putText(frame, "Torso Line", (vertical_end[0] + 10, vertical_end[1]),
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
        (f"Arm Angle from Torso: {arm_vertical_clean[idx]:.1f}°", (0, 165, 255)),
        (f"Elbow Extension: {elbow_extension_clean[idx]:.1f}°", (0, 255, 255)),
        (f"Wrist Height: {wrist_heights_clean[idx]:.4f}", (255, 200, 100)),
        (f"Arm Extension: {distances_clean[idx]:.4f}", (100, 255, 100)),
        (f"Wrist Velocity: {velocities_clean[idx]:.4f}", (255, 100, 255)),
        (f"Velocity Score: {norm_velocity[idx]:.3f}", (255, 255, 255))
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
plt.show()

# --- PRINT RESULTS ---
print("\n" + "="*60)
print("PITCH RELEASE FRAME DETECTION RESULTS")
print("="*60)
print(f"\nRelease Detection Method: Peak Wrist Velocity")
print(f"Predicted Release Frame: {release_frame}")
print(f"  - Wrist Velocity: {velocities_clean[release_frame]:.4f} (PEAK)")
print(f"  - Arm Angle from Torso: {arm_vertical_clean[release_frame]:.2f}°")
print(f"  - Elbow Extension: {elbow_extension_clean[release_frame]:.2f}°")
print(f"  - Wrist Height: {wrist_heights_clean[release_frame]:.4f}")
print(f"  - Arm Extension: {distances_clean[release_frame]:.4f}")
print("="*60)