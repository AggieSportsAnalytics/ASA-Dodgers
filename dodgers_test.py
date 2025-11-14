import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import mediapipe as mp


class PitchingAnalyzer:
    def __init__(self):
        """Initialize MediaPipe Pose detection with high accuracy"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        try:
            # Use static_image_mode=False for video, with high accuracy settings
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,  # 0, 1, or 2. Higher = more accurate but slower
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✓ MediaPipe loaded successfully")
        except Exception as e:
            print(f"⚠ Error loading model complexity 2, trying complexity 1...")
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # Lower complexity if download fails
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✓ MediaPipe loaded with complexity 1")
        
    def calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points
        point2 is the vertex (elbow for arm angle)
        Returns angle in degrees
        """
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_shoulder_abduction(self, shoulder, elbow, hip):
        """
        Calculate shoulder abduction angle (between upper arm and trunk)
        """
        return self.calculate_angle(hip, shoulder, elbow)
    
    def calculate_shoulder_external_rotation(self, shoulder, elbow, wrist):
        """
        Calculate shoulder external rotation
        This is the "lay back" angle - how far back the arm rotates
        """
        # Project points to calculate rotation in the coronal plane
        return self.calculate_angle(shoulder, elbow, wrist)
    
    def calculate_arm_slot(self, shoulder, elbow):
        """
        Calculate arm slot angle (relative to horizontal)
        Returns angle in degrees from horizontal
        Over 70° = Overhand, 45-70° = 3/4, Under 45° = Sidearm
        """
        dx = elbow[0] - shoulder[0]
        dy = elbow[1] - shoulder[1]
        angle = np.degrees(np.arctan2(-dy, dx))  # Negative dy because y increases downward
        
        # Normalize to 0-90 range
        angle = abs(angle)
        if angle > 90:
            angle = 180 - angle
            
        return angle
    
    def is_pitcher(self, landmarks, h, w):
        """
        Determine if detected person is the pitcher (not batter/catcher)
        Pitcher is typically in LEFT half of frame and in pitching stance
        """
        try:
            # Get nose position (center of person)
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            nose_x = nose.x * w
            
            # Get shoulders
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            # PITCHER IS IN LEFT PORTION OF FRAME (typically left 60%)
            # This is the key fix - pitcher is on the LEFT, batter is on the right
            if nose_x > w * 0.7:  # If person is in right 40%, likely BATTER not pitcher
                return False
            
            # Additional checks to confirm pitcher
            # Pitcher typically has more vertical arm movement
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            
            # Check if arms are in throwing position (one arm higher than shoulder)
            if (left_elbow.y < left_shoulder.y or right_elbow.y < right_shoulder.y):
                return True
                
            # Pitcher is usually more centered vertically than batter
            nose_y = nose.y * h
            if 0.2 * h < nose_y < 0.8 * h:  # Middle portion of frame
                return True
                
            return True  # Default to true if in left portion
            
        except:
            return False
    
    def get_landmarks(self, landmarks, h, w, side='right'):
        """
        Extract key landmarks for pitching analysis
        side: 'right' or 'left' for throwing arm
        """
        try:
            # First check if this is the pitcher
            if not self.is_pitcher(landmarks, h, w):
                return None
            
            if side == 'right':
                shoulder = [
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h
                ]
                elbow = [
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h
                ]
                wrist = [
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h
                ]
                hip = [
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * h
                ]
            else:  # left
                shoulder = [
                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h
                ]
                elbow = [
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h
                ]
                wrist = [
                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y * h
                ]
                hip = [
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h
                ]
            
            # Check visibility
            if side == 'right':
                vis_threshold = 0.3  # Lower threshold for pitching motion
                if (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility < vis_threshold or
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility < vis_threshold):
                    return None
            
            return shoulder, elbow, wrist, hip
            
        except Exception as e:
            return None
    
    def analyze_video(self, video_path, output_path=None, throwing_arm='right', roi=None):
        """
        Analyze pitching video and extract arm angles
        throwing_arm: 'right' or 'left'
        roi: Region of interest as (x, y, w, h) tuple to focus on pitcher area
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Auto-detect ROI for pitcher (LEFT portion of frame)
        if roi is None:
            roi = (0, int(height * 0.1), int(width * 0.6), int(height * 0.8))  # Left 60% of frame
        
        print(f"\nVideo Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f} seconds")
        print(f"  Analyzing {throwing_arm} arm")
        print(f"  Focus region: Left {int(roi[2]/width*100)}% of frame (PITCHER area)\n")
        
        # Setup video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Store metrics
        frame_data = []
        frame_count = 0
        detected_count = 0
        
        print("Processing video frames...")
        print("Press 'q' to quit\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Draw ROI rectangle on frame for visualization
            roi_x, roi_y, roi_w, roi_h = roi
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), 
                         (0, 255, 0), 2)  # Green for pitcher zone
            cv2.putText(frame, "PITCHER ZONE", (roi_x + 10, roi_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(image_rgb)
            
            # Create a copy for drawing
            annotated_frame = frame.copy()
            
            if results.pose_landmarks:
                # Draw pose landmarks on the frame
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Get landmarks
                landmarks = results.pose_landmarks.landmark
                h, w = frame.shape[:2]
                
                landmark_coords = self.get_landmarks(landmarks, h, w, throwing_arm)
                
                if landmark_coords:
                    detected_count += 1
                    shoulder, elbow, wrist, hip = landmark_coords
                    
                    # Verify pitcher is in ROI
                    roi_x, roi_y, roi_w, roi_h = roi
                    if not (roi_x <= shoulder[0] <= roi_x + roi_w and
                            roi_y <= shoulder[1] <= roi_y + roi_h):
                        # Person detected but not in pitcher zone (likely batter)
                        cv2.putText(annotated_frame, 'BATTER DETECTED (ignored)', 
                                   (width - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 0, 255), 2)
                        # Don't process this detection
                        landmark_coords = None
                    else:
                        # Mark as pitcher
                        cv2.putText(annotated_frame, 'PITCHER DETECTED', 
                                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.8, (0, 255, 0), 2)
                
                if landmark_coords:
                    shoulder, elbow, wrist, hip = landmark_coords
                    
                    # Calculate all angles
                    elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                    shoulder_abduction = self.calculate_shoulder_abduction(shoulder, elbow, hip)
                    shoulder_rotation = self.calculate_shoulder_external_rotation(shoulder, elbow, wrist)
                    arm_slot = self.calculate_arm_slot(shoulder, elbow)
                    
                    # Store data
                    frame_data.append({
                        'frame': frame_count,
                        'time': frame_count / fps,
                        'elbow_angle': elbow_angle,
                        'shoulder_abduction': shoulder_abduction,
                        'shoulder_rotation': shoulder_rotation,
                        'arm_slot': arm_slot
                    })
                    
                    # Display metrics on frame
                    y_offset = 80
                    cv2.putText(annotated_frame, f'Elbow Angle: {int(elbow_angle)}°', 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 35
                    cv2.putText(annotated_frame, f'Shoulder Abduction: {int(shoulder_abduction)}°', 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 35
                    cv2.putText(annotated_frame, f'Arm Slot: {int(arm_slot)}°', 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Highlight key points
                    cv2.circle(annotated_frame, tuple(map(int, shoulder)), 8, (255, 0, 0), -1)
                    cv2.circle(annotated_frame, tuple(map(int, elbow)), 8, (0, 255, 0), -1)
                    cv2.circle(annotated_frame, tuple(map(int, wrist)), 8, (0, 0, 255), -1)
                    
                    # Draw arm lines
                    cv2.line(annotated_frame, tuple(map(int, shoulder)), 
                            tuple(map(int, elbow)), (255, 255, 0), 3)
                    cv2.line(annotated_frame, tuple(map(int, elbow)), 
                            tuple(map(int, wrist)), (255, 255, 0), 3)
            
            # Progress indicator
            progress = int((frame_count / total_frames) * 100)
            cv2.putText(annotated_frame, f'{progress}% | Frame {frame_count}/{total_frames}', 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame
            if output_path:
                out.write(annotated_frame)
            
            # Display
            cv2.imshow('Pitching Analysis (Press Q to quit)', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break
            
            # Progress updates
            if frame_count % 30 == 0 or frame_count == total_frames:
                print(f"  Progress: {progress}% ({frame_count}/{total_frames} frames) - Pitcher Detected: {detected_count}")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Processing complete!")
        print(f"  Total frames: {frame_count}")
        print(f"  Frames with PITCHER detected: {detected_count} ({100*detected_count/frame_count:.1f}%)")
        
        return frame_data
    
    def generate_report(self, frame_data):
        """Generate comprehensive analysis report"""
        if not frame_data:
            print("\n⚠ No PITCHER pose data detected in video")
            print("\nTroubleshooting tips:")
            print("  - Ensure the PITCHER is fully visible in LEFT side of frame")
            print("  - Check that lighting is adequate")
            print("  - Verify the throwing arm side is correct")
            print("  - Use higher resolution video if possible")
            print("  - The batter may be getting detected instead - ensure pitcher is in left portion")
            return
        
        # Extract metrics
        elbow_angles = [d['elbow_angle'] for d in frame_data]
        shoulder_abductions = [d['shoulder_abduction'] for d in frame_data]
        shoulder_rotations = [d['shoulder_rotation'] for d in frame_data]
        arm_slots = [d['arm_slot'] for d in frame_data]
        times = [d['time'] for d in frame_data]
        
        print("\n" + "="*70)
        print(" "*20 + "PITCHING MECHANICS ANALYSIS")
        print("="*70)
        
        print(f"\n📊 DATA SUMMARY")
        print(f"  Analyzed frames: {len(frame_data)}")
        print(f"  Time span: {times[0]:.2f}s - {times[-1]:.2f}s ({times[-1]-times[0]:.2f}s duration)")
        
        print(f"\n🔧 ELBOW ANGLE ANALYSIS")
        print(f"  Maximum Extension: {max(elbow_angles):.1f}° (should be ~170° at release)")
        print(f"  Minimum Angle: {min(elbow_angles):.1f}°")
        print(f"  Average: {np.mean(elbow_angles):.1f}° ± {np.std(elbow_angles):.1f}°")
        
        print(f"\n💪 SHOULDER ABDUCTION ANALYSIS")
        print(f"  Maximum: {max(shoulder_abductions):.1f}° (should be ~90° at foot contact)")
        print(f"  Minimum: {min(shoulder_abductions):.1f}°")
        print(f"  Average: {np.mean(shoulder_abductions):.1f}° ± {np.std(shoulder_abductions):.1f}°")
        
        print(f"\n🔄 SHOULDER ROTATION")
        print(f"  Max External Rotation: {max(shoulder_rotations):.1f}°")
        print(f"  Average: {np.mean(shoulder_rotations):.1f}°")
        
        print(f"\n📐 ARM SLOT CLASSIFICATION")
        avg_slot = np.mean(arm_slots)
        print(f"  Average Arm Angle: {avg_slot:.1f}°")
        if avg_slot > 70:
            slot_type = "OVERHAND (High Slot)"
        elif avg_slot > 45:
            slot_type = "THREE-QUARTERS"
        else:
            slot_type = "SIDEARM (Low Slot)"
        print(f"  Classification: {slot_type}")
        
        print(f"\n✅ MECHANICS EVALUATION")
        
        # Elbow evaluation
        max_elbow = max(elbow_angles)
        if 160 <= max_elbow <= 180:
            print(f"  ✓ Excellent elbow extension at release ({max_elbow:.1f}°)")
        elif 150 <= max_elbow < 160:
            print(f"  ⚠ Good elbow extension, could extend more ({max_elbow:.1f}° vs 170° target)")
        else:
            print(f"  ❌ Poor elbow extension ({max_elbow:.1f}° vs 170° target)")
        
        # Shoulder abduction evaluation
        max_abduction = max(shoulder_abductions)
        if 85 <= max_abduction <= 95:
            print(f"  ✓ Optimal shoulder abduction ({max_abduction:.1f}°)")
        elif 80 <= max_abduction < 85 or 95 < max_abduction <= 100:
            print(f"  ⚠ Acceptable shoulder abduction ({max_abduction:.1f}° vs 90° target)")
        else:
            print(f"  ❌ Suboptimal shoulder abduction ({max_abduction:.1f}° vs 90° target)")
        
        print("="*70 + "\n")
        
        # Generate plots
        self.plot_analysis(times, elbow_angles, shoulder_abductions, 
                          shoulder_rotations, arm_slots)
    
    def plot_analysis(self, times, elbow_angles, shoulder_abductions, 
                     shoulder_rotations, arm_slots):
        """Create comprehensive visualization plots"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # Elbow angle plot
        axes[0].plot(times, elbow_angles, 'b-', linewidth=2, label='Elbow Angle')
        axes[0].axhline(y=170, color='g', linestyle='--', linewidth=1.5, label='Target (170°)')
        axes[0].fill_between(times, 160, 180, alpha=0.2, color='green', label='Optimal Range')
        axes[0].set_ylabel('Angle (degrees)', fontsize=11, fontweight='bold')
        axes[0].set_title('Elbow Extension Through Pitch', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best')
        
        # Shoulder abduction plot
        axes[1].plot(times, shoulder_abductions, 'r-', linewidth=2, label='Shoulder Abduction')
        axes[1].axhline(y=90, color='g', linestyle='--', linewidth=1.5, label='Target (90°)')
        axes[1].fill_between(times, 85, 95, alpha=0.2, color='green', label='Optimal Range')
        axes[1].set_ylabel('Angle (degrees)', fontsize=11, fontweight='bold')
        axes[1].set_title('Shoulder Abduction Through Pitch', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best')
        
        # Shoulder rotation plot
        axes[2].plot(times, shoulder_rotations, 'orange', linewidth=2, label='External Rotation')
        axes[2].set_ylabel('Angle (degrees)', fontsize=11, fontweight='bold')
        axes[2].set_title('Shoulder External Rotation', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='best')
        
        # Arm slot plot
        axes[3].plot(times, arm_slots, 'purple', linewidth=2, label='Arm Slot')
        axes[3].axhline(y=70, color='gray', linestyle=':', label='Overhand Threshold')
        axes[3].axhline(y=45, color='gray', linestyle=':', label='Sidearm Threshold')
        axes[3].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        axes[3].set_ylabel('Angle (degrees)', fontsize=11, fontweight='bold')
        axes[3].set_title('Arm Slot Angle', fontsize=13, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc='best')
        
        plt.tight_layout()
        plt.savefig('pitching_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Analysis plot saved as 'pitching_analysis.png'")
        plt.show()


if __name__ == "__main__":
    print("="*70)
    print(" "*20 + "PITCHING ANALYZER v2.0")
    print("="*70)
    print("IMPORTANT: Pitcher should be in LEFT side of frame")
    print("Batter detection will be ignored automatically")
    
    # Get video path
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = input("\nEnter path to pitching video: ").strip()
        if not video_path:
            print("\nUsage: python pitch_analyzer.py video.mp4")
            sys.exit(0)
    
    # Validate file
    if not os.path.exists(video_path):
        print(f"\n❌ Error: Video file '{video_path}' not found!")
        sys.exit(1)
    
    # Get throwing arm
    throwing_arm = input("Throwing arm side (right/left) [default: right]: ").strip().lower()
    if throwing_arm not in ['right', 'left']:
        throwing_arm = 'right'
    
    try:
        # Initialize analyzer
        analyzer = PitchingAnalyzer()
        
        # Analyze video
        output_path = "analyzed_pitch.mp4"
        frame_data = analyzer.analyze_video(video_path, output_path, throwing_arm)
        
        # Generate report
        if frame_data:
            print(f"\n✓ Annotated video saved: {output_path}")
            analyzer.generate_report(frame_data)
        else:
            print("\n⚠ No PITCHER data collected")
            print("Make sure the pitcher is clearly visible in the LEFT side of the frame")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()