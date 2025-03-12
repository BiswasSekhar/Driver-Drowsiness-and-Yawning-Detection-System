import cv2
import numpy as np
import time
import pygame
import mediapipe as mp
import math
import os
from scipy.spatial import distance

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize pygame for alert sound
pygame.mixer.init()
sound_available = False
try:
    # Use our local sound file
    sound_path = 'sounds/alarm.wav'
    if os.path.exists(sound_path):
        pygame.mixer.music.load(sound_path)
        sound_available = True
        print("Alert sound loaded successfully.")
    else:
        print("Alert sound file not found. No audio alerts will be played.")
except Exception as e:
    print(f"Error loading sound file: {e}. No audio alerts will be played.")

# Constants - More specific eye landmarks for MediaPipe Face Mesh
# Using specific landmarks for more accurate eye state detection
# These landmarks represent the upper and lower eyelids directly
LEFT_EYE = {
    "upper1": 159, "upper2": 145,
    "lower1": 158, "lower2": 153,
    "left_corner": 130, "right_corner": 243
}

RIGHT_EYE = {
    "upper1": 386, "upper2": 374,
    "lower1": 387, "lower2": 380,
    "left_corner": 463, "right_corner": 359
}

# Complete eye landmarks for visualization
LEFT_EYE_INDICES = [LEFT_EYE["upper1"], LEFT_EYE["upper2"], LEFT_EYE["lower1"], LEFT_EYE["lower2"], 
                    LEFT_EYE["left_corner"], LEFT_EYE["right_corner"]]
RIGHT_EYE_INDICES = [RIGHT_EYE["upper1"], RIGHT_EYE["upper2"], RIGHT_EYE["lower1"], RIGHT_EYE["lower2"],
                     RIGHT_EYE["left_corner"], RIGHT_EYE["right_corner"]]

# Mouth indices
UPPER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
LOWER_LIP_INDICES = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Detection parameters - adjusted for more sensitive eye closure detection
EYE_RATIO_THRESHOLD = 0.35  # Reset to a standard threshold to start with
EYE_AR_CONSEC_FRAMES = 10  # Reduced to detect drowsiness faster
MOUTH_AR_THRESHOLD = 0.5  # Threshold for yawn detection
DROWSY_TIME = 1.0  # Reduced time to determine drowsiness (seconds)
PROLONGED_YAWN_TIME = 3.0  # Time in seconds to consider a yawn as prolonged
PROLONGED_EYE_CLOSURE_TIME = 2.0  # Time in seconds to consider eye closure as dangerous
DEBUG = True  # Enable debug information display

# Add low light detection and enhancement parameters
LOW_LIGHT_THRESHOLD = 50  # Average brightness level to consider as low light
BRIGHTNESS_BOOST = 50     # How much to boost brightness in low light

# Counters and state variables
EYE_COUNTER = 0
MOUTH_COUNTER = 0
ALARM_ON = False
yawns = 0
yawn_status = False
drowsy_start_time = None
yawn_start_time = None
alert_messages = []  # For storing multiple alert messages
calibration_needed = True  # Flag to indicate if calibration is still needed

# Function to calculate Eye Aspect Ratio (EAR) - improved version
def calculate_eye_aspect_ratio(landmarks, eye_points):
    """
    Calculate the eye aspect ratio (EAR) which is:
    EAR = (vertical distance) / (horizontal distance)
    """
    # Get coordinates for the eye landmarks
    upper1 = np.array([landmarks.landmark[eye_points["upper1"]].x, landmarks.landmark[eye_points["upper1"]].y])
    upper2 = np.array([landmarks.landmark[eye_points["upper2"]].x, landmarks.landmark[eye_points["upper2"]].y])
    lower1 = np.array([landmarks.landmark[eye_points["lower1"]].x, landmarks.landmark[eye_points["lower1"]].y])
    lower2 = np.array([landmarks.landmark[eye_points["lower2"]].x, landmarks.landmark[eye_points["lower2"]].y])
    left_corner = np.array([landmarks.landmark[eye_points["left_corner"]].x, landmarks.landmark[eye_points["left_corner"]].y])
    right_corner = np.array([landmarks.landmark[eye_points["right_corner"]].x, landmarks.landmark[eye_points["right_corner"]].y])
    
    # Calculate vertical distances (between upper and lower eyelids)
    dist1 = distance.euclidean(upper1, lower1)
    dist2 = distance.euclidean(upper2, lower2)
    
    # Calculate horizontal distance (width of eye)
    horizontal_dist = distance.euclidean(left_corner, right_corner)
    
    # Calculate EAR
    ear = (dist1 + dist2) / (2.0 * horizontal_dist) if horizontal_dist > 0 else 0
    
    return ear, [upper1, upper2, lower1, lower2, left_corner, right_corner]

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(upper_lip, lower_lip, face_landmarks):
    # Get coordinates for upper and lower lips
    upper_coords = np.array([[face_landmarks.landmark[index].x, face_landmarks.landmark[index].y] 
                            for index in upper_lip])
    lower_coords = np.array([[face_landmarks.landmark[index].x, face_landmarks.landmark[index].y] 
                            for index in lower_lip])
    
    # Calculate distances
    # Vertical distance between upper and lower lip
    mouth_height = np.mean([distance.euclidean(upper_coords[i], lower_coords[i]) 
                           for i in range(len(upper_coords))])
    
    # Width of the mouth
    mouth_width = distance.euclidean(upper_coords[0], upper_coords[-1])
    
    # Calculate MAR
    mar = mouth_height / mouth_width if mouth_width > 0 else 0
    return mar

# Function for visual alert with time limit
def draw_alert(image, alert_text, position=(30, 60), color=(0, 0, 255)):
    cv2.putText(image, alert_text, position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

# Function to play different types of alarms
def sound_alarm(alarm_type="drowsy"):
    global ALARM_ON
    if not ALARM_ON:
        ALARM_ON = True
        if sound_available:
            try:
                # Different alarm patterns based on alert type
                if alarm_type == "drowsy":
                    # Regular alarm for drowsiness
                    pygame.mixer.music.play(-1)  # Loop the sound
                elif alarm_type == "yawn":
                    # Play pattern for yawning (play-pause-play)
                    pygame.mixer.music.play()
                    pygame.time.delay(200)
                    pygame.mixer.music.play()
                print(f"Playing {alarm_type} alarm sound")
            except Exception as e:
                print(f"Error playing sound alert: {e}")

def stop_alarm():
    global ALARM_ON
    if ALARM_ON:
        ALARM_ON = False
        if sound_available:
            try:
                pygame.mixer.music.stop()
                print("Stopping alarm sound")
            except Exception as e:
                print(f"Error stopping sound: {e}")

# Function to enhance image in low light conditions
def enhance_low_light(frame):
    """
    Apply image processing techniques to improve visibility in low light conditions
    """
    # Convert to HSV for better brightness manipulation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Calculate average brightness
    avg_brightness = np.mean(hsv[:, :, 2])
    
    # If in low light conditions, enhance brightness and contrast
    if avg_brightness < LOW_LIGHT_THRESHOLD:
        # Increase brightness (V channel in HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + BRIGHTNESS_BOOST, 0, 255).astype('uint8')
        
        # Convert back to BGR
        enhanced_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Apply adaptive histogram equalization for better contrast
        lab = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        
        # Merge back the channels
        merged = cv2.merge((cl, a, b))
        enhanced_frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        # Apply mild denoising if needed
        enhanced_frame = cv2.fastNlMeansDenoisingColored(enhanced_frame, None, 5, 5, 7, 21)
        
        # Display text indicating low light enhancement is active
        cv2.putText(frame, "Low Light Enhancement Active", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
        
        return enhanced_frame, True
    
    return frame, False

# Debug function to draw eye landmarks and measurements
def draw_eye_debug(frame, points, ear, is_closed=False):
    h, w, _ = frame.shape
    
    # Convert normalized points to pixel coordinates
    pixels = []
    for point in points:
        pixels.append((int(point[0] * w), int(point[1] * h)))
    
    # Draw points
    for px, py in pixels:
        cv2.circle(frame, (px, py), 2, (0, 255, 255) if not is_closed else (0, 0, 255), -1)
        
    # Draw horizontal line (eye width)
    cv2.line(frame, pixels[4], pixels[5], (0, 255, 0), 1)
    
    # Draw vertical lines (eye height)
    cv2.line(frame, pixels[0], pixels[2], (0, 0, 255), 1)
    cv2.line(frame, pixels[1], pixels[3], (0, 0, 255), 1)
    
    # Add EAR text
    midpoint = ((pixels[4][0] + pixels[5][0]) // 2, (pixels[4][1] + pixels[5][1]) // 2)
    cv2.putText(frame, f"{ear:.3f}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

def main():
    global EYE_COUNTER, MOUTH_COUNTER, ALARM_ON, yawns, yawn_status
    global drowsy_start_time, yawn_start_time, alert_messages, EYE_RATIO_THRESHOLD, calibration_needed
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Get screen dimensions
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video. Exiting.")
        return
    
    height, width = frame.shape[:2]
    
    # Initialize FaceMesh with refined landmarks for better eye detection
    # Adjust the min_detection_confidence to be more tolerant in low light
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,  # Reduced threshold for better detection in low light
        min_tracking_confidence=0.4) as face_mesh:  # Also reduced for better tracking
        
        print("Advanced Driver Drowsiness and Yawning Detection System started")
        print("Press 'q' to quit, 'c' to recalibrate")
        
        # For calibration
        calibration_stage = 0  # 0=not started, 1=eyes open, 2=eyes closed, 3=complete
        calibration_frames = 0
        calibration_max_frames = 50
        open_ear_values = []
        closed_ear_values = []
        
        # For running average of eye states to reduce false positives
        ear_history = []
        history_max = 10
        
        # Start time for FPS calculation
        start_time = time.time()
        frame_count = 0
        
        # For night mode control
        night_mode_active = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read from webcam!")
                break
            
            # Enhance frame if in low light conditions
            original_frame = frame.copy()
            frame, is_low_light = enhance_low_light(frame)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with FaceMesh
            results = face_mesh.process(rgb_frame)
            
            # If face is not detected in enhanced frame and we're in low light
            # try with even more extreme enhancement
            if is_low_light and not results.multi_face_landmarks:
                # Apply more aggressive enhancement
                hsv = cv2.cvtColor(original_frame, cv2.COLOR_BGR2HSV)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] + BRIGHTNESS_BOOST*1.5, 0, 255).astype('uint8')
                extreme_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                # Try processing with more enhanced frame
                extreme_rgb = cv2.cvtColor(extreme_frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(extreme_rgb)
                
                if results.multi_face_landmarks:
                    frame = extreme_frame
                    cv2.putText(frame, "Extreme Low Light Mode", (10, 440),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Handle calibration UI
            if calibration_needed:
                if calibration_stage == 0:
                    # Starting calibration
                    cv2.putText(frame, "Starting calibration...", (30, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Keep eyes OPEN and look at camera", (30, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    calibration_stage = 1
                    calibration_frames = 0
                elif calibration_stage == 1:
                    # Collecting open eyes data
                    cv2.putText(frame, f"Eyes OPEN calibration: {calibration_frames}/{calibration_max_frames}", (30, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif calibration_stage == 2:
                    # Collecting closed eyes data
                    cv2.putText(frame, f"Eyes CLOSED calibration: {calibration_frames}/{calibration_max_frames}", (30, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Clear alert messages at each frame
            alert_messages = []
            
            # Draw face mesh annotations
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate Eye Aspect Ratio (EAR) for both eyes
                    left_ear, left_points = calculate_eye_aspect_ratio(face_landmarks, LEFT_EYE)
                    right_ear, right_points = calculate_eye_aspect_ratio(face_landmarks, RIGHT_EYE)
                    
                    # Average EAR
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # Add to history for smoothing
                    ear_history.append(avg_ear)
                    if len(ear_history) > history_max:
                        ear_history.pop(0)
                    
                    # Get smoothed EAR
                    smoothed_ear = sum(ear_history) / len(ear_history)
                    
                    if DEBUG:
                        # Draw eye landmarks for debugging
                        is_closed = smoothed_ear < EYE_RATIO_THRESHOLD
                        draw_eye_debug(frame, left_points, left_ear, is_closed)
                        draw_eye_debug(frame, right_points, right_ear, is_closed)
                    
                    # Handle calibration data collection
                    if calibration_needed:
                        if calibration_stage == 1:  # Open eyes
                            open_ear_values.append(avg_ear)
                            calibration_frames += 1
                            if calibration_frames >= calibration_max_frames:
                                # Move to closed eyes calibration
                                calibration_stage = 2
                                calibration_frames = 0
                                cv2.putText(frame, "Now CLOSE your eyes", (30, 90),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                # Small delay to give user time to close eyes
                                for i in range(3, 0, -1):
                                    cv2.putText(frame, f"Closing eyes in {i}...", (30, 120),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    cv2.imshow('Advanced Driver Drowsiness Detection', frame)
                                    cv2.waitKey(1000)  # Wait 1 second
                        elif calibration_stage == 2:  # Closed eyes
                            closed_ear_values.append(avg_ear)
                            calibration_frames += 1
                            if calibration_frames >= calibration_max_frames:
                                # Calculate threshold from collected values
                                open_avg = np.mean(open_ear_values)
                                closed_avg = np.mean(closed_ear_values)
                                # Set threshold between open and closed
                                EYE_RATIO_THRESHOLD = (open_avg + closed_avg) / 2.0
                                
                                # Finalize calibration
                                calibration_stage = 3
                                calibration_needed = False
                                print(f"Calibration complete! Threshold set to {EYE_RATIO_THRESHOLD:.4f}")
                                print(f"  Open eyes avg: {open_avg:.4f}")
                                print(f"  Closed eyes avg: {closed_avg:.4f}")
                                
                                # Clear history for fresh start
                                ear_history = []
                    else:
                        # If calibration is complete, proceed with detection
                        
                        # Draw face mesh with custom settings
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                        
                        # Display EAR value with threshold
                        cv2.putText(frame, f"EAR: {smoothed_ear:.4f}/{EYE_RATIO_THRESHOLD:.4f}", (width-220, height-70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Eyes are considered closed if EAR is below threshold
                        # In low light, we might need to be more sensitive
                        if is_low_light:
                            # Slightly more sensitive threshold in low light
                            eyes_closed = smoothed_ear < (EYE_RATIO_THRESHOLD * 0.9)
                        else:
                            eyes_closed = smoothed_ear < EYE_RATIO_THRESHOLD
                        
                        # Eyes closed detection
                        if eyes_closed:
                            EYE_COUNTER += 1
                            cv2.putText(frame, "Eyes Closed", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            # Start timing for drowsiness
                            if drowsy_start_time is None:
                                drowsy_start_time = time.time()
                            
                            # Calculate how long eyes have been closed
                            eyes_closed_time = time.time() - drowsy_start_time
                                
                            # Check if eyes have been closed for enough time - regular drowsiness alert
                            if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES and eyes_closed_time >= DROWSY_TIME:
                                alert_messages.append(("DROWSINESS ALERT!", (30, 60)))
                                sound_alarm("drowsy")
                            
                            # Additional check for prolonged eye closure - more severe warning
                            if eyes_closed_time >= PROLONGED_EYE_CLOSURE_TIME:
                                alert_messages.append(("DANGER! EYES CLOSED TOO LONG!", (30, 120)))
                        else:
                            # Reset eye counter and timer if eyes are open
                            EYE_COUNTER = 0
                            drowsy_start_time = None
                            
                            cv2.putText(frame, "Eyes Open", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Turn off alarm if it was on
                            if ALARM_ON:
                                stop_alarm()
                        
                        # Calculate Mouth Aspect Ratio (MAR) for yawn detection
                        mar = mouth_aspect_ratio(UPPER_LIP_INDICES, LOWER_LIP_INDICES, face_landmarks)
                        
                        # Display MAR value
                        cv2.putText(frame, f"MAR: {mar:.2f}", (width-150, height-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Yawn detection logic
                        mouth_open = mar > MOUTH_AR_THRESHOLD
                        
                        if mouth_open:
                            # ... existing yawn detection code ...
                            MOUTH_COUNTER += 1
                            
                            # Start timing when mouth opens (potential yawn starts)
                            if yawn_start_time is None:
                                yawn_start_time = time.time()
                            
                            # Calculate how long the mouth has been open
                            mouth_open_time = time.time() - yawn_start_time
                            
                            if MOUTH_COUNTER >= 10:  # Require consecutive frames of open mouth
                                cv2.putText(frame, "Yawning", (30, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                
                                # Count yawn if not previously in yawn state
                                if not yawn_status:
                                    yawn_status = True
                                    yawns += 1
                            
                            # Check for prolonged yawning
                            if mouth_open_time >= PROLONGED_YAWN_TIME:
                                alert_messages.append(("PROLONGED YAWNING DETECTED!", (30, 150)))
                                
                                # If prolonged yawning coincides with drowsiness indicators, it's more concerning
                                if EYE_COUNTER > 0:
                                    alert_messages.append(("EXTREME FATIGUE WARNING!", (30, 180)))
                                    sound_alarm("yawn")
                        else:
                            # Reset mouth counter when mouth closes
                            MOUTH_COUNTER = max(0, MOUTH_COUNTER-1)  # Gradual decrease for smoother detection
                            if MOUTH_COUNTER == 0:
                                yawn_status = False
                                yawn_start_time = None
            else:
                # No face detected
                if not calibration_needed:
                    EYE_COUNTER += 1
                    if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES:
                        cv2.putText(frame, "NO FACE DETECTED - Check Driver!", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Start timing for alert
                        if drowsy_start_time is None:
                            drowsy_start_time = time.time()
                        elif time.time() - drowsy_start_time >= DROWSY_TIME:
                            sound_alarm()
            
            # Display all accumulated alerts
            for i, (message, pos) in enumerate(alert_messages):
                draw_alert(frame, message, pos)
                
            # Display yawn counter
            cv2.putText(frame, f"Yawn Count: {yawns}", (width-150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display light conditions info
            if is_low_light:
                cv2.putText(frame, "Low Light Detected", (width-220, height-120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                cv2.putText(frame, f"FPS: {fps:.2f}", (width-120, height-100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Reset for next calculation
                frame_count = 0
                start_time = time.time()
            
            # Display the resulting frame
            cv2.imshow('Advanced Driver Drowsiness Detection', frame)
            
            # Allow toggling night mode manually with 'n' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Restart calibration
                calibration_needed = True
                calibration_stage = 0
                print("Restarting calibration...")
            elif key == ord('n'):
                # Toggle night mode
                night_mode_active = not night_mode_active
                if night_mode_active:
                    BRIGHTNESS_BOOST = 70  # Increase boost when manually activated
                    print("Night mode activated")
                else:
                    BRIGHTNESS_BOOST = 50  # Reset to default
                    print("Night mode deactivated")
        
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    main()