# Driver Drowsiness and Yawning Detection System

A computer vision-based system that detects driver drowsiness and yawning in real-time, providing alerts to prevent accidents caused by fatigue.

![Driver Drowsiness Detection System](screenshots/main_screen.png)

## Table of Contents
- [Features](#features)
- [How It Works](#how-it-works)
- [Technical Details](#technical-details)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [System Parameters](#system-parameters)
- [Low Light Performance](#low-light-performance)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Real-time drowsiness detection** - Monitors eye closure patterns to detect signs of drowsiness
- **Yawning detection** - Identifies when the driver is yawning, a key indicator of fatigue
- **Audio alerts** - Provides sound alerts when drowsiness or prolonged yawning is detected
- **Visual warnings** - On-screen alerts with different severity levels
- **Low-light enhancement** - Specialized night mode for better detection in dark conditions
- **Personalized calibration** - Customizes detection thresholds to individual users
- **Face absence detection** - Alerts when the driver's face is not detected (looking away/nodding off)

## How It Works

The system uses computer vision and facial landmark detection to continuously monitor the driver's face for signs of drowsiness:

1. **Face Detection**: Detects the driver's face in each video frame
2. **Facial Landmark Detection**: Tracks 468 specific points on the face using MediaPipe Face Mesh
3. **Eye State Monitoring**: Calculates the Eye Aspect Ratio (EAR) to determine if eyes are open or closed
4. **Yawn Detection**: Measures mouth opening to detect yawning patterns
5. **Alert System**: Triggers warnings when signs of drowsiness are detected

![System Workflow](screenshots/system_workflow.png)

## Technical Details

The system leverages several advanced computer vision techniques:

- **MediaPipe Face Mesh**: For highly accurate facial landmark detection
- **Eye Aspect Ratio (EAR)**: Mathematical measurement of eye openness
- **Adaptive Thresholding**: Dynamic adjustment based on calibration and lighting conditions
- **Rolling Average Filtering**: To reduce false positives and increase accuracy
- **Low-light Image Enhancement**: For reliable performance at night or in poor lighting

## Requirements

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python     | ≥ 3.8   | Base programming language |
| OpenCV     | ≥ 4.5   | Computer vision operations |
| MediaPipe  | ≥ 0.8   | Facial landmark detection |
| NumPy      | ≥ 1.20  | Numerical operations |
| SciPy      | ≥ 1.7   | Distance calculations |
| PyGame     | ≥ 2.0   | Audio alert system |

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/driver-drowsiness-detection.git
   cd driver-drowsiness-detection
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Generate the alarm sound (if not already present):
   ```bash
   python create_alarm.py
   ```

## Usage

1. Run the main application:
   ```bash
   python app.py
   ```

2. When first launched, the system will guide you through a calibration process:
   - Look directly at the camera with eyes fully open
   - Close your eyes when prompted
   - The system will calculate personalized threshold values

3. Key commands during operation:
   - Press `q` to quit the application
   - Press `c` to recalibrate the system
   - Press `n` to toggle night mode for low-light conditions

![Calibration Process](screenshots/calibration.png)

## System Parameters

The following parameters can be adjusted in the code to fine-tune the system:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EYE_RATIO_THRESHOLD` | 0.35 (auto-calibrated) | Threshold for eye closure detection |
| `EYE_AR_CONSEC_FRAMES` | 10 | Consecutive frames of eye closure before alert |
| `MOUTH_AR_THRESHOLD` | 0.5 | Threshold for yawn detection |
| `DROWSY_TIME` | 1.0 | Time in seconds before drowsiness alert |
| `PROLONGED_EYE_CLOSURE_TIME` | 2.0 | Time for severe drowsiness warning |
| `PROLONGED_YAWN_TIME` | 3.0 | Time threshold for prolonged yawning |
| `LOW_LIGHT_THRESHOLD` | 50 | Brightness level for night mode activation |

## Low Light Performance

The system includes specialized low-light enhancement for night driving:

- **Automatic brightness adjustment**: Detects low-light conditions and enhances image
- **CLAHE enhancement**: Improves contrast for better facial landmark detection
- **Manual night mode**: Can be toggled with the 'n' key for challenging conditions

![Low Light Mode](screenshots/low_light_mode.png)

## Screenshots

### Normal Operation
![Normal Operation](screenshots/normal_operation.png)

### Drowsiness Detected
![Drowsiness Alert](screenshots/drowsiness_alert.png)

### Yawning Detection
![Yawn Detection](screenshots/yawn_detection.png)

### Low Light Mode
![Night Mode](screenshots/night_mode.png)

## Contributing

Contributions are welcome! If you'd like to improve the system:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: The screenshot placeholders in this README need to be replaced with actual screenshots from your system. Take screenshots of the application during different states (normal operation, drowsiness detected, yawning detected, night mode) and place them in a `screenshots` folder in your repository.