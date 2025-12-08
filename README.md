[English](README.md) | [Русский](docs/README-RU.md)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-FF6F00?style=for-the-badge&logo=google&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![NumPy](https://img.shields.io/badge/NumPy-<2.0-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Active-FF6B6B?style=for-the-badge&logo=opencv&logoColor=white)

</div>

# VisionPlay Board

> Interactive computer vision game application for electronic board that enables gesture-controlled games using OpenCV and MediaPipe pose detection.

## Description

VisionPlay Board is an application that uses computer vision to create interactive games. The application runs in fullscreen OpenCV mode and allows you to activate games either by mouse click or by detecting a person through MediaPipe.

## Features

- **Fullscreen mode** with OpenCV
- **Simplified architecture** with 2 layers: background and skeleton
- **Multi-threaded processing** of pose detection
- **Human detection** via MediaPipe with continuous skeleton rendering
- **Mirrored camera display** for natural interaction
- **Interactive game tiles** on the main screen
- **Automatic launch** of random game when human is detected (3 seconds)
- **"Skeleton Viewer" game** - view skeleton for up to 3 people
- **Performance monitoring** with FPS and statistics

## Installation

1. Make sure you have Python 3.11.11 installed (via pyenv):
   ```bash
   pyenv install 3.11.11
   pyenv local 3.11.11
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running

### Via pyenv (recommended):
```bash
# Make sure pyenv is configured
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv init --path)"

# Install Python 3.11.11 if not installed
pyenv install 3.11.11
pyenv shell 3.11.11

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Via startup scripts:
```bash
# Quick start (recommended)
./scripts/start.sh

# Or full script with installation
./scripts/run.sh
```

## Controls

- **ESC** - exit application or return to main menu
- **Q** - exit application
- **S** - show performance statistics
- **Mouse click** - activate game on tile
- **Human detection** - automatic launch of random game after 3 seconds

## Configuration

Application settings are in the `.env` file:

### Main settings:
- `CAMERA_INDEX` - camera index (default 0)
- `CAMERA_WIDTH/HEIGHT` - camera resolution
- `FULLSCREEN_MODE` - fullscreen mode (true/false)
- `HUMAN_DETECTION_TIMEOUT` - human detection time for auto-launch

### Pose detection settings:
- `SHOW_BODY_LANDMARKS` - show body landmarks
- `SHOW_FACE_LANDMARKS` - show face landmarks
- `SHOW_HAND_LANDMARKS` - show hand landmarks
- `SHOW_POSE_CONNECTIONS` - show body connections
- `SHOW_FACE_CONNECTIONS` - show face connections
- `SHOW_HAND_CONNECTIONS` - show hand connections

### Performance settings:
- `SHOW_FPS` - show FPS on screen
- `SHOW_STATISTICS_ON_S_KEY` - show statistics on S key
- `ENABLE_POSE_DETECTION` - enable/disable pose detection (false for better performance)

### Game settings:
- `MAX_PEOPLE_IN_FRAME` - maximum number of people in frame (up to 3)

## Project Structure

```
VisionPlay-Board/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── README.md              # Documentation
├── .env                   # Configuration
├── .python-version        # Python version
├── scripts/               # Startup scripts
│   ├── start.sh          # Quick start
│   └── run.sh            # Full start with installation
├── tests/                 # Test files
│   ├── test.py           # Camera performance test
│   ├── minimal_test.py   # Minimal camera test
│   ├── simple_test.py    # Simple camera test
│   └── camera_check.py   # Camera check
├── docs/                  # Documentation
│   └── camera_report.md  # Camera report
├── models/                # Machine learning models
│   └── yolov8n.pt        # YOLO person detection model
└── src/                   # Application source code
    ├── app.py             # Main application
    ├── utils/
    │   ├── config.py      # Configuration management
    │   ├── pose_detector.py # MediaPipe pose detection
    │   ├── layers.py      # Layer system for rendering
    │   ├── thread_manager.py # Multi-threaded processing
    │   ├── scaling.py     # Adaptive scaling
    │   ├── yolo_person_detector.py # YOLO person detector
    │   └── yolo_holistic_detector.py # YOLO + MediaPipe detector
    └── games/
        ├── base_game.py   # Base game class
        ├── skeleton_viewer_game.py # "Skeleton Viewer" game
        └── hide_and_seek_game.py # "Hide and Seek" game
```

## Application Architecture

### Simplified rendering system:
1. **Background Layer** - camera + UI elements
2. **Skeleton Layer** - pose detection and skeleton rendering

### Multi-threaded processing:
- **Main Thread** - frame capture from camera and rendering
- **Pose Detection Thread** - MediaPipe processing in separate thread

### Advantages of simplified architecture:
- **Improved performance** - fewer layers, faster rendering
- **Simplicity** - easy to understand and modify
- **Monitoring** - FPS and statistics tracking
- **Adaptability** - automatic performance tuning

## "Skeleton Viewer" Game

### Features:
- **Skeleton viewing** for up to 3 people simultaneously
- **Body, face and hand detection** with different colors for each person
- **Automatic exit** if no people in frame for 10 seconds
- **Adaptive performance** depending on movement speed
- **Display configuration** via `.env` file

### Display:
- **Green skeleton** - first body
- **Red skeleton** - second body  
- **Blue skeleton** - third body
- **Colored landmarks** for face and hands of each person
- **People counter** in frame
- **Exit warnings**

## Requirements

- Python 3.11.11
- OpenCV 4.9.x
- MediaPipe 0.10.x
- NumPy < 2.0
- Web camera

## Troubleshooting

### Dependency issues:
If you encounter version conflicts, reinstall dependencies:
```bash
pip install "opencv-python<4.10" "numpy<2" mediapipe python-dotenv
```

### Camera issues:
- Check camera index in `.env` file
- Make sure camera is not being used by other applications
- Try changing `CAMERA_INDEX` to 1, 2, etc.

### Pyenv issues:
Make sure pyenv is configured correctly:
```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

### Permission issues:
If you encounter permission errors with `.python-version`, use:
```bash
# Instead of pyenv local use pyenv shell
pyenv shell 3.11.11
python main.py
```

### MediaPipe errors:
If you encounter MediaPipe type errors, make sure you're using a compatible version:
```bash
pip install mediapipe==0.10.21
```

### Performance issues:
If the image is lagging, try the following solutions:

1. **Disable pose detection:**
   ```bash
   ENABLE_POSE_DETECTION=false
   ```

2. **Reduce camera resolution:**
   ```bash
   CAMERA_WIDTH=640
   CAMERA_HEIGHT=480
   ```

3. **Configure FPS (if needed):**
   ```bash
   CAMERA_FPS=30  # For 640x480
   CAMERA_FPS=15  # For 1280x720
   ```

4. **Camera diagnostics:**
   ```bash
   # Simple camera test
   python tests/test.py
   
   # Minimal test
   python tests/minimal_test.py
   
   # All cameras diagnostics
   python tests/simple_test.py
   
   # Camera check
   python tests/camera_check.py
   ```

5. **Check MJPEG support:**
   Make sure `USE_MJPEG_CODEC=true` in `.env` file.

### Wayland issues:
If you see warning "Ignoring XDG_SESSION_TYPE=wayland", this means the system is using Wayland instead of X11. The fix is already included in the code, but you can also set the environment variable:
```bash
export QT_QPA_PLATFORM=xcb
```
