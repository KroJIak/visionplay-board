# Camera Test Report

## Summary
Successfully tested and optimized camera setup for VisionPlay-Board project.

## Available Cameras

### 1. Logitech C920 HD Pro Webcam (Camera Index 0)
- **Status**: ✅ Working (Optimized)
- **Performance**: 24.8 FPS @ 640x480 MJPEG
- **Device**: `/dev/video0`
- **Backend**: V4L2 (Linux)
- **Optimization**: Required v4l2-ctl settings for proper performance

### 2. USB Camera (Camera Index 2) 
- **Status**: ✅ Working (Excellent)
- **Performance**: 28.0 FPS @ 640x480 MJPEG
- **Device**: `/dev/video2`
- **Backend**: V4L2 (Linux)
- **Note**: Best performing camera

### 3. USB Camera (Camera Index 4)
- **Status**: ✅ Working (Poor)
- **Performance**: 13.8 FPS @ 640x360
- **Device**: `/dev/video4`
- **Backend**: V4L2 (Linux)
- **Note**: Lower resolution, slower performance

## Optimization Applied

### For Logitech C920:
```bash
# Load UVC driver
sudo modprobe uvcvideo

# Set MJPEG format
v4l2-ctl --device=/dev/video0 --set-fmt-video=width=640,height=480,pixelformat=MJPG

# Set frame rate
v4l2-ctl --device=/dev/video0 --set-parm=30
```

### OpenCV Settings:
```python
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
```

## Recommendations

### For VisionPlay-Board:
1. **Primary Camera**: Use Camera Index 2 (USB Camera) - best performance
2. **Secondary Camera**: Use Camera Index 0 (Logitech C920) - good quality, optimized
3. **Avoid**: Camera Index 4 - poor performance

### Camera Selection Logic:
```python
# Try cameras in order of preference
camera_indices = [2, 0, 4]  # Best to worst performance

for camera_index in camera_indices:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Using camera {camera_index}")
            break
    cap.release()
```

## Troubleshooting

### If Logitech C920 is slow:
1. Run optimization commands above
2. Check if camera is used by other applications
3. Try different USB port
4. Check USB power (use powered USB hub if needed)

### If cameras not detected:
1. Check USB connections
2. Run `lsusb` to verify device detection
3. Check `dmesg | grep -i usb` for errors
4. Try `sudo modprobe uvcvideo`

## Test Scripts

- `simple_test.py` - Comprehensive camera testing with OpenCV
- `camera_check.py` - Basic camera detection without OpenCV dependency

Both scripts are available in the project directory and can be run with:
```bash
python3 simple_test.py
python3 camera_check.py
```

## Next Steps

1. Update VisionPlay-Board to use Camera Index 2 as primary
2. Implement camera fallback logic (2 → 0 → 4)
3. Apply Logitech optimization settings in the application
4. Test with the actual VisionPlay-Board application

