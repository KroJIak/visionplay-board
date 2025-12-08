#!/usr/bin/env python3
"""
Simple camera test to diagnose performance issues.
"""

import cv2
import time
import os
import glob

def find_all_cameras():
    """Find all available cameras on the system."""
    print("🔍 Searching for all available cameras...")
    
    cameras_found = []
    
    # Check /dev/video* devices with detailed info
    video_devices = glob.glob('/dev/video*')
    print(f"Found {len(video_devices)} video devices: {video_devices}")
    
    # Get detailed info about video devices
    for device in video_devices:
        try:
            # Try to get device info using v4l2-ctl if available
            import subprocess
            result = subprocess.run(['v4l2-ctl', '--device', device, '--info'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"  📹 {device}: {result.stdout.strip()}")
            else:
                print(f"  📹 {device}: Basic device (no v4l2-ctl info)")
        except:
            print(f"  📹 {device}: Basic device")
    
    # Check camera indices 0-10 with different backends
    backends_to_try = [
        (cv2.CAP_V4L2, "V4L2 (Linux)"),
        (cv2.CAP_ANY, "Any backend"),
        (cv2.CAP_FFMPEG, "FFMPEG")
    ]
    
    for camera_index in range(11):
        print(f"\n📹 Testing camera index {camera_index}...")
        
        camera_working = False
        
        for backend, backend_name in backends_to_try:
            print(f"  🔧 Trying {backend_name}...")
            
            cap = cv2.VideoCapture(camera_index, backend)
            
            if cap.isOpened():
                # Try to read a frame to verify it's working
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    print(f"    ✅ Camera {camera_index} ({backend_name}): {width}x{height} @ {fps}fps")
                    cameras_found.append({
                        'index': camera_index,
                        'backend': backend_name,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'working': True
                    })
                    camera_working = True
                    break
                else:
                    print(f"    ❌ Camera {camera_index} ({backend_name}): Opens but can't read frames")
            else:
                print(f"    ❌ Camera {camera_index} ({backend_name}): Cannot open")
            
            cap.release()
        
        if not camera_working:
            print(f"  ❌ Camera {camera_index}: No working backend found")
            cameras_found.append({
                'index': camera_index,
                'working': False
            })
    
    return cameras_found

def test_camera_performance(camera_index):
    """Test performance of a specific camera."""
    print(f"\n🚀 Testing performance of camera {camera_index}...")
    
    # Try different backends for Logitech cameras
    backends_to_try = [
        (cv2.CAP_V4L2, "V4L2 (Linux)"),
        (cv2.CAP_ANY, "Any backend"),
        (cv2.CAP_FFMPEG, "FFMPEG")
    ]
    
    cap = None
    working_backend = None
    
    for backend, backend_name in backends_to_try:
        print(f"  🔧 Trying {backend_name}...")
        cap = cv2.VideoCapture(camera_index, backend)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"    ✅ {backend_name} works!")
                working_backend = backend_name
                break
            else:
                cap.release()
                cap = None
        else:
            cap.release()
            cap = None
    
    if cap is None or not cap.isOpened():
        print(f"❌ Camera {camera_index} not available with any backend")
        return None
    
    # Get default properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Default: {width}x{height} @ {fps}fps")
    
    # Try different resolutions
    resolutions = [
        (640, 480),
        (1280, 720),
        (1920, 1080)
    ]
    
    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  {w}x{h} -> {actual_w}x{actual_h}")
    
    # Test MJPEG at different resolutions
    resolutions_to_test = [
        (640, 480, "640x480"),
        (1280, 720, "1280x720")
    ]
    
    best_fps = 0
    best_resolution = None
    best_config = None
    
    for w, h, name in resolutions_to_test:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"Testing MJPEG at {name}...")
        
        # Quick performance test - test for 3 seconds
        start_time = time.time()
        frame_count = 0
        test_duration = 3.0  # Test for 3 seconds
        
        print(f"  Testing for {test_duration} seconds...")
        
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
        
        elapsed = time.time() - start_time
        fps_actual = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"  Captured {frame_count} frames in {elapsed:.2f}s = {fps_actual:.1f} FPS")
        
        if fps_actual > best_fps:
            best_fps = fps_actual
            best_resolution = name
            best_config = (w, h)
    
    cap.release()
    
    print(f"Best performance: {best_fps:.1f} FPS at {best_resolution}")
    
    return {
        'camera_index': camera_index,
        'backend': working_backend,
        'best_fps': best_fps,
        'best_resolution': best_resolution,
        'best_config': best_config,
        'performance_rating': 'good' if best_fps > 20 else 'poor'
    }

def test_logitech_specific_settings(camera_index):
    """Test Logitech camera with specific settings that often work."""
    print(f"\n🎯 Testing Logitech-specific settings for camera {camera_index}...")
    
    # Logitech-specific settings that often work
    logitech_configs = [
        {
            'name': 'Standard Logitech 1080p',
            'width': 1920, 'height': 1080,
            'fps': 30,
            'fourcc': cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            'buffer_size': 1
        },
        {
            'name': 'Logitech 720p MJPEG',
            'width': 1280, 'height': 720,
            'fps': 30,
            'fourcc': cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            'buffer_size': 1
        },
        {
            'name': 'Logitech 480p MJPEG',
            'width': 640, 'height': 480,
            'fps': 30,
            'fourcc': cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            'buffer_size': 1
        },
        {
            'name': 'Logitech YUYV',
            'width': 640, 'height': 480,
            'fps': 30,
            'fourcc': cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'),
            'buffer_size': 1
        }
    ]
    
    best_config = None
    best_fps = 0
    
    for config in logitech_configs:
        print(f"  🔧 Testing {config['name']}...")
        
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        
        if cap.isOpened():
            # Apply Logitech-specific settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
            cap.set(cv2.CAP_PROP_FPS, config['fps'])
            cap.set(cv2.CAP_PROP_FOURCC, config['fourcc'])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, config['buffer_size'])
            
            # Test for 2 seconds
            start_time = time.time()
            frame_count = 0
            test_duration = 2.0
            
            while time.time() - start_time < test_duration:
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
            
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            
            print(f"    📊 {config['name']}: {fps_actual:.1f} FPS")
            
            if fps_actual > best_fps:
                best_fps = fps_actual
                best_config = config
        
        cap.release()
    
    if best_config:
        print(f"  🏆 Best Logitech config: {best_config['name']} ({best_fps:.1f} FPS)")
        return best_config, best_fps
    else:
        print(f"  ❌ No working Logitech configuration found")
        return None, 0

def simple_camera_test():
    """Ultra-simple camera test."""
    print("🎥 Starting comprehensive camera test...")
    
    # First, find all cameras
    cameras = find_all_cameras()
    
    print(f"\n📊 Found {len([c for c in cameras if c.get('working', False)])} working cameras")
    
    # Test performance of each working camera
    performance_results = []
    
    for camera in cameras:
        if camera.get('working', False):
            result = test_camera_performance(camera['index'])
            if result:
                performance_results.append(result)
                
                # Special Logitech testing
                print(f"\n🔍 Testing Logitech-specific settings for camera {camera['index']}...")
                logitech_config, logitech_fps = test_logitech_specific_settings(camera['index'])
                if logitech_config and logitech_fps > 0:
                    result['logitech_config'] = logitech_config
                    result['logitech_fps'] = logitech_fps
    
    # Summary
    print(f"\n📈 PERFORMANCE SUMMARY:")
    print("=" * 50)
    
    if performance_results:
        # Sort by performance
        performance_results.sort(key=lambda x: x['best_fps'], reverse=True)
        
        for i, result in enumerate(performance_results, 1):
            rating = "🟢 EXCELLENT" if result['best_fps'] > 30 else "🟡 GOOD" if result['best_fps'] > 20 else "🔴 POOR"
            logitech_info = ""
            if 'logitech_config' in result:
                logitech_info = f" | Logitech: {result['logitech_fps']:.1f} FPS ({result['logitech_config']['name']})"
            print(f"{i}. Camera {result['camera_index']}: {result['best_fps']:.1f} FPS at {result['best_resolution']} - {rating}{logitech_info}")
        
        # Recommend best camera
        best_camera = performance_results[0]
        print(f"\n🏆 RECOMMENDED: Camera {best_camera['camera_index']} ({best_camera['best_fps']:.1f} FPS)")
        
        # Check if any camera has good performance
        good_cameras = [r for r in performance_results if r['best_fps'] > 20]
        if good_cameras:
            print(f"✅ {len(good_cameras)} camera(s) have good performance")
        else:
            print("⚠️  No cameras have good performance - check camera drivers/settings")
    else:
        print("❌ No working cameras found!")
    
    print(f"\n🎯 Test completed!")
    
    # Special Logitech camera diagnostics
    print(f"\n🔧 LOGITECH CAMERA DIAGNOSTICS:")
    print("=" * 50)
    
    # Check if v4l2-ctl is available
    try:
        import subprocess
        result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("📋 Available video devices:")
            print(result.stdout)
        else:
            print("❌ v4l2-ctl not available - install with: sudo apt install v4l-utils")
    except:
        print("❌ v4l2-ctl not available - install with: sudo apt install v4l-utils")
    
    # Check for Logitech-specific issues
    print(f"\n🔍 Checking for common Logitech issues...")
    
    # Check if camera is being used by another process
    try:
        result = subprocess.run(['lsof', '/dev/video*'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            print("⚠️  Camera devices in use by other processes:")
            print(result.stdout)
        else:
            print("✅ No processes using camera devices")
    except:
        print("ℹ️  lsof not available - cannot check for device conflicts")
    
    # Check USB device info
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"\n📱 USB devices:")
            for line in result.stdout.split('\n'):
                if 'logitech' in line.lower() or 'camera' in line.lower():
                    print(f"  📹 {line}")
    except:
        print("ℹ️  lsusb not available")
    
    # Recommendations for Logitech cameras
    print(f"\n💡 LOGITECH CAMERA TROUBLESHOOTING:")
    print("1. Try different camera indices (0, 1, 2, etc.)")
    print("2. Check if camera is used by other applications")
    print("3. Try unplugging and reconnecting the camera")
    print("4. Check if camera works with other software (cheese, guvcview)")
    print("5. For Logitech 1080HD, try: sudo modprobe uvcvideo")
    print("6. Check dmesg for USB errors: dmesg | grep -i usb")

if __name__ == "__main__":
    simple_camera_test()
