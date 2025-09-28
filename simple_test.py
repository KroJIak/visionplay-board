#!/usr/bin/env python3
"""
Simple camera test to diagnose performance issues.
"""

import cv2
import time

def simple_camera_test():
    """Ultra-simple camera test."""
    print("Starting simple camera test...")
    
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        print(f"\nTrying camera {camera_index}...")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Camera {camera_index} not available")
            continue
        
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
        
        cap.release()
        
        print(f"Best performance: {best_fps:.1f} FPS at {best_resolution}")
        
        if best_fps > 20:  # Good performance
            print(f"Camera {camera_index} has good performance!")
            break
        else:
            print(f"Camera {camera_index} performance is poor")
    
    print("\nSimple test completed!")

if __name__ == "__main__":
    simple_camera_test()
