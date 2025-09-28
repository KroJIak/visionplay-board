#!/usr/bin/env python3
"""
Minimal camera test with lowest possible settings.
"""

import cv2
import time

def minimal_camera_test():
    """Minimal camera test with lowest settings."""
    print("Starting minimal camera test...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    # Minimal settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Try MJPEG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    print("Camera settings applied")
    
    # Create window
    cv2.namedWindow("Minimal Test", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    current_fps = 0
    
    print("Starting capture loop... Press ESC to exit")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read frame")
                break
            
            # Update FPS
            frame_count += 1
            current_time = time.time()
            
            if current_time - last_fps_time >= 1.0:
                current_fps = frame_count
                frame_count = 0
                last_fps_time = current_time
                print(f"FPS: {current_fps}")
            
            # Add FPS to frame
            cv2.putText(frame, f"FPS: {current_fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Minimal Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Test completed")

if __name__ == "__main__":
    minimal_camera_test()
