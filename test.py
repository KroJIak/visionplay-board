import cv2
import time
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def test_camera_performance():
    """Test camera performance with optimized settings."""
    
    # Get camera settings from config
    camera_index = int(os.getenv('CAMERA_INDEX', '0'))
    camera_width = int(os.getenv('CAMERA_WIDTH', '1920'))
    camera_height = int(os.getenv('CAMERA_HEIGHT', '1080'))
    
    print(f"Testing camera {camera_index} with resolution {camera_width}x{camera_height}")
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    # Apply camera settings (same as in main app)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Enable MJPEG mode for better performance
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    # Additional performance optimizations
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
    
    # Get actual camera properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    print(f"Actual resolution: {actual_width}x{actual_height}")
    print(f"Actual FPS: {actual_fps}")
    print(f"Codec: {fourcc}")
    
    # Performance monitoring
    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    current_fps = 0
    
    # Create window
    window_name = "Camera Test - Press ESC to exit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Try to set fullscreen
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("Fullscreen mode enabled")
    except:
        print("Fullscreen mode not available")
    
    print("Starting camera test... Press ESC to exit")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Update FPS counter
            frame_count += 1
            current_time = time.time()
            
            if current_time - last_fps_time >= 1.0:
                current_fps = frame_count
                frame_count = 0
                last_fps_time = current_time
                print(f"FPS: {current_fps}")
            
            # Add FPS text to frame
            cv2.putText(frame, f"FPS: {current_fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add resolution info
            cv2.putText(frame, f"Resolution: {actual_width}x{actual_height}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add codec info
            cv2.putText(frame, f"Codec: MJPEG", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('q'):
                break
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\nTest completed!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    test_camera_performance()