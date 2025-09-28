"""
Simplified thread manager for VisionPlay Board.
Only pose detection runs in separate thread, camera runs in main thread.
"""

import threading
import time
import queue
import cv2
import numpy as np
from typing import Optional, Dict, Any

class PoseDetectionThread:
    """Thread for pose detection only."""
    
    def __init__(self, pose_detector, frame_queue: queue.Queue, 
                 result_queue: queue.Queue):
        """Initialize pose detection thread."""
        self.pose_detector = pose_detector
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        
        self.running = False
        self.thread = None
        self.processing_time = 0
        self.frame_count = 0
        
    def start(self):
        """Start pose detection thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print("Pose detection thread started")
    
    def stop(self):
        """Stop pose detection thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("Pose detection thread stopped")
    
    def _detection_loop(self):
        """Main detection loop."""
        while self.running:
            try:
                # Get frame from queue with shorter timeout for more responsive processing
                frame = self.frame_queue.get(timeout=0.05)
                
                # Process frame (skip pose detection if disabled)
                start_time = time.time()
                pose_results = None
                if self.pose_detector and hasattr(self.pose_detector, 'detect_all'):
                    pose_results = self.pose_detector.detect_all(frame)
                processing_time = time.time() - start_time
                
                # Update statistics
                self.processing_time = processing_time
                self.frame_count += 1
                
                # Always put result in queue, replacing oldest if needed
                result = {
                    'frame': frame,
                    'pose_results': pose_results,
                    'timestamp': time.time(),
                    'processing_time': processing_time
                }
                
                # Clear queue and put new result to ensure we always have the latest
                while not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        break
                
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    pass  # This shouldn't happen since we cleared the queue
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in pose detection thread: {e}")
                continue
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread statistics."""
        return {
            'processing_time': self.processing_time,
            'frame_count': self.frame_count,
            'running': self.running
        }

class ThreadManager:
    """Simplified thread manager - only manages pose detection thread."""
    
    def __init__(self, config):
        """Initialize thread manager."""
        self.config = config
        self.pose_thread = None
        
        # Queues for communication between threads
        self.frame_queue = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE)
        self.result_queue = queue.Queue(maxsize=config.RESULT_QUEUE_SIZE)
        
        # Buffer for last pose result to prevent flickering
        self.last_pose_result = None
        self.last_pose_timestamp = 0
        self.pose_result_lock = threading.Lock()
        
        # Statistics
        self.start_time = None
        self.total_frames_processed = 0
        
    def start(self, pose_detector):
        """Start pose detection thread."""
        if self.pose_thread:
            print("Pose thread already running")
            return
        
        # Start pose detection thread
        self.pose_thread = PoseDetectionThread(
            pose_detector,
            self.frame_queue,
            self.result_queue
        )
        self.pose_thread.start()
        
        self.start_time = time.time()
        print("Pose detection thread started")
    
    def stop(self):
        """Stop all threads."""
        if self.pose_thread:
            self.pose_thread.stop()
            self.pose_thread = None
        
        print("All threads stopped")
    
    def get_latest_result(self, adaptive_cache_time: float = None) -> Optional[Dict[str, Any]]:
        """Get latest result from pose detection thread."""
        # Try to get new result first
        try:
            new_result = self.result_queue.get_nowait()
            # Update last result buffer
            with self.pose_result_lock:
                self.last_pose_result = new_result.get('pose_results')
                self.last_pose_timestamp = new_result.get('timestamp', 0)
            return new_result
        except queue.Empty:
            # No new result, return last known result if it's recent enough
            with self.pose_result_lock:
                current_time = time.time()
                # Use adaptive cache time if provided, otherwise use config default
                cache_time = adaptive_cache_time if adaptive_cache_time is not None else self.config.POSE_RESULT_CACHE_TIME
                if (self.last_pose_result is not None and 
                    current_time - self.last_pose_timestamp < cache_time):
                    return {
                        'frame': None,  # We don't have the frame, but that's OK
                        'pose_results': self.last_pose_result,
                        'timestamp': self.last_pose_timestamp,
                        'processing_time': 0
                    }
            return None
    
    def put_frame_for_processing(self, frame: np.ndarray):
        """Put frame in queue for pose detection processing."""
        # Keep queue small to reduce latency
        max_size = self.config.FRAME_QUEUE_SIZE
        if self.frame_queue.qsize() >= max_size:
            try:
                self.frame_queue.get_nowait()  # Remove oldest frame
            except queue.Empty:
                pass
        
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # Skip frame if queue is full
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        stats = {
            'total_frames_processed': self.total_frames_processed,
            'uptime': time.time() - self.start_time if self.start_time else 0
        }
        
        if self.pose_thread:
            stats['pose_thread'] = self.pose_thread.get_stats()
        
        return stats