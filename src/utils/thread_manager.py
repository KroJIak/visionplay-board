"""
Simplified thread manager for VisionPlay Board.
Runs person detection (HOG) in a separate thread; camera runs in main thread.
"""

import threading
import time
import queue
import cv2
import numpy as np
from typing import Optional, Dict, Any

class DetectionThread:
    """Thread for person detection only (produces bounding boxes)."""
    
    def __init__(self, detector, frame_queue: queue.Queue, 
                 result_queue: queue.Queue):
        """Initialize detection thread."""
        self.detector = detector
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        
        self.running = False
        self.thread = None
        self.processing_time = 0
        self.frame_count = 0
        
    def start(self):
        """Start detection thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print("Detection thread started")
    
    def stop(self):
        """Stop detection thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("Detection thread stopped")
    
    def _detection_loop(self):
        """Main detection loop."""
        while self.running:
            try:
                # Get frame from queue with shorter timeout for more responsive processing
                frame = self.frame_queue.get(timeout=0.05)
                
                # Process frame
                start_time = time.time()
                bboxes = []
                face_contours = []
                pose_lines = []
                hand_lines = []
                if self.detector and hasattr(self.detector, 'detect'):
                    try:
                        det_out = self.detector.detect(frame)
                        if isinstance(det_out, dict):
                            bboxes = det_out.get('bboxes', [])
                            face_contours = det_out.get('face_contours', [])
                            pose_lines = det_out.get('pose_lines', [])
                            hand_lines = det_out.get('hand_lines', [])
                        else:
                            bboxes = det_out
                    except Exception as e:
                        print(f"[Debug] Detection error: {e}")
                        bboxes = []
                        face_contours = []
                        pose_lines = []
                        hand_lines = []
                processing_time = time.time() - start_time
                
                # Debug: log slow detection
                if processing_time > 0.1:  # More than 100ms
                    print(f"[Debug] Slow detection: {processing_time:.3f}s")
                
                # Update statistics
                self.processing_time = processing_time
                self.frame_count += 1
                
                # Always put result in queue, replacing oldest if needed
                result = {
                    'frame': frame,
                    'bboxes': bboxes,
                    'face_contours': face_contours,
                    'pose_lines': pose_lines,
                    'hand_lines': hand_lines,
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
                print(f"Error in detection thread: {e}")
                continue
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread statistics."""
        return {
            'processing_time': self.processing_time,
            'frame_count': self.frame_count,
            'running': self.running
        }

class ThreadManager:
    """Simplified thread manager - manages detection thread (HOG)."""
    
    def __init__(self, config):
        """Initialize thread manager."""
        self.config = config
        self.det_thread = None
        
        # Queues for communication between threads
        self.frame_queue = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE)
        self.result_queue = queue.Queue(maxsize=config.RESULT_QUEUE_SIZE)
        
        # Buffer for last bboxes result to prevent flickering
        self.last_bboxes = []
        self.last_face_contours = []
        self.last_pose_lines = []
        self.last_hand_lines = []
        self.last_result_timestamp = 0
        self.result_lock = threading.Lock()
        self.has_valid_result = False
        
        # Statistics
        self.start_time = None
        self.total_frames_processed = 0
        
    def start(self, detector):
        """Start detection thread."""
        if self.det_thread:
            print("Detection thread already running")
            return
        
        # Start detection thread
        self.det_thread = DetectionThread(
            detector,
            self.frame_queue,
            self.result_queue
        )
        self.det_thread.start()
        
        self.start_time = time.time()
        print("Detection thread started")
    
    def stop(self):
        """Stop all threads."""
        if self.det_thread:
            self.det_thread.stop()
            self.det_thread = None
        
        print("All threads stopped")
    
    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """Get latest result from detection thread."""
        # Try to get new result first
        try:
            new_result = self.result_queue.get_nowait()
            # Update last result buffer
            with self.result_lock:
                self.last_bboxes = new_result.get('bboxes', [])
                self.last_face_contours = new_result.get('face_contours', [])
                self.last_pose_lines = new_result.get('pose_lines', [])
                self.last_hand_lines = new_result.get('hand_lines', [])
                self.last_result_timestamp = new_result.get('timestamp', 0)
                self.has_valid_result = True
            return new_result
        except queue.Empty:
            # No new result, return last known result if we have one
            with self.result_lock:
                if self.has_valid_result:
                    return {
                        'frame': None,  # We don't have the frame, but that's OK
                        'bboxes': self.last_bboxes,
                        'face_contours': self.last_face_contours,
                        'pose_lines': self.last_pose_lines,
                        'hand_lines': self.last_hand_lines,
                        'timestamp': self.last_result_timestamp,
                        'processing_time': 0
                    }
            # Return empty result if no valid result yet
            return {
                'frame': None,
                'bboxes': [],
                'face_contours': [],
                'pose_lines': [],
                'hand_lines': [],
                'timestamp': 0,
                'processing_time': 0
            }
    
    def put_frame_for_processing(self, frame: np.ndarray):
        """Put frame in queue for detection processing."""
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
        
        if self.det_thread:
            stats['det_thread'] = self.det_thread.get_stats()
        
        return stats