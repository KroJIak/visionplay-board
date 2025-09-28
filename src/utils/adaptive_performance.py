"""
Adaptive performance manager for VisionPlay Board.
Adjusts pose detection frequency and caching based on movement speed.
"""

import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque

class AdaptivePerformanceManager:
    """Manages adaptive performance based on movement speed."""
    
    def __init__(self, config):
        """Initialize adaptive performance manager."""
        self.config = config
        self.enabled = config.ENABLE_ADAPTIVE_PERFORMANCE
        
        # Movement tracking
        self.last_landmarks = None
        self.last_timestamp = 0
        self.movement_history = deque(maxlen=10)  # Track last 10 movements
        
        # Current adaptive settings
        self.current_skip_frames = config.POSE_DETECTION_SKIP_FRAMES
        self.current_cache_time = config.POSE_RESULT_CACHE_TIME
        
        # Smoothing
        self.smoothing_factor = config.ADAPTATION_SMOOTHING
        
        # Performance stats
        self.adaptation_count = 0
        self.last_adaptation_time = 0
        
    def update(self, pose_results: Optional[Dict[str, Any]]) -> Tuple[int, float]:
        """Update adaptive settings based on current pose results."""
        if not self.enabled or not pose_results:
            return self.current_skip_frames, self.current_cache_time
        
        current_time = time.time()
        
        # Calculate movement speed
        movement_speed = self._calculate_movement_speed(pose_results, current_time)
        
        if movement_speed is not None:
            # Update movement history
            self.movement_history.append(movement_speed)
            
            # Calculate average movement speed
            avg_speed = np.mean(list(self.movement_history)) if self.movement_history else 0
            
            # Adapt settings based on movement speed
            self._adapt_settings(avg_speed)
            
            # Update stats
            self.adaptation_count += 1
            self.last_adaptation_time = current_time
        
        return self.current_skip_frames, self.current_cache_time
    
    def _calculate_movement_speed(self, pose_results: Dict[str, Any], current_time: float) -> Optional[float]:
        """Calculate movement speed based on landmark positions."""
        if not pose_results or not pose_results.get('pose_landmarks'):
            return None
        
        landmarks = pose_results['pose_landmarks']
        
        # Get key body points for movement calculation
        key_points = self._extract_key_points(landmarks)
        if not key_points:
            return None
        
        # Calculate movement from last frame
        if self.last_landmarks is not None and self.last_timestamp > 0:
            time_delta = current_time - self.last_timestamp
            if time_delta > 0:
                # Calculate average movement distance
                total_movement = 0
                valid_points = 0
                
                for point_name, current_pos in key_points.items():
                    if point_name in self.last_landmarks:
                        last_pos = self.last_landmarks[point_name]
                        # Calculate Euclidean distance
                        distance = np.sqrt((current_pos[0] - last_pos[0])**2 + 
                                         (current_pos[1] - last_pos[1])**2)
                        total_movement += distance
                        valid_points += 1
                
                if valid_points > 0:
                    avg_movement = total_movement / valid_points
                    # Convert to pixels per frame (assuming ~30 FPS)
                    speed = avg_movement / time_delta
                    return speed
        
        # Store current landmarks for next frame
        self.last_landmarks = key_points.copy()
        self.last_timestamp = current_time
        
        return None
    
    def _extract_key_points(self, landmarks) -> Dict[str, Tuple[float, float]]:
        """Extract key body points for movement calculation."""
        key_points = {}
        
        # Handle both single pose and multiple poses
        if isinstance(landmarks, list):
            landmarks = landmarks[0] if landmarks else None
        
        if landmarks and hasattr(landmarks, 'landmark'):
            # Define key landmark indices for movement tracking
            landmark_indices = {
                'nose': 0,
                'left_shoulder': 11, 'right_shoulder': 12,
                'left_hip': 23, 'right_hip': 24,
                'left_wrist': 15, 'right_wrist': 16,
                'left_ankle': 27, 'right_ankle': 28
            }
            
            for name, idx in landmark_indices.items():
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    if landmark.visibility > self.config.LANDMARK_VISIBILITY_THRESHOLD:
                        key_points[name] = (landmark.x, landmark.y)
        
        return key_points
    
    def _adapt_settings(self, avg_speed: float):
        """Adapt settings based on average movement speed."""
        # Normalize speed (0-1 scale)
        # Speed above threshold = fast movement, below = slow movement
        speed_ratio = min(avg_speed / self.config.MOVEMENT_THRESHOLD, 1.0)
        
        # Calculate new settings
        # Fast movement: more frequent detection, shorter cache
        # Slow movement: less frequent detection, longer cache
        new_skip_frames = int(self.config.MIN_SKIP_FRAMES + 
                             (1.0 - speed_ratio) * (self.config.MAX_SKIP_FRAMES - self.config.MIN_SKIP_FRAMES))
        
        new_cache_time = self.config.MIN_CACHE_TIME + \
                        (1.0 - speed_ratio) * (self.config.MAX_CACHE_TIME - self.config.MIN_CACHE_TIME)
        
        # Apply smoothing to prevent rapid changes
        self.current_skip_frames = int(
            self.current_skip_frames * (1 - self.smoothing_factor) + 
            new_skip_frames * self.smoothing_factor
        )
        
        self.current_cache_time = (
            self.current_cache_time * (1 - self.smoothing_factor) + 
            new_cache_time * self.smoothing_factor
        )
        
        # Ensure values are within bounds
        self.current_skip_frames = max(self.config.MIN_SKIP_FRAMES, 
                                     min(self.current_skip_frames, self.config.MAX_SKIP_FRAMES))
        self.current_cache_time = max(self.config.MIN_CACHE_TIME, 
                                    min(self.current_cache_time, self.config.MAX_CACHE_TIME))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adaptive performance statistics."""
        return {
            'enabled': self.enabled,
            'current_skip_frames': self.current_skip_frames,
            'current_cache_time': self.current_cache_time,
            'adaptation_count': self.adaptation_count,
            'last_adaptation_time': self.last_adaptation_time,
            'movement_history_length': len(self.movement_history),
            'avg_movement_speed': np.mean(list(self.movement_history)) if self.movement_history else 0
        }
    
    def reset(self):
        """Reset adaptive settings to defaults."""
        self.current_skip_frames = self.config.POSE_DETECTION_SKIP_FRAMES
        self.current_cache_time = self.config.POSE_RESULT_CACHE_TIME
        self.movement_history.clear()
        self.last_landmarks = None
        self.last_timestamp = 0
        self.adaptation_count = 0
