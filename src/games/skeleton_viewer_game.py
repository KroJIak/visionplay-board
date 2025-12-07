"""
Game 1: Skeleton Viewer
Displays camera feed without blur and shows skeletons for up to 3 people.
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Optional
from .base_game import BaseGame

class SkeletonViewerGame(BaseGame):
    """Game that displays camera feed with skeleton visualization for up to 3 people."""
    
    def __init__(self, config, scaler=None, pose_detector=None):
        """Initialize skeleton viewer game."""
        super().__init__("Skeleton Viewer", config)
        self.max_people = 3
        self.people_count = 0
        self.last_update_time = 0
        self.scaler = scaler  # Store scaler for coordinate transformation
        self.pose_detector = pose_detector  # Store pose detector for drawing
        
        # Auto-exit timer when no people detected
        self.no_people_start_time = None
        self.no_people_timeout = 10.0  # 10 seconds without people
        self.exit_countdown_duration = 3.0  # 3 seconds countdown
        
        # Colors for different people
        self.people_colors = [
            (0, 255, 0),    # Green for person 1
            (255, 0, 0),    # Red for person 2  
            (0, 0, 255),    # Blue for person 3
        ]
    
    def start(self):
        """Start the skeleton viewer game."""
        self.is_active = True
        self.start_time = time.time()
        self.score = 0
        self.people_count = 0
        # Reset auto-exit timers
        self.no_people_start_time = None
        print(f"Started {self.name}")
    
    def stop(self):
        """Stop the skeleton viewer game."""
        self.is_active = False
        print(f"Stopped {self.name}. Score: {self.score}")
    
    def update(self, frame: np.ndarray, pose_results: Optional[Dict[str, Any]]) -> tuple:
        """Update game state and return modified frame and exit flag."""
        if not self.is_active:
            return frame, False
        
        # Create a copy of the frame
        game_frame = frame.copy()
        
        # Count detected people (set externally from YOLO bboxes in app)
        # Fallback to legacy counting if pose_results provided
        if pose_results is not None:
            self.people_count = self._count_detected_people(pose_results)
        
        # Check for auto-exit due to no people
        should_exit = self._check_auto_exit(pose_results)
        
        # Draw skeletons for detected people using pose detector
        if pose_results and hasattr(self, 'pose_detector') and self.pose_detector:
            game_frame = self.pose_detector.draw_comprehensive_pose(game_frame, pose_results, self.scaler)
        
        # Draw UI overlay
        game_frame = self._draw_ui_overlay(game_frame)
        
        return game_frame, should_exit
    
    def _count_detected_people(self, pose_results: Optional[Dict[str, Any]]) -> int:
        """Count the number of detected people."""
        if not pose_results:
            return 0
        
        # Count based on pose landmarks (body detection) - main indicator
        # NOTE: MediaPipe Pose can only detect 1 person at a time
        pose_count = 0
        if pose_results.get('pose_landmarks'):
            if isinstance(pose_results['pose_landmarks'], list):
                pose_count = len(pose_results['pose_landmarks'])
            else:
                pose_count = 1
        
        # Count faces (can be up to 3)
        face_count = 0
        if pose_results.get('face_landmarks'):
            if isinstance(pose_results['face_landmarks'], list):
                face_count = len(pose_results['face_landmarks'])
            else:
                face_count = 1
        
        # Count hands (divide by 2 since we have left and right hands)
        left_hand_count = 0
        right_hand_count = 0
        if pose_results.get('left_hand_landmarks'):
            if isinstance(pose_results['left_hand_landmarks'], list):
                left_hand_count = len(pose_results['left_hand_landmarks'])
            else:
                left_hand_count = 1
        if pose_results.get('right_hand_landmarks'):
            if isinstance(pose_results['right_hand_landmarks'], list):
                right_hand_count = len(pose_results['right_hand_landmarks'])
            else:
                right_hand_count = 1
        
        # Estimate people from hands (rough estimate)
        total_hands = left_hand_count + right_hand_count
        estimated_people_from_hands = (total_hands + 1) // 2
        
        # Take the maximum count from all indicators, but limit to 3
        total_count = max(pose_count, face_count, estimated_people_from_hands)
        return min(total_count, self.max_people)
    
    
    def _check_auto_exit(self, pose_results: Optional[Dict[str, Any]]) -> bool:
        """Check if game should auto-exit due to no people detected."""
        current_time = time.time()
        
        # If people are detected, reset all timers
        if self.people_count > 0:
            self.no_people_start_time = None
            return False
        
        # No people detected - start or continue timer
        if self.no_people_start_time is None:
            self.no_people_start_time = current_time
            return False
        
        # Calculate total time without people
        no_people_duration = current_time - self.no_people_start_time
        total_timeout = self.no_people_timeout + self.exit_countdown_duration
        
        # Check if total timeout reached (10s + 3s = 13s)
        if no_people_duration >= total_timeout:
            print("Auto-exit timeout reached. Returning to menu.")
            return True
        
        return False
    
    
    def _draw_ui_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI overlay with game information."""
        height, width = frame.shape[:2]
        
        # Game title
        cv2.putText(frame, "SKELETON VIEWER", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # People count
        cv2.putText(frame, f"People detected: {self.people_count}/{self.max_people}", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Auto-exit warnings - only show in last 3 seconds
        if self.no_people_start_time is not None:
            current_time = time.time()
            no_people_duration = current_time - self.no_people_start_time
            total_timeout = self.no_people_timeout + self.exit_countdown_duration
            remaining_time = total_timeout - no_people_duration
            
            # Only show warning in the last 3 seconds (when remaining_time <= 3.0)
            if 0 < remaining_time <= 3.0:
                if remaining_time > self.exit_countdown_duration:
                    # Still in warning phase
                    warning_text = f"No people detected! Game will exit in {remaining_time:.1f}s"
                    cv2.putText(frame, warning_text, (50, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    # In countdown phase
                    countdown_text = f"Returning to menu in {remaining_time:.1f}s"
                    cv2.putText(frame, countdown_text, (50, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
        
        # Color legend
        legend_y = height - 150
        cv2.putText(frame, "Color Legend:", (50, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        for i, color in enumerate(self.people_colors):
            legend_text = f"Person {i+1}"
            legend_x = 50 + i * 200
            legend_y += 30
            
            # Draw color circle
            cv2.circle(frame, (legend_x - 30, legend_y - 5), 8, color, -1)
            cv2.circle(frame, (legend_x - 30, legend_y - 5), 10, (255, 255, 255), 2)
            
            # Draw text
            cv2.putText(frame, legend_text, (legend_x, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press ESC to return to menu", (width - 400, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
