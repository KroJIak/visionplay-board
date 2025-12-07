import cv2
import numpy as np
import time
import random
import logging
from typing import List, Tuple, Dict, Any, Optional
from ..utils.config import Config
import mediapipe as mp

logger = logging.getLogger(__name__)

class HideAndSeekGame:
    """Hide and Seek game where players must avoid red zones."""
    
    def __init__(self):
        self.config = Config()
        self.is_running = False
        self.people_count = 0
        self.max_people = 3
        
        # Face validation using MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for 2m range, 1 for 5m range
            min_detection_confidence=0.5
        )
        
        # Game state
        self.game_state = "waiting"  # waiting, blinking, locked, caught, pause
        self.state_start_time = 0
        self.state_duration = 0
        
        # Red zone properties
        self.red_zone = None  # (x, y, width, height)
        self.zone_size_factor = 0.33  # Max 1/3 of screen
        self.blink_duration = 1.0  # seconds
        self.lock_duration = 3.0  # seconds
        self.caught_duration = 3.0  # seconds
        self.pause_duration = 3.0  # seconds
        
        # Caught players
        self.caught_players = []  # List of face images
        self.caught_faces = []  # List of face contours for collision detection
        self.caught_face_images = []  # Fullscreen face images
        
        # Rendering control
        self.suppress_overlay = False  # When True, hide overlay layers (e.g., during caught screen)
        
        # Visual properties
        self.blink_frequency = 10  # Hz
        self.last_blink_time = 0
        self.blink_state = True
        
    def start(self):
        """Start the game."""
        logger.info("Starting Hide and Seek game")
        self.is_running = True
        self.game_state = "waiting"
        self.state_start_time = time.time()
        self.state_duration = 2.0  # 2 seconds to prepare
        self.caught_players = []
        self.caught_faces = []
        
    def stop(self):
        """Stop the game."""
        logger.info("Stopping Hide and Seek game")
        self.is_running = False
        self.game_state = "waiting"
        
    def update(self, frame: np.ndarray, detection_data: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
        """
        Update game state and return modified frame.
        
        Args:
            frame: Current camera frame
            detection_data: Detection results from YoloHolisticDetector
            
        Returns:
            Tuple of (modified_frame, should_exit)
        """
        if not self.is_running:
            return frame, True
            
        current_time = time.time()
        elapsed = current_time - self.state_start_time
        
        # Update game state
        self._update_game_state(current_time, detection_data, frame)
        
        # Draw game elements
        frame = self._draw_game_elements(frame, current_time)
        
        # Check if should exit (no people detected for too long)
        if self.people_count == 0 and elapsed > 10.0:
            return frame, True
            
        return frame, False
    
    def _update_game_state(self, current_time: float, detection_data: Dict[str, Any], frame: np.ndarray):
        """Update the current game state."""
        elapsed = current_time - self.state_start_time
        
        if self.game_state == "waiting":
            if elapsed >= self.state_duration:
                self._start_new_round()
                
        elif self.game_state == "blinking":
            if elapsed >= self.blink_duration:
                self.game_state = "locked"
                self.state_start_time = current_time
                self.state_duration = self.lock_duration
                logger.info("Red zone locked!")
                
        elif self.game_state == "locked":
            # Check for collisions during lock period
            self._check_collisions(detection_data, frame)
            if elapsed >= self.lock_duration:
                self._end_round()
                
        elif self.game_state == "caught":
            if elapsed >= self.caught_duration:
                self._start_pause()
                
        elif self.game_state == "pause":
            if elapsed >= self.pause_duration:
                self._start_new_round()
    
    def _start_new_round(self):
        """Start a new round with a random red zone."""
        # Generate random red zone
        self.red_zone = self._generate_random_zone()
        self.game_state = "blinking"
        self.state_start_time = time.time()
        self.state_duration = self.blink_duration
        self.caught_players = []
        self.caught_faces = []
        self.caught_face_images = []
        self.last_blink_time = 0
        self.blink_state = True
        self.suppress_overlay = False
        
        logger.info(f"New round started! Red zone: {self.red_zone}")
    
    def _generate_random_zone(self) -> Tuple[int, int, int, int]:
        """Generate a random red zone position and size."""
        frame_height, frame_width = self.config.WINDOW_HEIGHT, self.config.WINDOW_WIDTH
        
        # Random size (up to 2/3 of screen - increased from 1/3)
        max_width = int(frame_width * self.zone_size_factor * 2)
        max_height = int(frame_height * self.zone_size_factor * 2)
        
        width = random.randint(max_width // 2, max_width)
        height = random.randint(max_height // 2, max_height)
        
        # Random position
        x = random.randint(0, frame_width - width)
        y = random.randint(0, frame_height - height)
        
        return (x, y, width, height)
    
    def _check_collisions(self, detection_data: Dict[str, Any], frame: np.ndarray):
        """Check if any player landmarks intersect with the red zone."""
        if not self.red_zone:
            return
            
        # Get pose landmarks from detection data
        pose_lines = detection_data.get('pose_lines', [])
        
        # Check each person's landmarks
        for person_idx, lines in enumerate(pose_lines):
            if person_idx in self.caught_players:
                continue  # Already caught
                
            # Check if any landmark is inside the red zone
            for x1, y1, x2, y2 in lines:
                if self._point_in_zone(x1, y1) or self._point_in_zone(x2, y2):
                    self._catch_player(person_idx, detection_data, frame)
                    break
    
    def _point_in_zone(self, x: int, y: int) -> bool:
        """Check if a point is inside the red zone."""
        if not self.red_zone:
            return False
            
        zone_x, zone_y, zone_w, zone_h = self.red_zone
        return zone_x <= x <= zone_x + zone_w and zone_y <= y <= zone_y + zone_h
    
    def _catch_player(self, player_idx: int, detection_data: Dict[str, Any], frame: np.ndarray):
        """Catch a player and store their face image (square crop)."""
        self.caught_players.append(player_idx)
        
        # Store face image based on face contour
        face_contours = detection_data.get('face_contours', [])
        if player_idx < len(face_contours):
            contour = face_contours[player_idx]
            if contour:
                try:
                    pts = np.array(contour, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(pts)
                    # Add padding (increased by 20% more)
                    pad = int(0.312 * max(w, h))  # 0.26 * 1.2 = 0.312
                    cx = x + w // 2
                    cy = y + h // 2
                    side = max(w, h) + 2 * pad
                    # Make square around center
                    sx = cx - side // 2
                    sy = cy - side // 2
                    # Clamp to frame bounds
                    fh, fw = frame.shape[:2]
                    sx = max(0, min(sx, fw - 1))
                    sy = max(0, min(sy, fh - 1))
                    ex = max(1, min(sx + side, fw))
                    ey = max(1, min(sy + side, fh))
                    # Ensure positive size
                    if ex > sx and ey > sy:
                        face_crop = frame[sy:ey, sx:ex].copy()
                        # Resize to window size (stretch)
                        resized = cv2.resize(face_crop, (fw, fh), interpolation=cv2.INTER_LINEAR)
                        # Keep latest face image for fullscreen display
                        if not hasattr(self, 'caught_face_images'):
                            self.caught_face_images = []
                        
                        # Validate face image before storing
                        if self._validate_face_image(resized):
                            self.caught_face_images.append(resized)
                            logger.info(f"Player {player_idx} caught with valid face!")
                        else:
                            logger.warning(f"Player {player_idx} caught but no face detected in image - continuing search")
                            # Remove from caught players if no valid face
                            if player_idx in self.caught_players:
                                self.caught_players.remove(player_idx)
                except Exception as e:
                    logger.error(f"Failed to crop face image: {e}")
            
    def _validate_face_image(self, face_image: np.ndarray) -> bool:
        """Validate that the face image actually contains a face using MediaPipe."""
        if face_image is None or face_image.size == 0:
            return False
            
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(rgb_image)
            
            # Check if any faces were detected
            return results.detections is not None and len(results.detections) > 0
        except Exception as e:
            logger.error(f"Face validation error: {e}")
            return False
    
    def _end_round(self):
        """End the current round."""
        if self.caught_players:
            self.game_state = "caught"
            self.state_start_time = time.time()
            self.state_duration = self.caught_duration
            self.suppress_overlay = True
            logger.info(f"Round ended! Caught {len(self.caught_players)} players")
        else:
            self._start_pause()
    
    def _start_pause(self):
        """Start pause period."""
        self.game_state = "pause"
        self.state_start_time = time.time()
        self.state_duration = self.pause_duration
        self.suppress_overlay = False
        logger.info("Pause period started")
    
    def _draw_game_elements(self, frame: np.ndarray, current_time: float) -> np.ndarray:
        """Draw all game elements on the frame."""
        if self.game_state == "waiting":
            return self._draw_waiting_screen(frame)
        elif self.game_state == "blinking":
            return self._draw_blinking_zone(frame, current_time)
        elif self.game_state == "locked":
            return self._draw_locked_zone(frame)
        elif self.game_state == "caught":
            return self._draw_caught_screen(frame)
        elif self.game_state == "pause":
            return self._draw_pause_screen(frame)
        
        return frame
    
    def _draw_waiting_screen(self, frame: np.ndarray) -> np.ndarray:
        """Draw waiting screen."""
        height, width = frame.shape[:2]
        
        # Draw title
        cv2.putText(frame, "HIDE AND SEEK", 
                   (width // 2 - 150, height // 2 - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        cv2.putText(frame, "Get ready!", 
                   (width // 2 - 80, height // 2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def _draw_blinking_zone(self, frame: np.ndarray, current_time: float) -> np.ndarray:
        """Draw blinking red zone."""
        if not self.red_zone:
            return frame
            
        # Blink animation
        if current_time - self.last_blink_time > 1.0 / self.blink_frequency:
            self.blink_state = not self.blink_state
            self.last_blink_time = current_time
        
        if self.blink_state:
            frame = self._draw_red_zone(frame, alpha=0.5)
        
        # Draw warning text
        zone_x, zone_y, zone_w, zone_h = self.red_zone
        cv2.putText(frame, "ESCAPE!", 
                   (zone_x + zone_w // 2 - 50, zone_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame
    
    def _draw_locked_zone(self, frame: np.ndarray) -> np.ndarray:
        """Draw locked red zone."""
        if not self.red_zone:
            return frame
            
        frame = self._draw_red_zone(frame, alpha=0.7)
        
        # Draw "LOCKED" text
        zone_x, zone_y, zone_w, zone_h = self.red_zone
        cv2.putText(frame, "LOCKED!", 
                   (zone_x + zone_w // 2 - 60, zone_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame
    
    def _draw_red_zone(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Draw the red zone with transparency."""
        if not self.red_zone:
            return frame
            
        zone_x, zone_y, zone_w, zone_h = self.red_zone
        
        # Create overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (zone_x, zone_y), (zone_x + zone_w, zone_y + zone_h), (0, 0, 255), -1)
        
        # Blend with original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (zone_x, zone_y), (zone_x + zone_w, zone_y + zone_h), (0, 0, 255), 3)
        
        return frame
    
    def _draw_caught_screen(self, frame: np.ndarray) -> np.ndarray:
        """Draw fullscreen caught face for 3 seconds."""
        height, width = frame.shape[:2]
        if hasattr(self, 'caught_face_images') and self.caught_face_images:
            # Show the last caught face fullscreen (already resized to window size)
            last_face = self.caught_face_images[-1]
            # Safety resize in case window changed
            if last_face.shape[1] != width or last_face.shape[0] != height:
                last_face = cv2.resize(last_face, (width, height), interpolation=cv2.INTER_LINEAR)
            frame[:] = last_face
            # Optional overlay label
            cv2.putText(frame, "CAUGHT", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        return frame
    
    def _draw_pause_screen(self, frame: np.ndarray) -> np.ndarray:
        """Draw pause screen."""
        height, width = frame.shape[:2]
        
        cv2.putText(frame, "PREPARING NEXT ROUND...", 
                   (width // 2 - 200, height // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        return frame
