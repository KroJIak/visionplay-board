"""
Comprehensive pose detection utilities using MediaPipe.
Supports body, face, and hand detection with detailed control.
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict, Any
from .config import Config

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensivePoseDetector:
    """Comprehensive MediaPipe detection wrapper for body, face, and hands."""
    
    def __init__(self, config: Config):
        """Initialize comprehensive pose detector with configuration."""
        self.config = config
        
        # Initialize MediaPipe solutions
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection (optimized for performance)
        # NOTE: MediaPipe Pose can only detect ONE person at a time
        # This is a limitation of MediaPipe Pose, not our code
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Reduced from 1 to 0 for better performance
            enable_segmentation=False,
            min_detection_confidence=config.POSE_CONFIDENCE_THRESHOLD,
            min_tracking_confidence=config.POSE_TRACKING_CONFIDENCE
        )
        
        # Initialize face mesh detection (more sensitive for face detection)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=config.MAX_PEOPLE_IN_FRAME,  # Use setting from .env
            refine_landmarks=True,
            min_detection_confidence=0.3,  # Lower threshold for face detection
            min_tracking_confidence=0.3
        )
        
        # Initialize hand detection (more sensitive for hand detection)
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_PEOPLE_IN_FRAME * 2,  # 2 hands per person
            model_complexity=0,  # Reduced from 1 to 0 for better performance
            min_detection_confidence=0.3,  # Lower threshold for hand detection
            min_tracking_confidence=0.3
        )
        
        # Optimization: Cache for face contour (per face)
        self._face_contour_caches = {}  # Dictionary to store caches for each face
        self._face_contour_frame_counts = {}  # Individual frame counters for each face
        self._face_contour_update_interval = 2  # Update every 2 frames instead of every frame
    
    def detect_all(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect pose, face, and hands in the given frame."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = {
            'pose_landmarks': None,
            'face_landmarks': None,
            'left_hand_landmarks': None,
            'right_hand_landmarks': None,
            'frame_shape': frame.shape
        }
        
        # Detect pose
        if self.config.SHOW_BODY_LANDMARKS or self.config.SHOW_POSE_CONNECTIONS:
            pose_results = self.pose.process(rgb_frame)
            if pose_results.pose_landmarks:
                # MediaPipe Pose can only detect one person at a time
                # Store as single pose for now
                results['pose_landmarks'] = [pose_results.pose_landmarks]
                results['num_poses'] = 1
                logger.debug(f"Detected 1 pose (MediaPipe Pose limitation)")
            else:
                # No pose detected
                results['pose_landmarks'] = []
                results['num_poses'] = 0
        
        # Detect face mesh
        if self.config.SHOW_FACE_LANDMARKS or self.config.SHOW_FACE_CONNECTIONS:
            face_results = self.face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                # Store all detected faces (up to 3)
                results['face_landmarks'] = face_results.multi_face_landmarks
                results['num_faces'] = len(face_results.multi_face_landmarks)
        
        # Detect hands
        if self.config.SHOW_HAND_LANDMARKS or self.config.SHOW_HAND_CONNECTIONS:
            hand_results = self.hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                # Store all hands as lists for multiple people support
                left_hands = []
                right_hands = []
                
                for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    if i < len(hand_results.multi_handedness):
                        handedness = hand_results.multi_handedness[i]
                        if handedness.classification[0].label == 'Left':
                            left_hands.append(hand_landmarks)
                        else:
                            right_hands.append(hand_landmarks)
                
                if left_hands:
                    results['left_hand_landmarks'] = left_hands
                if right_hands:
                    results['right_hand_landmarks'] = right_hands
        
        return results
    
    def is_person_detected(self, results: Dict[str, Any]) -> bool:
        """Check if a person is detected with sufficient confidence."""
        # Check pose landmarks
        if results.get('pose_landmarks'):
            pose_landmarks = results['pose_landmarks']
            # Handle both single pose and multiple poses
            if not isinstance(pose_landmarks, list):
                pose_landmarks = [pose_landmarks]
            
            # Check each pose
            for pose_landmark in pose_landmarks:
                key_landmarks = [
                    pose_landmark.landmark[self.mp_pose.PoseLandmark.NOSE],
                    pose_landmark.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                    pose_landmark.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                ]
                
                visible_count = sum(1 for lm in key_landmarks 
                                  if lm.visibility > self.config.LANDMARK_VISIBILITY_THRESHOLD)
                
                if visible_count >= 2:
                    return True
        
        # Check face landmarks as fallback
        if results.get('face_landmarks'):
            return True
        
        # Check hand landmarks as fallback
        if results.get('left_hand_landmarks') or results.get('right_hand_landmarks'):
            return True
        
        return False
    
    def draw_comprehensive_pose(self, frame: np.ndarray, results: Dict[str, Any], scaler=None) -> np.ndarray:
        """Draw all detected landmarks and connections with proper coordinate transformation."""
        if not results:
            return frame
        
        # If we have a scaler, we need to transform coordinates from original camera frame to scaled frame
        if scaler:
            # Create a temporary frame with original camera dimensions for MediaPipe drawing
            temp_frame = np.zeros((scaler.camera_height, scaler.camera_width, 3), dtype=np.uint8)
            
            # Draw on temporary frame (original camera dimensions)
            temp_frame = self._draw_landmarks_on_frame(temp_frame, results)
            
            # Now scale the temporary frame to match the target frame
            temp_scaled = scaler.scale_frame(temp_frame)
            
            # Blend the scaled landmarks onto the target frame
            frame = cv2.addWeighted(frame, 1.0, temp_scaled, 1.0, 0)
        else:
            # No scaler - draw directly on frame
            frame = self._draw_landmarks_on_frame(frame, results)
        
        return frame
    
    def _draw_landmarks_on_frame(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Draw landmarks on a frame (assumes frame dimensions match MediaPipe coordinate system)."""
        # Draw pose landmarks and connections
        if results.get('pose_landmarks'):
            pose_landmarks_list = results['pose_landmarks']
            # Handle both single pose and multiple poses
            if not isinstance(pose_landmarks_list, list):
                pose_landmarks_list = [pose_landmarks_list]
            
            # Draw each detected pose with different colors
            pose_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Red, Blue
            
            for i, pose_landmarks in enumerate(pose_landmarks_list):
                pose_color = pose_colors[i % len(pose_colors)]
                logger.debug(f"Drawing pose {i+1} with color {pose_color}")
                
                if self.config.SHOW_POSE_CONNECTIONS:
                    self.mp_drawing.draw_landmarks(
                        frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=pose_color, 
                            thickness=self.config.LANDMARK_THICKNESS, 
                            circle_radius=self.config.LANDMARK_CIRCLE_RADIUS
                        ),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=pose_color, 
                            thickness=self.config.LANDMARK_THICKNESS
                        )
                    )
                
                if self.config.SHOW_BODY_LANDMARKS:
                    self.mp_drawing.draw_landmarks(
                        frame, pose_landmarks, None,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=pose_color, 
                            thickness=self.config.LANDMARK_THICKNESS, 
                            circle_radius=self.config.LANDMARK_CIRCLE_RADIUS
                        ),
                        connection_drawing_spec=None
                    )
        
        # Draw face landmarks and connections
        if results.get('face_landmarks'):
            face_landmarks_list = results['face_landmarks']
            # Handle both single face and multiple faces
            if not isinstance(face_landmarks_list, list):
                face_landmarks_list = [face_landmarks_list]
            
            # Draw each detected face with different colors
            face_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]  # Cyan, Magenta, Yellow
            
            for i, face_landmarks in enumerate(face_landmarks_list):
                face_color = face_colors[i % len(face_colors)]
                logger.debug(f"Drawing face {i+1} with color {face_color}")
                
                # Log frame counter for this face
                cache_key = f"face_{i}"
                if cache_key in self._face_contour_frame_counts:
                    logger.debug(f"Face {i+1} frame count: {self._face_contour_frame_counts[cache_key]}")
                else:
                    logger.debug(f"Face {i+1} frame count: not initialized")
                
                if self.config.SHOW_FACE_CONNECTIONS:
                    # Draw dynamic face contour (outer boundary of face as flat figure)
                    self._draw_dynamic_face_contour(
                        frame, face_landmarks,
                        color=face_color,
                        thickness=self.config.LANDMARK_THICKNESS,
                        face_id=i
                    )
                
                if self.config.SHOW_FACE_LANDMARKS:
                    # Draw only the landmark points that are on the face boundary
                    self._draw_face_boundary_landmarks(
                        frame, face_landmarks,
                        color=face_color,
                        thickness=max(1, self.config.LANDMARK_THICKNESS // 2),
                        circle_radius=max(1, self.config.LANDMARK_CIRCLE_RADIUS // 2),
                        face_id=i
                    )
        
        # Draw hand landmarks and connections
        for hand_type, landmarks in [('left', results.get('left_hand_landmarks')), 
                                   ('right', results.get('right_hand_landmarks'))]:
            if landmarks:
                # Handle both single hand and multiple hands
                if not isinstance(landmarks, list):
                    landmarks = [landmarks]
                
                # Different colors for different people's hands
                hand_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
                
                for i, hand_landmarks in enumerate(landmarks):
                    hand_color = hand_colors[i % len(hand_colors)]
                    logger.debug(f"Drawing {hand_type} hand {i+1} with color {hand_color}")
                    
                    if self.config.SHOW_HAND_CONNECTIONS:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                                color=hand_color, 
                                thickness=self.config.LANDMARK_THICKNESS
                            )
                        )
                    
                    if self.config.SHOW_HAND_LANDMARKS:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, None,
                            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                                color=hand_color, 
                                thickness=self.config.LANDMARK_THICKNESS, 
                                circle_radius=self.config.LANDMARK_CIRCLE_RADIUS
                            ),
                            connection_drawing_spec=None
                        )
        
        return frame
    
    def _draw_dynamic_face_contour(self, frame: np.ndarray, landmarks, color: Tuple[int, int, int], thickness: int, face_id: int = 0):
        """Draw dynamic face contour - outer boundary of face as flat figure (optimized)."""
        # Use face-specific cache and frame counter
        cache_key = f"face_{face_id}"
        
        # Initialize frame counter for this face if not exists
        if cache_key not in self._face_contour_frame_counts:
            self._face_contour_frame_counts[cache_key] = 0
        
        # Increment frame counter for this specific face
        self._face_contour_frame_counts[cache_key] += 1
        
        # Check if we should use cached contour for this face
        if (cache_key in self._face_contour_caches and 
            self._face_contour_frame_counts[cache_key] % self._face_contour_update_interval != 0):
            # Use cached contour for this specific face
            cv2.polylines(frame, [self._face_contour_caches[cache_key]], True, color, thickness)
            return
        
        # Optimization: Use only key face points instead of all 468 points
        # These are the most important points for face boundary (50 points total)
        key_face_indices = [
            # Chin and jawline (8 points)
            10, 152, 234, 454, 172, 136, 150, 149,
            # Forehead (8 points)
            33, 7, 163, 144, 9, 10, 151, 337,
            # Left side face (12 points)
            172, 136, 150, 149, 148, 21, 162, 127, 234, 93, 132, 58,
            # Right side face (12 points)
            397, 288, 361, 323, 454, 356, 389, 251, 234, 361, 323, 288,
            # Nose area (10 points)
            70, 63, 105, 66, 107, 55, 65, 52, 53, 46
        ]
        
        points = []
        height, width = frame.shape[:2]
        
        # Use only key points (much faster than all 468 points)
        for i in key_face_indices:
            if i < len(landmarks.landmark):
                landmark = landmarks.landmark[i]
                # Convert normalized coordinates to pixel coordinates
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                
                # Ensure coordinates are within frame bounds
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                
                points.append([x, y])
        
        if len(points) < 3:
            return
        
        # Convert to numpy array
        points = np.array(points, dtype=np.int32)
        
        try:
            # Find convex hull (outer boundary) of key face points
            hull = cv2.convexHull(points)
            
            # Cache the result for future frames (face-specific)
            self._face_contour_caches[cache_key] = hull
            
            # Draw the contour
            cv2.polylines(frame, [hull], True, color, thickness)
            
        except Exception as e:
            logger.error(f"Exception in convex hull calculation: {e}")
            logger.error(f"Points array shape: {points.shape}")
            raise
    
    def _draw_face_boundary_landmarks(self, frame: np.ndarray, landmarks, color: Tuple[int, int, int], 
                                     thickness: int, circle_radius: int, face_id: int = 0):
        """Draw only the landmark points that are on the face boundary (optimized)."""
        # Use face-specific cache and frame counter
        cache_key = f"face_{face_id}"
        
        # Check if we should use cached boundary points for this face
        if (cache_key in self._face_contour_caches and 
            cache_key in self._face_contour_frame_counts and
            self._face_contour_frame_counts[cache_key] % self._face_contour_update_interval != 0):
            # Draw cached boundary points for this specific face
            for point in self._face_contour_caches[cache_key]:
                x, y = point[0]
                cv2.circle(frame, (x, y), circle_radius, color, thickness)
            return
        
        # Use the same key face indices as in contour calculation (50 points total)
        key_face_indices = [
            # Chin and jawline (8 points)
            10, 152, 234, 454, 172, 136, 150, 149,
            # Forehead (8 points)
            33, 7, 163, 144, 9, 10, 151, 337,
            # Left side face (12 points)
            172, 136, 150, 149, 148, 21, 162, 127, 234, 93, 132, 58,
            # Right side face (12 points)
            397, 288, 361, 323, 454, 356, 389, 251, 234, 361, 323, 288,
            # Nose area (10 points)
            70, 63, 105, 66, 107, 55, 65, 52, 53, 46
        ]
        
        points = []
        point_indices = []
        height, width = frame.shape[:2]
        
        # Use only key points (much faster)
        for i in key_face_indices:
            if i < len(landmarks.landmark):
                landmark = landmarks.landmark[i]
                # Convert normalized coordinates to pixel coordinates
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                
                # Ensure coordinates are within frame bounds
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                
                points.append([x, y])
                point_indices.append(i)
        
        if len(points) < 3:
            return
        
        # Convert to numpy array
        points = np.array(points, dtype=np.int32)
        
        try:
            # Find convex hull (outer boundary) of key face points
            hull_indices = cv2.convexHull(points, returnPoints=False)
            
            # Draw only the landmarks that are on the boundary
            for hull_idx in hull_indices:
                point_idx = point_indices[hull_idx[0]]
                landmark = landmarks.landmark[point_idx]
                
                # Convert normalized coordinates to pixel coordinates
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                
                # Ensure coordinates are within frame bounds
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                
                # Draw the landmark point
                cv2.circle(frame, (x, y), circle_radius, color, thickness)
                
        except Exception as e:
            logger.error(f"Exception in boundary landmarks calculation: {e}")
            raise
    
    
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        if hasattr(self, 'hands'):
            self.hands.close()

# Backward compatibility
PoseDetector = ComprehensivePoseDetector