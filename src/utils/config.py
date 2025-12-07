"""
Configuration utilities for VisionPlay Board application.
"""

import os
from typing import Tuple, Any
from dotenv import load_dotenv

class Config:
    """Configuration manager for the application."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        load_dotenv()
        
        # Camera settings
        self.CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
        self.CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '640'))
        self.CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '480'))
        self.CAMERA_FPS = int(os.getenv('CAMERA_FPS', '30'))
        self.USE_MJPEG_CODEC = os.getenv('USE_MJPEG_CODEC', 'true').lower() == 'true'
        self.CAMERA_BUFFER_SIZE = int(os.getenv('CAMERA_BUFFER_SIZE', '1'))
        self.CAMERA_FOURCC = os.getenv('CAMERA_FOURCC', 'MJPG')
        
        # Display window settings (fixed size)
        self.WINDOW_WIDTH = int(os.getenv('WINDOW_WIDTH', '1920'))
        self.WINDOW_HEIGHT = int(os.getenv('WINDOW_HEIGHT', '1080'))
        
        # Game settings
        self.HUMAN_DETECTION_TIMEOUT = float(os.getenv('HUMAN_DETECTION_TIMEOUT', '3.0'))
        self.GAME_TILE_WIDTH = int(os.getenv('GAME_TILE_WIDTH', '200'))
        self.GAME_TILE_HEIGHT = int(os.getenv('GAME_TILE_HEIGHT', '300'))
        self.GAME_TILE_SPACING = int(os.getenv('GAME_TILE_SPACING', '50'))
        
        # Menu settings
        self.MENU_BLUR_KERNEL = int(os.getenv('MENU_BLUR_KERNEL', '40'))
        
        # Display settings
        self.FULLSCREEN_MODE = os.getenv('FULLSCREEN_MODE', 'true').lower() == 'true'
        self.WINDOW_TITLE = os.getenv('WINDOW_TITLE', 'VisionPlay Board')
        
        # MediaPipe settings
        self.POSE_CONFIDENCE_THRESHOLD = float(os.getenv('POSE_CONFIDENCE_THRESHOLD', '0.5'))
        self.POSE_TRACKING_CONFIDENCE = float(os.getenv('POSE_TRACKING_CONFIDENCE', '0.5'))
        self.POSE_PRESENCE_CONFIDENCE = float(os.getenv('POSE_PRESENCE_CONFIDENCE', '0.5'))
        self.MAX_PEOPLE_IN_FRAME = int(os.getenv('MAX_PEOPLE_IN_FRAME', '4'))
        self.ENABLE_POSE_DETECTION = os.getenv('ENABLE_POSE_DETECTION', 'false').lower() == 'true'
        self.MODEL_COMPLEXITY = int(os.getenv('MODEL_COMPLEXITY', '1'))
        
        # Detailed pose visualization control
        self.SHOW_BODY_LANDMARKS = os.getenv('SHOW_BODY_LANDMARKS', 'true').lower() == 'true'
        self.SHOW_FACE_LANDMARKS = os.getenv('SHOW_FACE_LANDMARKS', 'true').lower() == 'true'
        self.SHOW_HAND_LANDMARKS = os.getenv('SHOW_HAND_LANDMARKS', 'true').lower() == 'true'
        self.SHOW_POSE_CONNECTIONS = os.getenv('SHOW_POSE_CONNECTIONS', 'true').lower() == 'true'
        self.SHOW_FACE_CONNECTIONS = os.getenv('SHOW_FACE_CONNECTIONS', 'true').lower() == 'true'
        self.SHOW_HAND_CONNECTIONS = os.getenv('SHOW_HAND_CONNECTIONS', 'true').lower() == 'true'
        self.LANDMARK_VISIBILITY_THRESHOLD = float(os.getenv('LANDMARK_VISIBILITY_THRESHOLD', '0.3'))
        self.LANDMARK_CIRCLE_RADIUS = int(os.getenv('LANDMARK_CIRCLE_RADIUS', '4'))
        self.LANDMARK_THICKNESS = int(os.getenv('LANDMARK_THICKNESS', '3'))
        
        # Thread queue settings (simplified)
        self.FRAME_QUEUE_SIZE = int(os.getenv('FRAME_QUEUE_SIZE', '2'))
        self.RESULT_QUEUE_SIZE = int(os.getenv('RESULT_QUEUE_SIZE', '1'))
        
        
        
        # Performance settings
        self.SHOW_FPS = os.getenv('SHOW_FPS', 'true').lower() == 'true'
        self.SHOW_STATISTICS_ON_S_KEY = os.getenv('SHOW_STATISTICS_ON_S_KEY', 'true').lower() == 'true'
        
        # Camera optimization settings
        self.USE_MJPEG_CODEC = os.getenv('USE_MJPEG_CODEC', 'true').lower() == 'true'
        self.CAMERA_BUFFER_SIZE = int(os.getenv('CAMERA_BUFFER_SIZE', '1'))
        self.CAMERA_FPS = int(os.getenv('CAMERA_FPS', '30'))
        
        # Colors (BGR format for OpenCV) - Blue palette
        self.COLOR_SUCCESS = self._parse_color(os.getenv('COLOR_SUCCESS', '(0, 255, 0)'))
        self.COLOR_FAILURE = self._parse_color(os.getenv('COLOR_FAILURE', '(0, 0, 255)'))
        self.COLOR_NEUTRAL = self._parse_color(os.getenv('COLOR_NEUTRAL', '(255, 255, 255)'))
        self.COLOR_BLACK = self._parse_color(os.getenv('COLOR_BLACK', '(0, 0, 0)'))
        
        # Blue palette colors
        self.COLOR_BLUE_DARK = self._parse_color(os.getenv('COLOR_BLUE_DARK', '(139, 69, 19)'))      # Dark blue
        self.COLOR_BLUE_MEDIUM = self._parse_color(os.getenv('COLOR_BLUE_MEDIUM', '(255, 140, 0)'))   # Medium blue  
        self.COLOR_BLUE_LIGHT = self._parse_color(os.getenv('COLOR_BLUE_LIGHT', '(255, 191, 0)'))     # Light blue
        self.COLOR_BLUE_ACCENT = self._parse_color(os.getenv('COLOR_BLUE_ACCENT', '(255, 215, 0)'))   # Accent blue
        
        # Tile colors
        self.COLOR_TILE_BG = self.COLOR_BLUE_MEDIUM
        self.COLOR_TILE_ACTIVE = self.COLOR_BLUE_ACCENT
    
    def _parse_color(self, color_str: str) -> Tuple[int, int, int]:
        """Parse color string in format '(B, G, R)' to tuple."""
        try:
            # Remove parentheses and split by comma
            color_str = color_str.strip('()')
            b, g, r = map(int, color_str.split(','))
            return (b, g, r)
        except (ValueError, AttributeError):
            # Default to white if parsing fails
            return (255, 255, 255)
    
    def get_camera_resolution(self) -> Tuple[int, int]:
        """Get camera resolution as tuple."""
        return (self.CAMERA_WIDTH, self.CAMERA_HEIGHT)
