"""
Adaptive scaling utilities for camera frames.
"""

import cv2
import numpy as np
from typing import Tuple

class AdaptiveScaler:
    """Handles adaptive scaling of camera frames to fit display window."""
    
    def __init__(self, camera_width: int, camera_height: int, 
                 window_width: int, window_height: int):
        """Initialize scaler with camera and window dimensions."""
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.window_width = window_width
        self.window_height = window_height
        
        # Calculate scaling factors
        self.scale_x = window_width / camera_width
        self.scale_y = window_height / camera_height
        
        # Use the larger scale to ensure the image covers the entire window
        self.scale = max(self.scale_x, self.scale_y)
        
        # Calculate scaled dimensions
        self.scaled_width = int(camera_width * self.scale)
        self.scaled_height = int(camera_height * self.scale)
        
        # Calculate crop offsets to center the image
        self.crop_x = (self.scaled_width - window_width) // 2
        self.crop_y = (self.scaled_height - window_height) // 2
        
        print(f"AdaptiveScaler initialized:")
        print(f"  Camera: {camera_width}x{camera_height}")
        print(f"  Window: {window_width}x{window_height}")
        print(f"  Scale: {self.scale:.3f}")
        print(f"  Scaled: {self.scaled_width}x{self.scaled_height}")
        print(f"  Crop offset: ({self.crop_x}, {self.crop_y})")
    
    def scale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Scale and crop frame to fit window dimensions."""
        if frame is None:
            return None
        
        # Scale the frame
        scaled_frame = cv2.resize(frame, (self.scaled_width, self.scaled_height), 
                                 interpolation=cv2.INTER_LINEAR)
        
        # Crop to window dimensions
        cropped_frame = scaled_frame[
            self.crop_y:self.crop_y + self.window_height,
            self.crop_x:self.crop_x + self.window_width
        ]
        
        return cropped_frame
    
    def scale_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """Scale coordinates from camera space to window space."""
        # First scale up
        scaled_x = int(x * self.scale)
        scaled_y = int(y * self.scale)
        
        # Then apply crop offset
        window_x = scaled_x - self.crop_x
        window_y = scaled_y - self.crop_y
        
        return window_x, window_y
    
    def scale_rectangle(self, x: int, y: int, width: int, height: int) -> Tuple[int, int, int, int]:
        """Scale rectangle from camera space to window space."""
        # Scale position
        scaled_x, scaled_y = self.scale_coordinates(x, y)
        
        # Scale dimensions
        scaled_width = int(width * self.scale)
        scaled_height = int(height * self.scale)
        
        return scaled_x, scaled_y, scaled_width, scaled_height
    
    def get_scale_factor(self) -> float:
        """Get the scaling factor used."""
        return self.scale
