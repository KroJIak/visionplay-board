"""
Simplified layer system for VisionPlay Board rendering.
Only 2 layers: Background (camera + UI) and Skeleton (pose detection).
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import threading
import time

class Layer(ABC):
    """Abstract base class for rendering layers."""
    
    def __init__(self, name: str, alpha: float = 1.0, enabled: bool = True):
        """Initialize layer."""
        self.name = name
        self.alpha = alpha
        self.enabled = enabled
        self.last_render_time = 0
        self.render_count = 0
    
    @abstractmethod
    def render(self, base_frame: np.ndarray, data: dict) -> np.ndarray:
        """Render layer content on base frame."""
        pass
    
    def is_enabled(self) -> bool:
        """Check if layer is enabled."""
        return self.enabled
    
    def set_enabled(self, enabled: bool):
        """Enable or disable layer."""
        self.enabled = enabled
    
    def get_render_stats(self) -> dict:
        """Get rendering statistics."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'alpha': self.alpha,
            'render_count': self.render_count,
            'last_render_time': self.last_render_time
        }

class CameraLayer(Layer):
    """Background layer - camera + UI elements."""
    
    def __init__(self, name: str = "Background"):
        """Initialize background layer."""
        super().__init__(name, alpha=1.0, enabled=True)
    
    def render(self, base_frame: np.ndarray, data: dict) -> np.ndarray:
        """Render camera frame with UI elements."""
        if not self.enabled:
            return base_frame
        
        start_time = time.time()
        
        # Start with camera frame
        result_frame = base_frame.copy()
        
        # Apply UI elements if present
        if 'ui_elements' in data:
            result_frame = self._render_ui_elements(result_frame, data['ui_elements'])
        
        # Apply blur if needed
        if 'blur_background' in data:
            result_frame = self._apply_blur(result_frame, data['blur_background'])
        
        self.last_render_time = time.time() - start_time
        self.render_count += 1
        
        return result_frame
    
    def _render_ui_elements(self, frame: np.ndarray, ui_elements: list) -> np.ndarray:
        """Render UI elements on frame."""
        for element in ui_elements:
            try:
                if element['type'] == 'text':
                    cv2.putText(frame, element['text'], element['position'],
                               element['font'], element['scale'], element['color'], element['thickness'])
                elif element['type'] == 'rectangle':
                    if 'pt1' in element and 'pt2' in element:
                        cv2.rectangle(frame, element['pt1'], element['pt2'],
                                     element['color'], element['thickness'])
                    elif 'position' in element and 'size' in element:
                        x, y = element['position']
                        w, h = element['size']
                        cv2.rectangle(frame, (x, y), (x + w, y + h),
                                     element['color'], element['thickness'])
                elif element['type'] == 'circle':
                    cv2.circle(frame, element['center'], element['radius'],
                              element['color'], element['thickness'])
            except KeyError as e:
                print(f"Missing key in UI element: {e}")
                continue
        
        return frame
    
    def _apply_blur(self, frame: np.ndarray, blur_data: dict) -> np.ndarray:
        """Apply blur effect to frame."""
        kernel_size = blur_data.get('kernel_size', 15)
        return cv2.blur(frame, (kernel_size, kernel_size))

class PoseLayer(Layer):
    """Skeleton layer - only for pose detection visualization."""
    
    def __init__(self, name: str = "Skeleton", alpha: float = 1.0, scaler=None):
        """Initialize skeleton layer."""
        super().__init__(name, alpha, enabled=True)
        self.scaler = scaler
    
    def render(self, base_frame: np.ndarray, data: dict) -> np.ndarray:
        """Render skeleton only."""
        if not self.enabled:
            return base_frame
        
        frame = base_frame.copy()
        pose_results = data.get('pose_results')
        
        # Use the comprehensive pose detector to draw all landmarks
        if pose_results and hasattr(self, 'pose_detector') and self.pose_detector:
            frame = self.pose_detector.draw_comprehensive_pose(frame, pose_results, self.scaler)
        
        self.render_count += 1
        self.last_render_time = time.time()
        
        return frame

class LayerManager:
    """Manages rendering layers and composition."""
    
    def __init__(self):
        """Initialize layer manager."""
        self.layers: List[Layer] = []
        self.render_order = []
        self.lock = threading.Lock()
    
    def add_layer(self, layer: Layer, order: int = 0):
        """Add a layer to the manager."""
        with self.lock:
            self.layers.append(layer)
            self.render_order.append((order, layer))
            self.render_order.sort(key=lambda x: x[0])  # Sort by order
    
    def remove_layer(self, layer_name: str):
        """Remove a layer by name."""
        with self.lock:
            self.layers = [l for l in self.layers if l.name != layer_name]
            self.render_order = [(o, l) for o, l in self.render_order if l.name != layer_name]
    
    def get_layer(self, layer_name: str) -> Optional[Layer]:
        """Get a layer by name."""
        for layer in self.layers:
            if layer.name == layer_name:
                return layer
        return None
    
    def render_all(self, base_frame: np.ndarray, data: dict) -> np.ndarray:
        """Render all layers in order."""
        if not base_frame is not None:
            return base_frame
        
        with self.lock:
            current_frame = base_frame.copy()
            
            # Render each layer in order
            for order, layer in self.render_order:
                if layer.is_enabled():
                    current_frame = layer.render(current_frame, data)
            
            return current_frame
    
    def get_all_stats(self) -> dict:
        """Get statistics for all layers."""
        with self.lock:
            stats = {}
            for layer in self.layers:
                stats[layer.name] = layer.get_render_stats()
            return stats
    
    def enable_layer(self, layer_name: str, enabled: bool = True):
        """Enable or disable a layer."""
        layer = self.get_layer(layer_name)
        if layer:
            layer.set_enabled(enabled)
    
    def set_layer_alpha(self, layer_name: str, alpha: float):
        """Set alpha value for a layer."""
        layer = self.get_layer(layer_name)
        if layer:
            layer.alpha = alpha