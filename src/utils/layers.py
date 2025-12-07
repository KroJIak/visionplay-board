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

class BBoxLayer(Layer):
    """Layer that draws bounding boxes (e.g., person detections)."""
    
    def __init__(self, name: str = "BBoxes", alpha: float = 1.0, scaler=None):
        super().__init__(name, alpha, enabled=True)
        
        # Skeleton fade-out tracking
        self.skeleton_fade_times = {}  # Track when skeletons disappeared
        self.fade_duration = 0.5  # 0.5 seconds fade duration
        self.last_skeleton_data = {}  # Store last known skeleton positions
        self.distance_threshold = 300  # Pixels - if skeleton moves more than this, it's a new person
        self.scaler = scaler

    def render(self, base_frame: np.ndarray, data: dict) -> np.ndarray:
        if not self.enabled:
            return base_frame

        frame = base_frame.copy()
        bboxes = data.get('bboxes') or []
        face_contours = data.get('face_contours') or []
        pose_lines = data.get('pose_lines') or []
        hand_lines = data.get('hand_lines') or []
        
        current_time = time.time()
        
        # Update skeleton tracking for fade-out
        self._update_skeleton_tracking(pose_lines, hand_lines, face_contours, current_time)

        for (x, y, w, h) in bboxes:
            # Scale bbox coordinates if scaler is provided (camera->window)
            if self.scaler:
                x1, y1 = self.scaler.scale_coordinates(x, y)
                x2, y2 = self.scaler.scale_coordinates(x + w, y + h)
            else:
                x1, y1 = x, y
                x2, y2 = x + w, y + h

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # Draw label with count index
            idx = bboxes.index((x, y, w, h)) + 1
            label = f"P{idx}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw face contours (if provided) with fade-out
        for person_idx, contour in enumerate(face_contours):
            if not contour:
                continue
            alpha = self._get_skeleton_alpha(person_idx, current_time)
            if alpha > 0:
                pts = np.array(contour, dtype=np.int32)
                # No scaling needed: coordinates are already in window space if bboxes were scaled in app
                color = (0, int(255 * alpha), 0)  # Green with alpha
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        # Draw pose lines (skeleton) with fade-out
        total_pose_segments = 0
        for person_idx, person_lines in enumerate(pose_lines):
            alpha = self._get_skeleton_alpha(person_idx, current_time)
            if alpha > 0:
                for (x1, y1, x2, y2) in person_lines:
                    # Coordinates are already scaled to window in app
                    color = (0, int(255 * alpha), 0)  # Green with alpha
                    cv2.line(frame, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
                    # Joints for visibility
                    cv2.circle(frame, (x1, y1), 4, (255, 255, 255), -1)
                    cv2.circle(frame, (x1, y1), 2, color, -1)
                    cv2.circle(frame, (x2, y2), 4, (255, 255, 255), -1)
                    cv2.circle(frame, (x2, y2), 2, color, -1)
                    total_pose_segments += 1
        # Debug: if present, mark the first joint with a big dot
        if pose_lines and pose_lines[0]:
            x1, y1, x2, y2 = pose_lines[0][0]
            cv2.circle(frame, (x1, y1), 8, (0, 0, 255), -1)

        # Draw hand lines with fade-out
        total_hand_segments = 0
        for person_idx, person_hands in enumerate(hand_lines):
            alpha = self._get_skeleton_alpha(person_idx, current_time)
            if alpha > 0:
                for (x1, y1, x2, y2) in person_hands:
                    color = (int(255 * alpha), 0, int(255 * alpha))  # Magenta with alpha
                    cv2.line(frame, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
                    cv2.circle(frame, (x1, y1), 3, (255, 255, 255), -1)
                    cv2.circle(frame, (x1, y1), 2, color, -1)
                    cv2.circle(frame, (x2, y2), 3, (255, 255, 255), -1)
                    cv2.circle(frame, (x2, y2), 2, color, -1)
                    total_hand_segments += 1

        # On-screen debug counters (top-left)
        cv2.putText(frame, f"pose_segments: {total_pose_segments}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"hand_segments: {total_hand_segments}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)

        self.render_count += 1
        self.last_render_time = time.time()
        return frame
    
    def _update_skeleton_tracking(self, pose_lines, hand_lines, face_contours, current_time):
        """Update skeleton tracking for fade-out effect."""
        # Track current skeletons and their positions
        current_skeletons = {}
        
        # Check pose lines and get center positions
        for person_idx, person_lines in enumerate(pose_lines):
            if person_lines:  # If person has pose data
                center_pos = self._get_skeleton_center(person_lines)
                current_skeletons[person_idx] = center_pos
        
        # Check hand lines and get center positions
        for person_idx, person_hands in enumerate(hand_lines):
            if person_hands:  # If person has hand data
                center_pos = self._get_skeleton_center(person_hands)
                if person_idx in current_skeletons:
                    # Use average of pose and hand centers
                    current_skeletons[person_idx] = (
                        (current_skeletons[person_idx][0] + center_pos[0]) // 2,
                        (current_skeletons[person_idx][1] + center_pos[1]) // 2
                    )
                else:
                    current_skeletons[person_idx] = center_pos
        
        # Check face contours and get center positions
        for person_idx, contour in enumerate(face_contours):
            if contour:  # If person has face data
                center_pos = self._get_contour_center(contour)
                if person_idx in current_skeletons:
                    # Use average of existing and face centers
                    current_skeletons[person_idx] = (
                        (current_skeletons[person_idx][0] + center_pos[0]) // 2,
                        (current_skeletons[person_idx][1] + center_pos[1]) // 2
                    )
                else:
                    current_skeletons[person_idx] = center_pos
        
        # Check for position jumps (new person in different location)
        for person_idx, current_pos in current_skeletons.items():
            if person_idx in self.last_skeleton_data:
                last_pos = self.last_skeleton_data[person_idx]
                distance = ((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)**0.5
                
                if distance > self.distance_threshold:
                    # Person moved too far - treat as new person, fade out old position
                    print(f"Person {person_idx} moved {distance:.1f} pixels - triggering fade-out")
                    self.skeleton_fade_times[person_idx] = current_time
        
        # Update fade times for disappeared skeletons
        for person_idx in list(self.skeleton_fade_times.keys()):
            if person_idx not in current_skeletons:
                # Skeleton disappeared, start fade timer if not already started
                if person_idx not in self.skeleton_fade_times:
                    self.skeleton_fade_times[person_idx] = current_time
            else:
                # Skeleton is present and hasn't moved too far, remove from fade tracking
                if person_idx in self.skeleton_fade_times:
                    del self.skeleton_fade_times[person_idx]
        
        # Clean up old fade times (skeletons that have completely faded)
        for person_idx in list(self.skeleton_fade_times.keys()):
            if current_time - self.skeleton_fade_times[person_idx] > self.fade_duration:
                del self.skeleton_fade_times[person_idx]
        
        # Update last known positions
        self.last_skeleton_data = current_skeletons.copy()
    
    def _get_skeleton_alpha(self, person_idx, current_time):
        """Get alpha value for skeleton fade-out effect."""
        if person_idx in self.skeleton_fade_times:
            # Skeleton is fading out
            fade_elapsed = current_time - self.skeleton_fade_times[person_idx]
            if fade_elapsed >= self.fade_duration:
                return 0.0  # Completely faded
            else:
                # Linear fade from 1.0 to 0.0
                alpha = 1.0 - (fade_elapsed / self.fade_duration)
                return max(0.0, alpha)
        else:
            # Skeleton is present and visible
            return 1.0
    
    def _get_skeleton_center(self, lines):
        """Get center position of skeleton lines."""
        if not lines:
            return (0, 0)
        
        x_coords = []
        y_coords = []
        for (x1, y1, x2, y2) in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        if x_coords and y_coords:
            center_x = sum(x_coords) // len(x_coords)
            center_y = sum(y_coords) // len(y_coords)
            return (center_x, center_y)
        return (0, 0)
    
    def _get_contour_center(self, contour):
        """Get center position of face contour."""
        if not contour:
            return (0, 0)
        
        pts = np.array(contour, dtype=np.int32)
        if len(pts) > 0:
            # Calculate centroid
            moments = cv2.moments(pts)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                return (cx, cy)
        
        # Fallback: use bounding box center
        x, y, w, h = cv2.boundingRect(pts)
        return (x + w // 2, y + h // 2)

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