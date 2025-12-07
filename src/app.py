"""
Main VisionPlay Board application.
"""

import cv2
import numpy as np
import time
import random
from typing import List, Optional, Tuple

from .utils.config import Config
from .utils.layers import LayerManager, CameraLayer, BBoxLayer
from .utils.thread_manager import ThreadManager
from .utils.yolo_person_detector import YoloPersonDetector
from .utils.yolo_holistic_detector import YoloHolisticDetector
from .utils.scaling import AdaptiveScaler
from .games.skeleton_viewer_game import SkeletonViewerGame
from .games.hide_and_seek_game import HideAndSeekGame

class GameTile:
    """Represents a game tile on the main menu."""
    
    def __init__(self, x: int, y: int, width: int, height: int, game_name: str):
        """Initialize game tile."""
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.game_name = game_name
        self.is_active = False
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside tile."""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)

class VisionPlayApp:
    """Main application class for VisionPlay Board."""
    
    def __init__(self):
        """Initialize the application."""
        self.config = Config()
        
        # Initialize YOLO + FaceMesh detector (only if pose detection is enabled)
        if self.config.ENABLE_POSE_DETECTION:
            self.detector = YoloHolisticDetector()
            print("YOLO + FaceMesh detector initialized")
        else:
            self.detector = None
            print("Pose detection disabled - detector not initialized")
        
        # Initialize thread manager
        self.thread_manager = ThreadManager(self.config)
        
        
        # Initialize camera in main thread
        self.cap = None
        self._init_camera()
        
        # Initialize adaptive scaler
        self.scaler = AdaptiveScaler(
            self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT,
            self.config.WINDOW_WIDTH, self.config.WINDOW_HEIGHT
        )
        
        # Initialize layer manager for rendering
        self.layer_manager = LayerManager()
        self._setup_layers()
        
        # Initialize games
        self.games = {
            "Skeleton Viewer": SkeletonViewerGame(self.config, self.scaler, None),
            "Hide and Seek": HideAndSeekGame()
        }
        
        # Game state
        self.is_in_menu = True
        self.current_game = None
        self.game_tiles = self._create_game_tiles()
        
        # Human detection
        self.human_detection_start = None
        self.human_detection_timeout = self.config.HUMAN_DETECTION_TIMEOUT
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.frame_counter = 0  # For frame skipping
        
        # Mouse callback for tile interaction
        self.mouse_callback_set = False
    
    def _init_camera(self):
        """Initialize camera in main thread."""
        # Try V4L2 backend first for better Linux compatibility
        self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera {self.config.CAMERA_INDEX}")
        
        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
        
        # Enable MJPEG codec for better performance
        if self.config.USE_MJPEG_CODEC:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            print("MJPEG codec enabled for better performance")
        
        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.CAMERA_BUFFER_SIZE)
        
        # Additional optimizations for better performance
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
        
        # Verify camera settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    def _setup_layers(self):
        """Setup rendering layers - simplified to 2 layers only."""
        # Background layer (camera + UI elements)
        background_layer = CameraLayer("Background")
        self.layer_manager.add_layer(background_layer, order=0)
        
        # BBoxes layer (person detection only)
        bbox_layer = BBoxLayer("BBoxes", alpha=1.0, scaler=self.scaler)
        bbox_layer.set_enabled(True)
        self.layer_manager.add_layer(bbox_layer, order=1)
    
    def _create_game_tiles(self) -> List[GameTile]:
        """Create game tiles for the main menu."""
        tiles = []
        
        # Available games
        game_names = list(self.games.keys())
        
        # Calculate tile positions (use window dimensions for UI layout)
        total_width = len(game_names) * self.config.GAME_TILE_WIDTH + (len(game_names) - 1) * self.config.GAME_TILE_SPACING
        start_x = (self.config.WINDOW_WIDTH - total_width) // 2
        start_y = (self.config.WINDOW_HEIGHT - self.config.GAME_TILE_HEIGHT) // 2
        
        for i, game_name in enumerate(game_names):
            x = start_x + i * (self.config.GAME_TILE_WIDTH + self.config.GAME_TILE_SPACING)
            y = start_y
            tile = GameTile(x, y, self.config.GAME_TILE_WIDTH, self.config.GAME_TILE_HEIGHT, game_name)
            tiles.append(tile)
        
        return tiles
    
    def _setup_mouse_callback(self, window_name: str):
        """Setup mouse callback for tile interaction."""
        if not self.mouse_callback_set:
            cv2.setMouseCallback(window_name, self._mouse_callback)
            self.mouse_callback_set = True
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for game tile selection."""
        if event == cv2.EVENT_LBUTTONDOWN and self.is_in_menu:
            for tile in self.game_tiles:
                if tile.contains_point(x, y):
                    self._start_game(tile.game_name)
                    break
    
    def _start_game(self, game_name: str):
        """Start a specific game."""
        if game_name in self.games:
            self.current_game = self.games[game_name]
            self.current_game.start()
            self.is_in_menu = False
            # Reset human detection timer when starting game manually
            self.human_detection_start = None
            print(f"Started game: {game_name}")
        else:
            print(f"Game not found: {game_name}")
    
    def _start_random_game(self):
        """Start a random game."""
        if self.games:
            game_name = random.choice(list(self.games.keys()))
            self._start_game(game_name)
            # Reset human detection timer after starting random game
            self.human_detection_start = None
    
    def _check_human_detection_from_bboxes(self, bboxes):
        """Check for human detection using bboxes and start random game if timeout reached."""
        # Only check if pose detection is enabled
        if not self.config.ENABLE_POSE_DETECTION:
            return
            
        if not bboxes:
            self.human_detection_start = None
            return
        if self.human_detection_start is None:
            self.human_detection_start = time.time()
        elif time.time() - self.human_detection_start >= self.human_detection_timeout:
            self._start_random_game()
            self.human_detection_start = None
    
    def _update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def _get_menu_ui_elements(self) -> List[dict]:
        """Get UI elements for main menu."""
        elements = []
        
        # Create blurred background
        elements.append({
            'type': 'blur_background',
            'kernel_size': self.config.MENU_BLUR_KERNEL
        })
        
        # Title - VISIONPLAY BOARD (large, centered at top)
        title_text = "VISIONPLAY BOARD"
        title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 5)[0]
        title_x = (self.config.WINDOW_WIDTH - title_size[0]) // 2
        title_y = 120
        
        # Title shadow
        elements.append({
            'type': 'text',
            'text': title_text,
            'position': (title_x + 3, title_y + 3),
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'scale': 3.0,
            'color': self.config.COLOR_BLACK,
            'thickness': 5
        })
        
        # Title main (thicker)
        elements.append({
            'type': 'text',
            'text': title_text,
            'position': (title_x, title_y),
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'scale': 3.0,
            'color': self.config.COLOR_NEUTRAL,
            'thickness': 8
        })
        
        # Subtitle
        if self.config.ENABLE_POSE_DETECTION:
            subtitle_text = 'Click on a game or stand in front of camera for 3 seconds'
        else:
            subtitle_text = 'Click on a game to start playing'
        
        subtitle_size = cv2.getTextSize(subtitle_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        subtitle_x = (self.config.WINDOW_WIDTH - subtitle_size[0]) // 2
        subtitle_y = title_y + 80
        
        # Subtitle shadow
        elements.append({
            'type': 'text',
            'text': subtitle_text,
            'position': (subtitle_x + 2, subtitle_y + 2),
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'scale': 1.2,
            'color': self.config.COLOR_BLACK,
            'thickness': 3
        })
        
        # Subtitle main (white)
        elements.append({
            'type': 'text',
            'text': subtitle_text,
            'position': (subtitle_x, subtitle_y),
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'scale': 1.2,
            'color': self.config.COLOR_NEUTRAL,
            'thickness': 3
        })
        
        # Game tiles with modern blue design
        for tile in self.game_tiles:
            # Tile shadow
            elements.append({
                'type': 'rectangle',
                'position': (tile.x + 5, tile.y + 5),
                'size': (tile.width, tile.height),
                'color': self.config.COLOR_BLACK,
                'thickness': 0,
                'filled': True
            })
            
            # Tile background
            tile_color = self.config.COLOR_BLUE_ACCENT if tile.is_active else self.config.COLOR_BLUE_MEDIUM
            elements.append({
                'type': 'rectangle',
                'position': (tile.x, tile.y),
                'size': (tile.width, tile.height),
                'color': tile_color,
                'thickness': 0,
                'filled': True
            })
            
            # Tile border
            border_color = self.config.COLOR_NEUTRAL if tile.is_active else self.config.COLOR_BLUE_LIGHT
            elements.append({
                'type': 'rectangle',
                'position': (tile.x, tile.y),
                'size': (tile.width, tile.height),
                'color': border_color,
                'thickness': 4,
                'filled': False
            })
            
            # Game name text with better positioning
            game_text = tile.game_name.upper()
            text_size = cv2.getTextSize(game_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            text_x = tile.x + (tile.width - text_size[0]) // 2
            text_y = tile.y + (tile.height + text_size[1]) // 2
            
            # Text shadow
            elements.append({
                'type': 'text',
                'text': game_text,
                'position': (text_x + 2, text_y + 2),
                'font': cv2.FONT_HERSHEY_SIMPLEX,
                'scale': 1.0,
                'color': self.config.COLOR_BLACK,
                'thickness': 3
            })
            
            # Text main
            elements.append({
                'type': 'text',
                'text': game_text,
                'position': (text_x, text_y),
                'font': cv2.FONT_HERSHEY_SIMPLEX,
                'scale': 1.0,
                'color': self.config.COLOR_NEUTRAL,
                'thickness': 3
            })
        
        # Human detection status
        if self.config.ENABLE_POSE_DETECTION and self.human_detection_start:
            remaining_time = self.config.HUMAN_DETECTION_TIMEOUT - (time.time() - self.human_detection_start)
            if remaining_time > 0:
                status_text = f"Human detected! Starting random game in {remaining_time:.1f}s"
                status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                status_x = (self.config.WINDOW_WIDTH - status_size[0]) // 2
                status_y = self.config.WINDOW_HEIGHT - 80
                
                # Status shadow
                elements.append({
                    'type': 'text',
                    'text': status_text,
                    'position': (status_x + 2, status_y + 2),
                    'font': cv2.FONT_HERSHEY_SIMPLEX,
                    'scale': 1.5,
                    'color': self.config.COLOR_BLACK,
                    'thickness': 3
                })
                
                # Status main
                elements.append({
                    'type': 'text',
                    'text': status_text,
                    'position': (status_x, status_y),
                    'font': cv2.FONT_HERSHEY_SIMPLEX,
                    'scale': 1.5,
                    'color': self.config.COLOR_SUCCESS,
                    'thickness': 3
                })
        
        return elements
    
    def _get_game_ui_elements(self) -> List[dict]:
        """Get UI elements for game mode."""
        elements = []
        
        # FPS counter (top right corner)
        elements.append({
            'type': 'text',
            'text': f"FPS: {self.current_fps:.1f}",
            'position': (self.config.WINDOW_WIDTH - 150, 30),
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'scale': 0.7,
            'color': self.config.COLOR_NEUTRAL,
            'thickness': 2
        })
        
        return elements
    
    def _show_statistics(self):
        """Show performance statistics."""
        stats = self.thread_manager.get_stats()
        layer_stats = self.layer_manager.get_all_stats()
        print("\n=== Performance Statistics ===")
        print(f"Main thread FPS: {self.current_fps:.1f}")
        if 'pose_thread' in stats:
            print(f"Pose processing time: {stats['pose_thread'].get('processing_time', 0):.3f}s")
            print(f"Pose frames processed: {stats['pose_thread'].get('frame_count', 0)}")
        
        print("\n=== Layer Statistics ===")
        for layer_name, layer_stat in layer_stats.items():
            print(f"{layer_name}: {layer_stat['render_count']} renders, "
                  f"enabled: {layer_stat['enabled']}")
        print("=============================\n")
    
    def run(self):
        """Run the main application loop."""
        print("Starting VisionPlay Board application...")
        print("Press ESC to exit, click on tiles to start games")
        
        # Start detection thread (only if pose detection is enabled)
        if self.config.ENABLE_POSE_DETECTION and self.detector:
            self.thread_manager.start(self.detector)
        else:
            print("Detection thread not started - pose detection disabled")
        
        # Create window
        window_name = self.config.WINDOW_TITLE
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        if self.config.FULLSCREEN_MODE:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Setup mouse callback
        self._setup_mouse_callback(window_name)
        
        try:
            while True:
                # Capture frame from camera in main thread
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    continue
                
                # Mirror the frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Clear camera buffer to get latest frame (prevent lag)
                # Read and discard old frames to get the freshest one
                for _ in range(1):  # Only clear 1 frame to avoid too much lag
                    ret, _ = self.cap.read()
                    if not ret:
                        break
                
                # Scale frame to window dimensions FIRST (most important for display)
                scaled_frame = self.scaler.scale_frame(frame)
                
                # Send frame to detection thread (only if pose detection is enabled)
                self.frame_counter += 1
                if self.config.ENABLE_POSE_DETECTION and self.detector:
                    # Process every frame (no skipping)
                    self.thread_manager.put_frame_for_processing(frame)
                    
                    # Get latest detection result
                    result = self.thread_manager.get_latest_result()
                    bboxes = result.get('bboxes') if result else []
                    face_contours = result.get('face_contours') if result else []
                    pose_lines = result.get('pose_lines') if result else []
                    hand_lines = result.get('hand_lines') if result else []
                    
                    # Debug: log when detection results are empty
                    if not bboxes and not pose_lines and not hand_lines:
                        print(f"[Debug] Empty detection result at frame {self.frame_counter}")
                    elif bboxes or pose_lines or hand_lines:
                        print(f"[Debug] Detection OK: {len(bboxes)} bboxes, {sum(len(pl) for pl in pose_lines)} pose_segments, {sum(len(hl) for hl in hand_lines)} hand_segments")
                else:
                    # No detection - empty results
                    bboxes = []
                    face_contours = []
                    pose_lines = []
                    hand_lines = []
                
                # Update FPS
                self._update_fps()
                
                # Prepare layer data
                layer_data = {}
                
                if self.is_in_menu:
                    # Main menu mode
                    self._check_human_detection_from_bboxes(bboxes)
                    layer_data['ui_elements'] = self._get_menu_ui_elements()
                    layer_data['bboxes'] = bboxes
                    layer_data['face_contours'] = self._scale_contours_to_window(face_contours)
                    scaled_pose = self._scale_lines_to_window(pose_lines)
                    scaled_hands = self._scale_lines_to_window(hand_lines)
                    layer_data['pose_lines'] = scaled_pose
                    layer_data['hand_lines'] = scaled_hands
                    try:
                        pose_segments_count = sum(len(pl) for pl in scaled_pose)
                        hand_segments_count = sum(len(hl) for hl in scaled_hands)
                        print(f"[Render Debug] pose_segments={pose_segments_count} hand_segments={hand_segments_count}")
                    except Exception:
                        pass
                    
                    # Render all layers
                    final_frame = self.layer_manager.render_all(scaled_frame, layer_data)
                else:
                    # Game mode
                    if self.current_game:
                        # Let the game handle the frame
                        # Prepare detection data for the game (scaled to window/game frame)
                        detection_data_for_game = {
                            'bboxes': bboxes,
                            'face_contours': self._scale_contours_to_window(face_contours),
                            'pose_lines': self._scale_lines_to_window(pose_lines),
                            'hand_lines': self._scale_lines_to_window(hand_lines),
                        }
                        game_result = self.current_game.update(scaled_frame, detection_data_for_game)
                        if isinstance(game_result, tuple):
                            game_frame, should_exit = game_result
                            if should_exit:
                                # Game wants to exit, return to menu
                                self.current_game.stop()
                                self.is_in_menu = True
                                self.current_game = None
                                self.human_detection_start = None
                                print("Game auto-exited due to no people detected")
                                # After exit, render menu below
                                layer_data['ui_elements'] = self._get_menu_ui_elements()
                                layer_data['bboxes'] = bboxes
                                final_frame = self.layer_manager.render_all(scaled_frame, layer_data)
                                
                                # Show and continue
                            else:
                                # Overlay BBox layer on top of game's frame unless game suppresses overlays
                                if hasattr(self.current_game, 'people_count'):
                                    self.current_game.people_count = min(len(bboxes), getattr(self.current_game, 'max_people', len(bboxes)))
                                if getattr(self.current_game, 'suppress_overlay', False):
                                    # Only render UI elements; hide bbox/contours/lines
                                    layer_data['ui_elements'] = []
                                    final_frame = game_frame
                                else:
                                    layer_data['bboxes'] = bboxes
                                    layer_data['face_contours'] = self._scale_contours_to_window(face_contours)
                                    scaled_pose = self._scale_lines_to_window(pose_lines)
                                    scaled_hands = self._scale_lines_to_window(hand_lines)
                                    layer_data['pose_lines'] = scaled_pose
                                    layer_data['hand_lines'] = scaled_hands
                                    try:
                                        pose_segments_count = sum(len(pl) for pl in scaled_pose)
                                        hand_segments_count = sum(len(hl) for hl in scaled_hands)
                                        print(f"[Render Debug] (game) pose_segments={pose_segments_count} hand_segments={hand_segments_count}")
                                    except Exception:
                                        pass
                                    final_frame = self.layer_manager.render_all(game_frame, layer_data)
                        else:
                            game_frame = game_result
                            # Overlay BBox layer on top of game's frame unless suppressed
                            if hasattr(self.current_game, 'people_count'):
                                self.current_game.people_count = min(len(bboxes), getattr(self.current_game, 'max_people', len(bboxes)))
                            if getattr(self.current_game, 'suppress_overlay', False):
                                layer_data['ui_elements'] = []
                                final_frame = game_frame
                            else:
                                layer_data['bboxes'] = bboxes
                                layer_data['face_contours'] = self._scale_contours_to_window(face_contours)
                                scaled_pose = self._scale_lines_to_window(pose_lines)
                                scaled_hands = self._scale_lines_to_window(hand_lines)
                                layer_data['pose_lines'] = scaled_pose
                                layer_data['hand_lines'] = scaled_hands
                                try:
                                    pose_segments_count = sum(len(pl) for pl in scaled_pose)
                                    hand_segments_count = sum(len(hl) for hl in scaled_hands)
                                    print(f"[Render Debug] (game2) pose_segments={pose_segments_count} hand_segments={hand_segments_count}")
                                except Exception:
                                    pass
                                final_frame = self.layer_manager.render_all(game_frame, layer_data)
                    else:
                        # Fallback to menu if no game
                        layer_data['ui_elements'] = self._get_menu_ui_elements()
                        layer_data['bboxes'] = bboxes
                        final_frame = self.layer_manager.render_all(scaled_frame, layer_data)
                
                # Display frame
                cv2.imshow(window_name, final_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    if not self.is_in_menu and self.current_game:
                        # Return to menu from game
                        self.current_game.stop()
                        self.is_in_menu = True
                        self.current_game = None
                        # Reset human detection timer when returning to menu
                        self.human_detection_start = None
                        print("Returned to main menu")
                    else:
                        # Exit application
                        break
                elif key == ord('s') and self.config.SHOW_STATISTICS_ON_S_KEY:
                    self._show_statistics()
                elif key == ord('m') and not self.is_in_menu:
                    # Return to menu
                    if self.current_game:
                        self.current_game.stop()
                    self.is_in_menu = True
                    self.current_game = None
                    # Reset human detection timer when returning to menu
                    self.human_detection_start = None
                    print("Returned to main menu")
        
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        
        # Stop threads
        self.thread_manager.stop()
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # No specific cleanup needed for HOG detector
        
        print("Cleanup complete")

    def _scale_contours_to_window(self, face_contours):
        """Scale face contour coordinates from camera space to window space using scaler."""
        if not face_contours:
            return []
        scaled = []
        for contour in face_contours:
            if not contour:
                scaled.append([])
                continue
            scaled_pts = []
            for (x, y) in contour:
                sx, sy = self.scaler.scale_coordinates(x, y)
                # Clamp to window bounds
                sx = max(0, min(sx, self.config.WINDOW_WIDTH - 1))
                sy = max(0, min(sy, self.config.WINDOW_HEIGHT - 1))
                scaled_pts.append((sx, sy))
            scaled.append(scaled_pts)
        return scaled

    def _scale_lines_to_window(self, lines_per_person):
        if not lines_per_person:
            return []
        scaled_all = []
        for lines in lines_per_person:
            scaled_lines = []
            for (x1, y1, x2, y2) in lines:
                sx1, sy1 = self.scaler.scale_coordinates(x1, y1)
                sx2, sy2 = self.scaler.scale_coordinates(x2, y2)
                sx1 = max(0, min(sx1, self.config.WINDOW_WIDTH - 1))
                sy1 = max(0, min(sy1, self.config.WINDOW_HEIGHT - 1))
                sx2 = max(0, min(sx2, self.config.WINDOW_WIDTH - 1))
                sy2 = max(0, min(sy2, self.config.WINDOW_HEIGHT - 1))
                scaled_lines.append((sx1, sy1, sx2, sy2))
            scaled_all.append(scaled_lines)
        return scaled_all

def main():
    """Main entry point."""
    app = VisionPlayApp()
    app.run()

if __name__ == "__main__":
    main()