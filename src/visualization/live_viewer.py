"""
Live Visualization System for RSU Fusion
Real-time display of camera feed with detection boxes and tracking trails
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
import threading
from collections import deque
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger


class LiveViewer:
    """Live visualization for detection and tracking results"""
    
    def __init__(self, window_name: str = "RSU Fusion - Phase 1"):
        self.logger = get_logger("LiveViewer")
        self.window_name = window_name
        
        # Display configuration
        self.display_width = 1280
        self.display_height = 720
        self.fps_target = 30.0
        
        # Visualization state
        self.running = False
        self.current_image = None
        self.current_detections = []
        self.current_tracks = []
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Thread safety
        self.display_lock = threading.Lock()
        
        # Colors for visualization (BGR format for OpenCV)
        self.colors = {
            'detection_box': (0, 255, 0),      # Green for detections
            'track_box': (255, 0, 0),          # Blue for tracks
            'track_trail': (0, 255, 255),      # Yellow for trails
            'ambulance': (0, 0, 255),          # Red for ambulance
            'text': (255, 255, 255),           # White for text
            'background': (0, 0, 0)            # Black for background
        }
        
        self.logger.info("Live viewer initialized")
    
    def start(self):
        """Start the live viewer"""
        self.running = True
        
        # Create OpenCV window
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        except:
            # Fallback for older OpenCV versions
            cv2.namedWindow(self.window_name)
            try:
                cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
            except:
                pass  # Window will use default size
        
        self.logger.info("✅ Live viewer started")
    
    def update_image(self, image: np.ndarray):
        """Update the base camera image"""
        with self.display_lock:
            # Resize image to display size
            self.current_image = cv2.resize(image, (self.display_width, self.display_height))
    
    def update_detections(self, detections: List[Dict[str, Any]]):
        """Update detection results"""
        with self.display_lock:
            self.current_detections = detections.copy()
    
    def update_tracks(self, tracks: List[Dict[str, Any]]):
        """Update tracking results"""
        with self.display_lock:
            self.current_tracks = tracks.copy()
    
    def render_frame(self) -> Optional[np.ndarray]:
        """Render current frame with overlays"""
        # Copy data quickly while holding lock, then release it
        with self.display_lock:
            if self.current_image is None:
                return None
            
            # Quickly copy all data we need
            frame = self.current_image.copy()
            detections = self.current_detections.copy()
            tracks = self.current_tracks.copy()
        
        # Now draw overlays WITHOUT holding the lock
        # This allows new images/data to be updated while we draw
        
        # Draw tracking trails first (background layer)  
        self._draw_tracking_trails(frame, tracks)
        
        # Draw detection boxes
        self._draw_detections(frame, detections)
        
        # Draw tracking boxes (on top of detections)
        self._draw_tracks(frame, tracks)
        
        # Draw performance info
        self._draw_performance_info(frame)
        
        return frame
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]):
        """Draw detection bounding boxes"""
        for detection in detections:
            try:
                # Get detection info
                bbox = detection.get('bbox', [])
                confidence = detection.get('confidence', 0.0)
                class_name = detection.get('class_name', 'unknown')
                
                if len(bbox) != 4:
                    continue
                
                # Scale bbox to display size
                x1, y1, x2, y2 = self._scale_bbox_to_display(bbox)
                
                # Choose color based on class
                color = self.colors['ambulance'] if 'ambulance' in class_name.lower() else self.colors['detection_box']
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Background for label
                cv2.rectangle(frame, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            color, -1)
                
                # Label text
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 2)
                
            except Exception as e:
                self.logger.error(f"Error drawing detection: {e}")
    
    def _draw_tracks(self, frame: np.ndarray, tracks: List[Dict[str, Any]]):
        """Draw tracking results"""
        for track in tracks:
            try:
                # Get track info
                track_id = track.get('track_id', 0)
                bbox = track.get('bbox', [])
                class_name = track.get('class_name', 'unknown')
                confidence = track.get('confidence', 0.0)
                
                if len(bbox) != 4:
                    continue
                
                # Scale bbox to display size
                x1, y1, x2, y2 = self._scale_bbox_to_display(bbox)
                
                # Choose color based on class
                color = self.colors['ambulance'] if 'ambulance' in class_name.lower() else self.colors['track_box']
                
                # Draw tracking box (thicker than detection)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw track ID
                track_label = f"ID:{track_id}"
                cv2.putText(frame, track_label, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 3, color, -1)
                
            except Exception as e:
                self.logger.error(f"Error drawing track: {e}")
    
    def _draw_tracking_trails(self, frame: np.ndarray, tracks: List[Dict[str, Any]]):
        """Draw tracking history trails"""
        for track in tracks:
            try:
                history = track.get('history', [])
                
                if len(history) < 2:
                    continue
                
                # Convert history to display coordinates
                trail_points = []
                for point in history:
                    if len(point) >= 2:
                        # Scale point to display size
                        x = int(point[0] * self.display_width / 1920)  # Assuming original 1920 width
                        y = int(point[1] * self.display_height / 1080)  # Assuming original 1080 height
                        trail_points.append((x, y))
                
                # Draw trail as connected lines
                if len(trail_points) >= 2:
                    for i in range(1, len(trail_points)):
                        # Fade alpha based on position in history
                        alpha = max(0.3, (i / len(trail_points)))
                        
                        cv2.line(frame, trail_points[i-1], trail_points[i], 
                                self.colors['track_trail'], 2)
                        
                        # Draw small circles at trail points
                        cv2.circle(frame, trail_points[i], 2, self.colors['track_trail'], -1)
                
            except Exception as e:
                self.logger.error(f"Error drawing trail: {e}")
    
    def _draw_performance_info(self, frame: np.ndarray):
        """Draw performance information overlay"""
        try:
            # Calculate current FPS
            current_time = time.time()
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
            self.last_frame_time = current_time
            
            if len(self.frame_times) > 0:
                avg_frame_time = np.mean(self.frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            else:
                fps = 0
            
            # Performance stats
            info_lines = [
                f"FPS: {fps:.1f}",
                f"Detections: {len(self.current_detections)}",
                f"Tracks: {len(self.current_tracks)}",
                f"Time: {time.strftime('%H:%M:%S')}"
            ]
            
            # Draw info background
            info_height = len(info_lines) * 25 + 10
            cv2.rectangle(frame, (10, 10), (200, info_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (200, info_height), (255, 255, 255), 1)
            
            # Draw info text
            for i, line in enumerate(info_lines):
                y_pos = 30 + i * 25
                cv2.putText(frame, line, (15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            
        except Exception as e:
            self.logger.error(f"Error drawing performance info: {e}")
    
    def _scale_bbox_to_display(self, bbox: List[float]) -> Tuple[int, int, int, int]:
        """Scale bounding box coordinates to display size"""
        x1, y1, x2, y2 = bbox
        
        # EXACT reference camera image size: 960x540 (VIEW_WIDTH/HEIGHT in reference)
        original_width = 960
        original_height = 540
        
        # Scale to display size
        x1_scaled = int(x1 * self.display_width / original_width)
        y1_scaled = int(y1 * self.display_height / original_height)
        x2_scaled = int(x2 * self.display_width / original_width)
        y2_scaled = int(y2 * self.display_height / original_height)
        
        return x1_scaled, y1_scaled, x2_scaled, y2_scaled
    
    def show_frame(self):
        """Display current frame"""
        if not self.running:
            return False
        
        frame = self.render_frame()
        if frame is not None:
            cv2.imshow(self.window_name, frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            return False
        elif key == ord('r'):  # 'r' to reset
            self.logger.info("Reset requested via keyboard")
        
        return True
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get viewer performance statistics"""
        if len(self.frame_times) > 0:
            avg_frame_time = np.mean(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            avg_frame_time = 0
            fps = 0
        
        return {
            'avg_fps': fps,
            'avg_frame_time_ms': avg_frame_time * 1000,
            'detections_count': len(self.current_detections),
            'tracks_count': len(self.current_tracks)
        }
    
    def stop(self):
        """Stop the live viewer"""
        self.running = False
        cv2.destroyWindow(self.window_name)
        self.logger.info("Live viewer stopped")
    
    def cleanup(self):
        """Clean up viewer resources"""
        self.logger.info("Cleaning up live viewer...")
        self.stop()
        cv2.destroyAllWindows()
