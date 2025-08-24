"""
Overlay Renderer for Detection and Tracking Visualization
Renders detection boxes, tracking information, and metadata overlays
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger


class OverlayRenderer:
    """Renders various overlays for detection and tracking visualization"""
    
    def __init__(self):
        self.logger = get_logger("OverlayRenderer")
        
        # Color schemes for different elements
        self.colors = {
            # Detection colors
            'detection_box': (0, 255, 0),        # Green
            'detection_text': (255, 255, 255),   # White
            
            # Tracking colors (will be dynamically assigned)
            'track_box': (255, 0, 0),            # Blue default
            'track_text': (255, 255, 255),       # White
            'track_id': (0, 255, 255),           # Yellow
            
            # Special vehicle colors
            'ambulance': (0, 0, 255),            # Red
            'emergency': (255, 0, 255),          # Magenta
            
            # UI colors
            'background': (0, 0, 0),             # Black
            'border': (128, 128, 128),           # Gray
            'info_text': (255, 255, 255),        # White
        }
        
        # Track color assignment
        self.track_colors = {}
        self.color_palette = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 255),  # Purple
            (255, 128, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (128, 255, 0),  # Lime
        ]
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.large_font_scale = 0.7
        self.large_font_thickness = 2
        
        self.logger.info("Overlay renderer initialized")
    
    def render_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Render detection bounding boxes and labels"""
        try:
            rendered_image = image.copy()
            
            for detection in detections:
                self._draw_detection_box(rendered_image, detection)
            
            return rendered_image
            
        except Exception as e:
            self.logger.error(f"Error rendering detections: {e}")
            return image
    
    def render_tracks(self, image: np.ndarray, tracks: List[Dict[str, Any]]) -> np.ndarray:
        """Render tracking bounding boxes and track IDs"""
        try:
            rendered_image = image.copy()
            
            for track in tracks:
                self._draw_track_box(rendered_image, track)
            
            return rendered_image
            
        except Exception as e:
            self.logger.error(f"Error rendering tracks: {e}")
            return image
    
    def render_combined(self, image: np.ndarray, 
                       detections: List[Dict[str, Any]], 
                       tracks: List[Dict[str, Any]]) -> np.ndarray:
        """Render both detections and tracks with proper layering"""
        try:
            rendered_image = image.copy()
            
            # First render detections (background layer)
            for detection in detections:
                self._draw_detection_box(rendered_image, detection, thickness=1)
            
            # Then render tracks (foreground layer)
            for track in tracks:
                self._draw_track_box(rendered_image, track, thickness=2)
            
            return rendered_image
            
        except Exception as e:
            self.logger.error(f"Error rendering combined overlay: {e}")
            return image
    
    def render_info_panel(self, image: np.ndarray, 
                         info_data: Dict[str, Any],
                         position: Tuple[int, int] = (10, 10)) -> np.ndarray:
        """Render information panel with system stats"""
        try:
            rendered_image = image.copy()
            
            x, y = position
            line_height = 25
            panel_width = 200
            
            # Prepare info lines
            info_lines = []
            
            # FPS information
            if 'fps' in info_data:
                info_lines.append(f"FPS: {info_data['fps']:.1f}")
            
            # Detection count
            if 'detection_count' in info_data:
                info_lines.append(f"Detections: {info_data['detection_count']}")
            
            # Tracking count  
            if 'track_count' in info_data:
                info_lines.append(f"Tracks: {info_data['track_count']}")
            
            # Confirmed tracks
            if 'confirmed_tracks' in info_data:
                info_lines.append(f"Confirmed: {info_data['confirmed_tracks']}")
            
            # Timestamp
            if 'timestamp' in info_data:
                info_lines.append(f"Time: {info_data['timestamp']}")
            
            # Emergency vehicles
            if 'emergency_count' in info_data and info_data['emergency_count'] > 0:
                info_lines.append(f"Emergency: {info_data['emergency_count']}")
            
            # Draw panel background
            panel_height = len(info_lines) * line_height + 10
            cv2.rectangle(rendered_image, 
                        (x - 5, y - 5), 
                        (x + panel_width, y + panel_height), 
                        self.colors['background'], -1)
            
            cv2.rectangle(rendered_image, 
                        (x - 5, y - 5), 
                        (x + panel_width, y + panel_height), 
                        self.colors['border'], 1)
            
            # Draw info lines
            for i, line in enumerate(info_lines):
                text_y = y + (i + 1) * line_height
                cv2.putText(rendered_image, line, (x, text_y),
                           self.font, self.font_scale, self.colors['info_text'], 
                           self.font_thickness)
            
            return rendered_image
            
        except Exception as e:
            self.logger.error(f"Error rendering info panel: {e}")
            return image
    
    def render_vehicle_labels(self, image: np.ndarray, 
                            vehicles: List[Dict[str, Any]]) -> np.ndarray:
        """Render detailed vehicle labels with classification info"""
        try:
            rendered_image = image.copy()
            
            for vehicle in vehicles:
                bbox = vehicle.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Prepare label text
                class_name = vehicle.get('class_name', 'unknown')
                confidence = vehicle.get('confidence', 0.0)
                track_id = vehicle.get('track_id', None)
                
                label_parts = []
                label_parts.append(class_name.title())
                
                if confidence > 0:
                    label_parts.append(f"{confidence:.2f}")
                
                if track_id is not None:
                    label_parts.append(f"ID:{track_id}")
                
                label = " | ".join(label_parts)
                
                # Get text size for background
                (text_width, text_height), _ = cv2.getTextSize(
                    label, self.font, self.font_scale, self.font_thickness)
                
                # Choose color based on vehicle type
                color = self._get_vehicle_color(vehicle)
                
                # Draw label background
                cv2.rectangle(rendered_image,
                            (x1, y1 - text_height - 10),
                            (x1 + text_width + 5, y1),
                            color, -1)
                
                # Draw label text
                cv2.putText(rendered_image, label, (x1 + 2, y1 - 5),
                           self.font, self.font_scale, self.colors['detection_text'],
                           self.font_thickness)
            
            return rendered_image
            
        except Exception as e:
            self.logger.error(f"Error rendering vehicle labels: {e}")
            return image
    
    def render_track_trajectories(self, image: np.ndarray, 
                                tracks: List[Dict[str, Any]],
                                max_points: int = 30) -> np.ndarray:
        """Render track trajectory points (used by trail renderer)"""
        try:
            rendered_image = image.copy()
            
            for track in tracks:
                track_id = track.get('track_id', -1)
                history = track.get('history', [])
                
                if len(history) < 2:
                    continue
                
                # Get track color
                color = self._get_track_color(track_id)
                
                # Limit history points
                recent_history = history[-max_points:]
                
                # Draw trajectory points
                for i, point in enumerate(recent_history):
                    if len(point) >= 2:
                        x, y = int(point[0]), int(point[1])
                        
                        # Fade effect - newer points are brighter
                        alpha = (i + 1) / len(recent_history)
                        point_color = tuple(int(c * alpha) for c in color)
                        
                        # Draw small circle for each point
                        cv2.circle(rendered_image, (x, y), 2, point_color, -1)
            
            return rendered_image
            
        except Exception as e:
            self.logger.error(f"Error rendering track trajectories: {e}")
            return image
    
    def render_region_of_interest(self, image: np.ndarray, 
                                roi_bounds: Tuple[int, int, int, int]) -> np.ndarray:
        """Render region of interest overlay"""
        try:
            rendered_image = image.copy()
            
            x1, y1, x2, y2 = roi_bounds
            
            # Draw ROI rectangle
            cv2.rectangle(rendered_image, (x1, y1), (x2, y2), 
                        self.colors['border'], 2)
            
            # Add ROI label
            label = "ROI"
            cv2.putText(rendered_image, label, (x1 + 5, y1 + 20),
                       self.font, self.font_scale, self.colors['info_text'],
                       self.font_thickness)
            
            return rendered_image
            
        except Exception as e:
            self.logger.error(f"Error rendering ROI: {e}")
            return image
    
    def _draw_detection_box(self, image: np.ndarray, detection: Dict[str, Any], 
                           thickness: int = 2):
        """Draw single detection bounding box"""
        try:
            bbox = detection.get('bbox', [])
            if len(bbox) != 4:
                return
            
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get color based on vehicle type
            color = self._get_vehicle_color(detection)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence if available
            confidence = detection.get('confidence', 0.0)
            if confidence > 0:
                conf_text = f"{confidence:.2f}"
                cv2.putText(image, conf_text, (x1, y2 + 15),
                           self.font, self.font_scale, color, self.font_thickness)
            
        except Exception as e:
            self.logger.error(f"Error drawing detection box: {e}")
    
    def _draw_track_box(self, image: np.ndarray, track: Dict[str, Any], 
                       thickness: int = 2):
        """Draw single track bounding box with ID"""
        try:
            bbox = track.get('bbox', [])
            if len(bbox) != 4:
                return
            
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            track_id = track.get('track_id', -1)
            
            # Get track-specific color
            color = self._get_track_color(track_id)
            
            # Draw thicker bounding box for tracks
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw track ID
            id_text = f"ID:{track_id}"
            cv2.putText(image, id_text, (x1, y2 + 15),
                       self.font, self.font_scale, self.colors['track_id'], 
                       self.font_thickness)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(image, (center_x, center_y), 3, color, -1)
            
        except Exception as e:
            self.logger.error(f"Error drawing track box: {e}")
    
    def _get_vehicle_color(self, vehicle: Dict[str, Any]) -> Tuple[int, int, int]:
        """Get color for vehicle based on type"""
        class_name = vehicle.get('class_name', '').lower()
        
        if 'ambulance' in class_name or vehicle.get('is_emergency_vehicle', False):
            return self.colors['ambulance']
        elif 'emergency' in class_name:
            return self.colors['emergency']
        else:
            return self.colors['detection_box']
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for track ID"""
        if track_id not in self.track_colors:
            # Assign color based on track ID
            color_index = (track_id - 1) % len(self.color_palette)
            self.track_colors[track_id] = self.color_palette[color_index]
        
        return self.track_colors[track_id]
    
    def create_legend(self, image: np.ndarray, 
                     position: Tuple[int, int] = (10, 200)) -> np.ndarray:
        """Create color legend for different vehicle types"""
        try:
            rendered_image = image.copy()
            
            x, y = position
            line_height = 20
            box_size = 15
            
            legend_items = [
                ("Detection", self.colors['detection_box']),
                ("Track", self.colors['track_box']),
                ("Ambulance", self.colors['ambulance']),
                ("Emergency", self.colors['emergency'])
            ]
            
            # Draw legend background
            legend_width = 150
            legend_height = len(legend_items) * line_height + 10
            
            cv2.rectangle(rendered_image,
                        (x - 5, y - 5),
                        (x + legend_width, y + legend_height),
                        self.colors['background'], -1)
            
            cv2.rectangle(rendered_image,
                        (x - 5, y - 5),
                        (x + legend_width, y + legend_height),
                        self.colors['border'], 1)
            
            # Draw legend items
            for i, (label, color) in enumerate(legend_items):
                item_y = y + i * line_height + 10
                
                # Draw color box
                cv2.rectangle(rendered_image,
                            (x, item_y - box_size//2),
                            (x + box_size, item_y + box_size//2),
                            color, -1)
                
                # Draw label
                cv2.putText(rendered_image, label, (x + box_size + 5, item_y + 3),
                           self.font, self.font_scale, self.colors['info_text'],
                           self.font_thickness)
            
            return rendered_image
            
        except Exception as e:
            self.logger.error(f"Error creating legend: {e}")
            return image
    
    def clear_track_colors(self):
        """Clear track color assignments (useful for reset)"""
        self.track_colors.clear()
        self.logger.debug("Track color assignments cleared")
    
    def update_colors(self, new_colors: Dict[str, Tuple[int, int, int]]):
        """Update color scheme"""
        try:
            self.colors.update(new_colors)
            self.logger.info(f"Updated {len(new_colors)} color settings")
        except Exception as e:
            self.logger.error(f"Error updating colors: {e}")
    
    def get_rendering_stats(self) -> Dict[str, Any]:
        """Get rendering statistics"""
        return {
            'assigned_track_colors': len(self.track_colors),
            'available_colors': len(self.color_palette),
            'color_scheme': len(self.colors)
        }