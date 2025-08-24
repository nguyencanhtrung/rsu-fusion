"""
Trail Renderer for Tracking Visualization
Renders tracking trails and trajectory paths for vehicles
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Deque
from collections import deque, defaultdict
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger


class TrailRenderer:
    """Renders tracking trails and trajectory visualization"""
    
    def __init__(self, max_trail_length: int = 30):
        self.logger = get_logger("TrailRenderer")
        
        # Trail configuration
        self.max_trail_length = max_trail_length
        self.trail_storage: Dict[int, Deque[Tuple[int, int, float]]] = defaultdict(
            lambda: deque(maxlen=self.max_trail_length)
        )
        
        # Trail colors - cycling through distinct colors
        self.trail_colors = [
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
            (255, 165, 0),  # Orange
            (255, 192, 203), # Pink
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
            (128, 0, 128),  # Purple
        ]
        
        # Trail rendering settings
        self.trail_thickness = 2
        self.point_radius = 3
        self.fade_enabled = True
        self.min_alpha = 0.2
        self.max_alpha = 1.0
        
        # Performance tracking
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 30.0  # seconds
        
        self.logger.info(f"Trail renderer initialized with max length {max_trail_length}")
    
    def update_tracks(self, tracks: List[Dict[str, Any]]) -> None:
        """Update trail data with current track positions"""
        try:
            current_time = time.time()
            active_track_ids = set()
            
            for track in tracks:
                track_id = track.get('track_id', -1)
                if track_id < 0:
                    continue
                
                bbox = track.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                # Calculate center point
                x1, y1, x2, y2 = bbox
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Add point to trail with timestamp
                self.trail_storage[track_id].append((center_x, center_y, current_time))
                active_track_ids.add(track_id)
            
            # Periodic cleanup of old trails
            if current_time - self.last_cleanup_time > self.cleanup_interval:
                self._cleanup_old_trails(active_track_ids)
                self.last_cleanup_time = current_time
                
        except Exception as e:
            self.logger.error(f"Error updating track trails: {e}")
    
    def render_trails(self, image: np.ndarray, 
                     tracks: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        """Render all tracking trails on the image"""
        try:
            rendered_image = image.copy()
            
            if tracks is not None:
                self.update_tracks(tracks)
            
            # Render trails for all active tracks
            for track_id, trail_points in self.trail_storage.items():
                if len(trail_points) < 2:
                    continue
                
                self._render_single_trail(rendered_image, track_id, trail_points)
            
            return rendered_image
            
        except Exception as e:
            self.logger.error(f"Error rendering trails: {e}")
            return image
    
    def _render_single_trail(self, image: np.ndarray, track_id: int, 
                           trail_points: Deque[Tuple[int, int, float]]) -> None:
        """Render a single track's trail"""
        try:
            if len(trail_points) < 2:
                return
            
            # Get color for this track
            color = self._get_track_color(track_id)
            trail_points_list = list(trail_points)
            
            # Draw trail segments with fade effect
            for i in range(len(trail_points_list) - 1):
                pt1 = trail_points_list[i]
                pt2 = trail_points_list[i + 1]
                
                if self.fade_enabled:
                    # Calculate alpha based on position in trail
                    alpha = self._calculate_fade_alpha(i, len(trail_points_list))
                    faded_color = self._apply_alpha(color, alpha)
                else:
                    faded_color = color
                
                # Draw line segment
                cv2.line(image, 
                        (pt1[0], pt1[1]), 
                        (pt2[0], pt2[1]), 
                        faded_color, 
                        self.trail_thickness)
            
            # Draw trail points
            for i, (x, y, timestamp) in enumerate(trail_points_list):
                if self.fade_enabled:
                    alpha = self._calculate_fade_alpha(i, len(trail_points_list))
                    point_color = self._apply_alpha(color, alpha)
                else:
                    point_color = color
                
                # Draw point
                cv2.circle(image, (x, y), self.point_radius, point_color, -1)
                
                # Draw outline for better visibility
                cv2.circle(image, (x, y), self.point_radius + 1, (0, 0, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Error rendering single trail {track_id}: {e}")
    
    def render_trail_with_prediction(self, image: np.ndarray, 
                                   track_id: int, 
                                   prediction_steps: int = 5) -> np.ndarray:
        """Render trail with predicted future positions"""
        try:
            rendered_image = image.copy()
            
            if track_id not in self.trail_storage:
                return rendered_image
            
            trail_points = list(self.trail_storage[track_id])
            if len(trail_points) < 3:
                return rendered_image
            
            # Render existing trail
            self._render_single_trail(rendered_image, track_id, 
                                    deque(trail_points, maxlen=self.max_trail_length))
            
            # Calculate and render prediction
            predicted_points = self._calculate_prediction(trail_points, prediction_steps)
            if predicted_points:
                color = self._get_track_color(track_id)
                prediction_color = self._apply_alpha(color, 0.5)  # Semi-transparent
                
                # Draw prediction line
                last_real_point = trail_points[-1]
                prev_point = (last_real_point[0], last_real_point[1])
                
                for pred_x, pred_y in predicted_points:
                    cv2.line(rendered_image, prev_point, (pred_x, pred_y), 
                            prediction_color, max(1, self.trail_thickness // 2))
                    cv2.circle(rendered_image, (pred_x, pred_y), 2, prediction_color, -1)
                    prev_point = (pred_x, pred_y)
            
            return rendered_image
            
        except Exception as e:
            self.logger.error(f"Error rendering prediction trail: {e}")
            return image
    
    def get_trail_statistics(self) -> Dict[str, Any]:
        """Get statistics about current trails"""
        try:
            stats = {
                'active_trails': len(self.trail_storage),
                'total_trail_points': sum(len(trail) for trail in self.trail_storage.values()),
                'avg_trail_length': 0.0,
                'longest_trail': 0,
                'shortest_trail': 0
            }
            
            if self.trail_storage:
                trail_lengths = [len(trail) for trail in self.trail_storage.values()]
                stats['avg_trail_length'] = np.mean(trail_lengths)
                stats['longest_trail'] = max(trail_lengths)
                stats['shortest_trail'] = min(trail_lengths)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting trail statistics: {e}")
            return {'active_trails': 0}
    
    def clear_trail(self, track_id: int) -> None:
        """Clear trail for specific track ID"""
        if track_id in self.trail_storage:
            self.trail_storage[track_id].clear()
            self.logger.debug(f"Cleared trail for track {track_id}")
    
    def clear_all_trails(self) -> None:
        """Clear all trails"""
        self.trail_storage.clear()
        self.logger.debug("Cleared all trails")
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for track ID"""
        color_index = (track_id - 1) % len(self.trail_colors)
        return self.trail_colors[color_index]
    
    def _calculate_fade_alpha(self, point_index: int, total_points: int) -> float:
        """Calculate fade alpha for trail point based on age"""
        if total_points <= 1:
            return self.max_alpha
        
        # Newer points (higher index) have higher alpha
        alpha_ratio = point_index / (total_points - 1)
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * alpha_ratio
        return max(self.min_alpha, min(self.max_alpha, alpha))
    
    def _apply_alpha(self, color: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
        """Apply alpha to color (simple fade to black)"""
        return tuple(int(c * alpha) for c in color)
    
    def _calculate_prediction(self, trail_points: List[Tuple[int, int, float]], 
                            steps: int) -> List[Tuple[int, int]]:
        """Calculate predicted future positions based on trail history"""
        try:
            if len(trail_points) < 3:
                return []
            
            # Use last 3 points to estimate velocity
            recent_points = trail_points[-3:]
            
            # Calculate average velocity
            velocities = []
            for i in range(1, len(recent_points)):
                dt = recent_points[i][2] - recent_points[i-1][2]  # time difference
                if dt > 0:
                    dx = recent_points[i][0] - recent_points[i-1][0]
                    dy = recent_points[i][1] - recent_points[i-1][1]
                    velocities.append((dx / dt, dy / dt))
            
            if not velocities:
                return []
            
            # Average velocity
            avg_vx = np.mean([v[0] for v in velocities])
            avg_vy = np.mean([v[1] for v in velocities])
            
            # Generate predicted points
            predicted_points = []
            last_point = trail_points[-1]
            
            for step in range(1, steps + 1):
                # Assume constant time step
                dt = 0.1  # 100ms steps
                pred_x = int(last_point[0] + avg_vx * dt * step)
                pred_y = int(last_point[1] + avg_vy * dt * step)
                predicted_points.append((pred_x, pred_y))
            
            return predicted_points
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction: {e}")
            return []
    
    def _cleanup_old_trails(self, active_track_ids: set) -> None:
        """Remove trails for tracks that are no longer active"""
        try:
            trails_to_remove = []
            current_time = time.time()
            
            for track_id, trail in self.trail_storage.items():
                # Remove if track is not active and trail is old
                if track_id not in active_track_ids:
                    if not trail or (current_time - trail[-1][2]) > 5.0:  # 5 seconds old
                        trails_to_remove.append(track_id)
            
            for track_id in trails_to_remove:
                del self.trail_storage[track_id]
                self.logger.debug(f"Cleaned up old trail for track {track_id}")
                
        except Exception as e:
            self.logger.error(f"Error during trail cleanup: {e}")
    
    def update_settings(self, settings: Dict[str, Any]) -> None:
        """Update trail rendering settings"""
        try:
            if 'max_trail_length' in settings:
                self.max_trail_length = settings['max_trail_length']
                # Update existing deques
                for track_id in list(self.trail_storage.keys()):
                    old_trail = list(self.trail_storage[track_id])
                    self.trail_storage[track_id] = deque(
                        old_trail[-self.max_trail_length:], 
                        maxlen=self.max_trail_length
                    )
            
            if 'trail_thickness' in settings:
                self.trail_thickness = settings['trail_thickness']
            
            if 'fade_enabled' in settings:
                self.fade_enabled = settings['fade_enabled']
            
            if 'point_radius' in settings:
                self.point_radius = settings['point_radius']
            
            self.logger.info(f"Updated trail settings: {settings}")
            
        except Exception as e:
            self.logger.error(f"Error updating trail settings: {e}")