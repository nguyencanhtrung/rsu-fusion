"""
Geometry Utilities
Mathematical helpers for geometric calculations and transformations
"""

import numpy as np
import math
from typing import Tuple, List, Optional, Dict, Any, Union
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger


class GeometryUtils:
    """Utility class for geometric calculations"""
    
    def __init__(self):
        self.logger = get_logger("GeometryUtils")
    
    @staticmethod
    def euclidean_distance(point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two 2D points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def euclidean_distance_3d(point1: Tuple[float, float, float], 
                             point2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two 3D points"""
        return math.sqrt(
            (point1[0] - point2[0])**2 + 
            (point1[1] - point2[1])**2 + 
            (point1[2] - point2[2])**2
        )
    
    @staticmethod
    def manhattan_distance(point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """Calculate Manhattan distance between two 2D points"""
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
    
    @staticmethod
    def angle_between_vectors(v1: Tuple[float, float], 
                             v2: Tuple[float, float]) -> float:
        """Calculate angle between two 2D vectors in radians"""
        try:
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
            magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if magnitude_v1 == 0 or magnitude_v2 == 0:
                return 0.0
            
            cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
            # Clamp to avoid numerical errors
            cos_angle = max(-1.0, min(1.0, cos_angle))
            
            return math.acos(cos_angle)
        except:
            return 0.0
    
    @staticmethod
    def angle_between_points(center: Tuple[float, float], 
                           point1: Tuple[float, float], 
                           point2: Tuple[float, float]) -> float:
        """Calculate angle between two points relative to a center point"""
        v1 = (point1[0] - center[0], point1[1] - center[1])
        v2 = (point2[0] - center[0], point2[1] - center[1])
        return GeometryUtils.angle_between_vectors(v1, v2)
    
    @staticmethod
    def rotate_point_2d(point: Tuple[float, float], 
                       angle: float, 
                       center: Tuple[float, float] = (0, 0)) -> Tuple[float, float]:
        """Rotate a 2D point around a center by given angle (radians)"""
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        
        # Translate to origin
        translated_x = point[0] - center[0]
        translated_y = point[1] - center[1]
        
        # Rotate
        rotated_x = translated_x * cos_angle - translated_y * sin_angle
        rotated_y = translated_x * sin_angle + translated_y * cos_angle
        
        # Translate back
        final_x = rotated_x + center[0]
        final_y = rotated_y + center[1]
        
        return (final_x, final_y)
    
    @staticmethod
    def point_in_polygon(point: Tuple[float, float], 
                        polygon: List[Tuple[float, float]]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm"""
        if len(polygon) < 3:
            return False
        
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    @staticmethod
    def point_to_line_distance(point: Tuple[float, float], 
                              line_start: Tuple[float, float], 
                              line_end: Tuple[float, float]) -> float:
        """Calculate shortest distance from point to line segment"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Vector from line_start to line_end
        line_vec = (x2 - x1, y2 - y1)
        line_length_squared = line_vec[0]**2 + line_vec[1]**2
        
        if line_length_squared == 0:
            # Line is actually a point
            return GeometryUtils.euclidean_distance(point, line_start)
        
        # Vector from line_start to point
        point_vec = (x0 - x1, y0 - y1)
        
        # Project point onto line
        t = (point_vec[0] * line_vec[0] + point_vec[1] * line_vec[1]) / line_length_squared
        
        if t < 0:
            # Closest point is line_start
            return GeometryUtils.euclidean_distance(point, line_start)
        elif t > 1:
            # Closest point is line_end
            return GeometryUtils.euclidean_distance(point, line_end)
        else:
            # Closest point is on the line segment
            closest_point = (x1 + t * line_vec[0], y1 + t * line_vec[1])
            return GeometryUtils.euclidean_distance(point, closest_point)
    
    @staticmethod
    def calculate_polygon_area(polygon: List[Tuple[float, float]]) -> float:
        """Calculate area of a polygon using shoelace formula"""
        if len(polygon) < 3:
            return 0.0
        
        area = 0.0
        n = len(polygon)
        
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        
        return abs(area) / 2.0
    
    @staticmethod
    def calculate_polygon_centroid(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate centroid of a polygon"""
        if not polygon:
            return (0.0, 0.0)
        
        if len(polygon) == 1:
            return polygon[0]
        
        area = GeometryUtils.calculate_polygon_area(polygon)
        if area == 0:
            # Degenerate polygon, return average of vertices
            avg_x = sum(p[0] for p in polygon) / len(polygon)
            avg_y = sum(p[1] for p in polygon) / len(polygon)
            return (avg_x, avg_y)
        
        cx = 0.0
        cy = 0.0
        n = len(polygon)
        
        for i in range(n):
            j = (i + 1) % n
            cross = polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1]
            cx += (polygon[i][0] + polygon[j][0]) * cross
            cy += (polygon[i][1] + polygon[j][1]) * cross
        
        factor = 1.0 / (6.0 * area)
        cx *= factor
        cy *= factor
        
        return (cx, cy)
    
    @staticmethod
    def line_intersection(line1_start: Tuple[float, float], 
                         line1_end: Tuple[float, float],
                         line2_start: Tuple[float, float], 
                         line2_end: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Find intersection point of two line segments"""
        x1, y1 = line1_start
        x2, y2 = line1_end
        x3, y3 = line2_start
        x4, y4 = line2_end
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            # Lines are parallel
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Check if intersection is within both line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return (intersection_x, intersection_y)
        
        return None
    
    @staticmethod
    def create_bounding_box(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """Create axis-aligned bounding box for a set of points"""
        if not points:
            return (0.0, 0.0, 0.0, 0.0)
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        return (min_x, min_y, max_x, max_y)
    
    @staticmethod
    def expand_bounding_box(bbox: Tuple[float, float, float, float], 
                          expansion: float) -> Tuple[float, float, float, float]:
        """Expand bounding box by given amount in all directions"""
        min_x, min_y, max_x, max_y = bbox
        return (
            min_x - expansion,
            min_y - expansion,
            max_x + expansion,
            max_y + expansion
        )
    
    @staticmethod
    def bounding_box_intersection(bbox1: Tuple[float, float, float, float],
                                bbox2: Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float, float]]:
        """Calculate intersection of two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection bounds
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        # Check if there's an intersection
        if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
            return (inter_x_min, inter_y_min, inter_x_max, inter_y_max)
        
        return None
    
    @staticmethod
    def bounding_box_union(bbox1: Tuple[float, float, float, float],
                          bbox2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Calculate union of two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        return (
            min(x1_min, x2_min),
            min(y1_min, y2_min),
            max(x1_max, x2_max),
            max(y1_max, y2_max)
        )
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-π, π] range"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    @staticmethod
    def angle_difference(angle1: float, angle2: float) -> float:
        """Calculate smallest difference between two angles"""
        diff = angle2 - angle1
        return GeometryUtils.normalize_angle(diff)
    
    @staticmethod
    def interpolate_points(point1: Tuple[float, float], 
                          point2: Tuple[float, float], 
                          t: float) -> Tuple[float, float]:
        """Linearly interpolate between two points (t=0 gives point1, t=1 gives point2)"""
        x = point1[0] + t * (point2[0] - point1[0])
        y = point1[1] + t * (point2[1] - point1[1])
        return (x, y)
    
    @staticmethod
    def create_circle_points(center: Tuple[float, float], 
                           radius: float, 
                           num_points: int = 32) -> List[Tuple[float, float]]:
        """Generate points on a circle"""
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        return points
    
    @staticmethod
    def smooth_path(path: List[Tuple[float, float]], 
                   window_size: int = 3) -> List[Tuple[float, float]]:
        """Smooth a path using moving average"""
        if len(path) <= window_size:
            return path.copy()
        
        smoothed_path = []
        half_window = window_size // 2
        
        for i in range(len(path)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(path), i + half_window + 1)
            
            avg_x = sum(p[0] for p in path[start_idx:end_idx]) / (end_idx - start_idx)
            avg_y = sum(p[1] for p in path[start_idx:end_idx]) / (end_idx - start_idx)
            
            smoothed_path.append((avg_x, avg_y))
        
        return smoothed_path
    
    @staticmethod
    def calculate_path_length(path: List[Tuple[float, float]]) -> float:
        """Calculate total length of a path"""
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(path)):
            total_length += GeometryUtils.euclidean_distance(path[i-1], path[i])
        
        return total_length
    
    @staticmethod
    def resample_path(path: List[Tuple[float, float]], 
                     target_spacing: float) -> List[Tuple[float, float]]:
        """Resample path to have approximately uniform spacing"""
        if len(path) < 2:
            return path.copy()
        
        resampled_path = [path[0]]
        current_distance = 0.0
        
        for i in range(1, len(path)):
            segment_length = GeometryUtils.euclidean_distance(path[i-1], path[i])
            
            if current_distance + segment_length >= target_spacing:
                # Need to add point(s) in this segment
                remaining_distance = target_spacing - current_distance
                
                while remaining_distance <= segment_length:
                    t = remaining_distance / segment_length
                    new_point = GeometryUtils.interpolate_points(path[i-1], path[i], t)
                    resampled_path.append(new_point)
                    
                    segment_length -= remaining_distance
                    remaining_distance = target_spacing
                
                current_distance = segment_length
            else:
                current_distance += segment_length
        
        # Always include the last point
        if len(resampled_path) == 0 or resampled_path[-1] != path[-1]:
            resampled_path.append(path[-1])
        
        return resampled_path