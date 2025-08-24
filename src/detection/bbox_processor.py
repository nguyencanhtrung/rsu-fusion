"""
Bounding Box Processing Utilities
Handles bounding box operations, transformations, and utilities
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger


class BBoxProcessor:
    """Utility class for bounding box operations and transformations"""
    
    def __init__(self):
        self.logger = get_logger("BBoxProcessor")
    
    def normalize_bbox(self, bbox: List[float], image_width: int, image_height: int) -> List[float]:
        """Normalize bounding box coordinates to [0,1] range"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Normalize to image dimensions
            norm_bbox = [
                x1 / image_width,
                y1 / image_height,
                x2 / image_width,
                y2 / image_height
            ]
            
            # Clamp to [0,1] range
            norm_bbox = [max(0.0, min(1.0, coord)) for coord in norm_bbox]
            
            return norm_bbox
            
        except Exception as e:
            self.logger.error(f"Error normalizing bbox: {e}")
            return bbox
    
    def denormalize_bbox(self, norm_bbox: List[float], image_width: int, image_height: int) -> List[float]:
        """Convert normalized bounding box back to pixel coordinates"""
        try:
            x1_norm, y1_norm, x2_norm, y2_norm = norm_bbox
            
            bbox = [
                x1_norm * image_width,
                y1_norm * image_height,
                x2_norm * image_width,
                y2_norm * image_height
            ]
            
            return bbox
            
        except Exception as e:
            self.logger.error(f"Error denormalizing bbox: {e}")
            return norm_bbox
    
    def calculate_bbox_properties(self, bbox: List[float]) -> Dict[str, float]:
        """Calculate comprehensive bounding box properties"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Basic dimensions
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Aspect ratio
            aspect_ratio = width / height if height > 0 else 0.0
            
            # Perimeter
            perimeter = 2 * (width + height)
            
            # Diagonal length
            diagonal = np.sqrt(width**2 + height**2)
            
            return {
                'width': width,
                'height': height,
                'area': area,
                'center_x': center_x,
                'center_y': center_y,
                'aspect_ratio': aspect_ratio,
                'perimeter': perimeter,
                'diagonal': diagonal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating bbox properties: {e}")
            return {}
    
    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calculate intersection coordinates
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)
            
            # Check if there's an intersection
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0
            
            # Calculate intersection area
            intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            
            # Calculate union area
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - intersection_area
            
            # Calculate IoU
            iou = intersection_area / union_area if union_area > 0 else 0.0
            
            return iou
            
        except Exception as e:
            self.logger.error(f"Error calculating IoU: {e}")
            return 0.0
    
    def calculate_center_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Euclidean distance between bbox centers"""
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            center1_x = (x1_1 + x2_1) / 2
            center1_y = (y1_1 + y2_1) / 2
            center2_x = (x1_2 + x2_2) / 2
            center2_y = (y1_2 + y2_2) / 2
            
            distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
            
            return distance
            
        except Exception as e:
            self.logger.error(f"Error calculating center distance: {e}")
            return float('inf')
    
    def expand_bbox(self, bbox: List[float], expansion_factor: float) -> List[float]:
        """Expand bounding box by a factor while maintaining center"""
        try:
            x1, y1, x2, y2 = bbox
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            new_width = width * expansion_factor
            new_height = height * expansion_factor
            
            expanded_bbox = [
                center_x - new_width / 2,
                center_y - new_height / 2,
                center_x + new_width / 2,
                center_y + new_height / 2
            ]
            
            return expanded_bbox
            
        except Exception as e:
            self.logger.error(f"Error expanding bbox: {e}")
            return bbox
    
    def crop_bbox_from_image(self, image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Crop image region defined by bounding box"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Convert to integer coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Clamp to image bounds
            height, width = image.shape[:2]
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))
            
            # Ensure valid crop region
            if x2 <= x1 or y2 <= y1:
                self.logger.warning("Invalid crop region")
                return None
            
            # Crop the image
            cropped = image[y1:y2, x1:x2]
            
            return cropped
            
        except Exception as e:
            self.logger.error(f"Error cropping bbox from image: {e}")
            return None
    
    def draw_bbox(self, image: np.ndarray, bbox: List[float], 
                  color: Tuple[int, int, int] = (0, 255, 0), 
                  thickness: int = 2, 
                  label: Optional[str] = None) -> np.ndarray:
        """Draw bounding box on image with optional label"""
        try:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label if provided
            if label:
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Background for label
                cv2.rectangle(image, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            color, -1)
                
                # Label text
                cv2.putText(image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error drawing bbox: {e}")
            return image
    
    def non_max_suppression(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        try:
            if not detections:
                return []
            
            # Sort by confidence (highest first)
            detections_sorted = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
            
            keep_indices = []
            for i, detection in enumerate(detections_sorted):
                bbox_i = detection.get('bbox', [])
                if len(bbox_i) != 4:
                    continue
                
                should_keep = True
                for j in keep_indices:
                    bbox_j = detections_sorted[j].get('bbox', [])
                    if len(bbox_j) != 4:
                        continue
                    
                    # Calculate IoU
                    iou = self.calculate_iou(bbox_i, bbox_j)
                    
                    # If IoU is too high, suppress this detection
                    if iou > iou_threshold:
                        should_keep = False
                        break
                
                if should_keep:
                    keep_indices.append(i)
            
            # Return kept detections
            suppressed_detections = [detections_sorted[i] for i in keep_indices]
            
            self.logger.debug(f"NMS: {len(detections)} -> {len(suppressed_detections)} detections")
            return suppressed_detections
            
        except Exception as e:
            self.logger.error(f"Error in non-max suppression: {e}")
            return detections
    
    def scale_bbox(self, bbox: List[float], scale_x: float, scale_y: float) -> List[float]:
        """Scale bounding box coordinates by given factors"""
        try:
            x1, y1, x2, y2 = bbox
            
            scaled_bbox = [
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y
            ]
            
            return scaled_bbox
            
        except Exception as e:
            self.logger.error(f"Error scaling bbox: {e}")
            return bbox
    
    def validate_bbox(self, bbox: List[float], image_width: int, image_height: int) -> bool:
        """Validate bounding box coordinates"""
        try:
            if len(bbox) != 4:
                return False
            
            x1, y1, x2, y2 = bbox
            
            # Check coordinate order
            if x2 <= x1 or y2 <= y1:
                return False
            
            # Check bounds
            if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
                return False
            
            # Check minimum size
            if (x2 - x1) < 1 or (y2 - y1) < 1:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating bbox: {e}")
            return False
    
    def convert_format(self, bbox: List[float], from_format: str, to_format: str) -> List[float]:
        """Convert between different bounding box formats"""
        try:
            # Supported formats: 'xyxy', 'xywh', 'cxcywh'
            if from_format == to_format:
                return bbox
            
            if from_format == 'xyxy':
                x1, y1, x2, y2 = bbox
                if to_format == 'xywh':
                    return [x1, y1, x2 - x1, y2 - y1]
                elif to_format == 'cxcywh':
                    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
            
            elif from_format == 'xywh':
                x, y, w, h = bbox
                if to_format == 'xyxy':
                    return [x, y, x + w, y + h]
                elif to_format == 'cxcywh':
                    return [x + w / 2, y + h / 2, w, h]
            
            elif from_format == 'cxcywh':
                cx, cy, w, h = bbox
                if to_format == 'xyxy':
                    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
                elif to_format == 'xywh':
                    return [cx - w / 2, cy - h / 2, w, h]
            
            self.logger.warning(f"Unsupported format conversion: {from_format} -> {to_format}")
            return bbox
            
        except Exception as e:
            self.logger.error(f"Error converting bbox format: {e}")
            return bbox