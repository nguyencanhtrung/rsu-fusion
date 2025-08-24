"""
Vehicle Class Filtering for Detection System
Handles filtering and classification of detected vehicles
"""

import numpy as np
from typing import List, Dict, Set, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger


class DetectionFilter:
    """Filters and processes vehicle detections based on class and criteria"""
    
    def __init__(self):
        self.logger = get_logger("DetectionFilter")
        
        # COCO class mappings for vehicle detection
        self.coco_vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        # Phase 1 target classes (expanded from reference truck-only)
        self.target_classes = {2, 5, 7}  # car, bus, truck
        
        # Vehicle priority for emergency scenarios
        self.priority_vehicles = {
            'ambulance': 1,  # Highest priority
            'bus': 2,        # Includes emergency vehicles
            'truck': 3,      # Large vehicles
            'car': 4         # Standard vehicles
        }
        
        # Filtering thresholds
        self.min_confidence = 0.5
        self.min_box_area = 500      # Minimum bounding box area in pixels
        self.max_box_area = 200000   # Maximum bounding box area in pixels
        self.min_aspect_ratio = 0.2  # Width/height minimum ratio
        self.max_aspect_ratio = 5.0  # Width/height maximum ratio
        
        self.logger.info("Detection filter initialized")
        self.logger.info(f"Target classes: {[self.coco_vehicle_classes[c] for c in self.target_classes]}")
    
    def filter_vehicle_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections to keep only vehicles meeting criteria"""
        filtered_detections = []
        
        for detection in detections:
            if self._is_valid_vehicle_detection(detection):
                # Add vehicle classification
                detection = self._classify_vehicle(detection)
                filtered_detections.append(detection)
        
        self.logger.debug(f"Filtered {len(detections)} -> {len(filtered_detections)} vehicle detections")
        return filtered_detections
    
    def _is_valid_vehicle_detection(self, detection: Dict) -> bool:
        """Check if detection meets vehicle filtering criteria"""
        try:
            # Check if it's a target vehicle class
            class_id = detection.get('class_id', -1)
            if class_id not in self.target_classes:
                return False
            
            # Check confidence threshold
            confidence = detection.get('confidence', 0.0)
            if confidence < self.min_confidence:
                return False
            
            # Check bounding box properties
            bbox = detection.get('bbox', [])
            if len(bbox) != 4:
                return False
            
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Check area bounds
            if not (self.min_box_area <= area <= self.max_box_area):
                return False
            
            # Check aspect ratio bounds  
            if height > 0:
                aspect_ratio = width / height
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    return False
            else:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating detection: {e}")
            return False
    
    def _classify_vehicle(self, detection: Dict) -> Dict:
        """Classify and enhance vehicle detection with additional metadata"""
        try:
            class_id = detection.get('class_id', -1)
            class_name = self.coco_vehicle_classes.get(class_id, 'unknown')
            
            # Special case: detect ambulances within bus class
            if class_id == 5:  # bus class
                # Could enhance with sub-classification logic here
                # For now, assume all buses could be emergency vehicles
                if self._might_be_ambulance(detection):
                    class_name = 'ambulance'
            
            # Add enhanced information
            detection['class_name'] = class_name
            detection['vehicle_priority'] = self.priority_vehicles.get(class_name, 5)
            detection['is_emergency_vehicle'] = (class_name == 'ambulance')
            
            return detection
            
        except Exception as e:
            self.logger.error(f"Error classifying vehicle: {e}")
            return detection
    
    def _might_be_ambulance(self, detection: Dict) -> bool:
        """Heuristic to identify potential ambulance from bus detections"""
        # Simple heuristics - could be enhanced with more sophisticated logic
        bbox = detection.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # Ambulances tend to be taller and more square than buses
            if height > 0:
                aspect_ratio = width / height
                # Ambulance aspect ratio typically between 1.2 and 2.0
                if 1.2 <= aspect_ratio <= 2.0:
                    return True
        
        return False
    
    def prioritize_detections(self, detections: List[Dict]) -> List[Dict]:
        """Sort detections by vehicle priority (emergency vehicles first)"""
        try:
            # Sort by priority (lower number = higher priority) then by confidence
            sorted_detections = sorted(
                detections, 
                key=lambda d: (
                    d.get('vehicle_priority', 5),
                    -d.get('confidence', 0.0)  # Negative for descending order
                )
            )
            
            return sorted_detections
            
        except Exception as e:
            self.logger.error(f"Error prioritizing detections: {e}")
            return detections
    
    def filter_by_region_of_interest(self, detections: List[Dict], 
                                   roi_bounds: Tuple[int, int, int, int]) -> List[Dict]:
        """Filter detections to only those within region of interest"""
        if not roi_bounds:
            return detections
        
        try:
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
            filtered_detections = []
            
            for detection in detections:
                bbox = detection.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Check if center is within ROI
                    if (roi_x1 <= center_x <= roi_x2 and 
                        roi_y1 <= center_y <= roi_y2):
                        filtered_detections.append(detection)
            
            self.logger.debug(f"ROI filtering: {len(detections)} -> {len(filtered_detections)}")
            return filtered_detections
            
        except Exception as e:
            self.logger.error(f"Error filtering by ROI: {e}")
            return detections
    
    def get_detection_statistics(self, detections: List[Dict]) -> Dict:
        """Get statistics about filtered detections"""
        try:
            stats = {
                'total_detections': len(detections),
                'class_counts': {},
                'emergency_vehicles': 0,
                'avg_confidence': 0.0,
                'confidence_range': (0.0, 0.0)
            }
            
            if not detections:
                return stats
            
            # Count by class
            confidences = []
            for detection in detections:
                class_name = detection.get('class_name', 'unknown')
                stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
                
                if detection.get('is_emergency_vehicle', False):
                    stats['emergency_vehicles'] += 1
                
                confidence = detection.get('confidence', 0.0)
                confidences.append(confidence)
            
            # Confidence statistics
            if confidences:
                stats['avg_confidence'] = np.mean(confidences)
                stats['confidence_range'] = (min(confidences), max(confidences))
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            return {'total_detections': len(detections)}
    
    def update_filter_parameters(self, new_params: Dict):
        """Update filtering parameters dynamically"""
        try:
            if 'min_confidence' in new_params:
                self.min_confidence = new_params['min_confidence']
                self.logger.info(f"Updated min_confidence to {self.min_confidence}")
            
            if 'min_box_area' in new_params:
                self.min_box_area = new_params['min_box_area']
                self.logger.info(f"Updated min_box_area to {self.min_box_area}")
            
            if 'target_classes' in new_params:
                self.target_classes = set(new_params['target_classes'])
                self.logger.info(f"Updated target_classes to {self.target_classes}")
            
        except Exception as e:
            self.logger.error(f"Error updating filter parameters: {e}")