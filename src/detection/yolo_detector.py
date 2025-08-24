"""
YOLO Detection System for RSU Fusion
YOLOv8m integration for vehicle detection (cars, trucks)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger
from core.config_manager import get_config_manager


class Detection:
    """Single detection result"""
    
    def __init__(self, bbox: List[float], confidence: float, class_id: int, class_name: str):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        
        # Calculate center and dimensions
        self.center_x = (bbox[0] + bbox[2]) / 2
        self.center_y = (bbox[1] + bbox[3]) / 2
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.area = self.width * self.height
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'bbox': self.bbox,
            'center': [self.center_x, self.center_y],
            'width': self.width,
            'height': self.height,
            'area': self.area,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name
        }


class YOLODetector:
    """YOLOv8 detector for vehicle detection"""
    
    def __init__(self, model_path: str = "yolov8m.pt"):
        self.logger = get_logger("YOLODetector")
        self.config_manager = get_config_manager()
        
        # Load detection configuration
        try:
            detection_config = self.config_manager.get_module_config('detection')['detection']
        except:
            # Fallback configuration
            detection_config = {
                'classes': {'car': 2, 'truck': 7},
                'filtering': {
                    'confidence_threshold': 0.6,
                    'nms_threshold': 0.4,
                    'min_box_area': 500,
                    'max_box_area': 200000
                }
            }
        
        # Target classes: cars and trucks only
        self.target_classes = [detection_config['classes']['car'], 
                              detection_config['classes']['truck']]
        self.class_names = {2: 'car', 7: 'truck'}
        
        # Detection parameters (accuracy priority)
        filtering_config = detection_config['filtering']
        self.confidence_threshold = filtering_config['confidence_threshold']
        self.nms_threshold = filtering_config['nms_threshold']
        self.min_box_area = filtering_config['min_box_area']
        self.max_box_area = filtering_config['max_box_area']
        
        self.logger.info(f"Initializing YOLOv8 detector with model: {model_path}")
        self.logger.info(f"Target classes: {self.class_names}")
        self.logger.info(f"Confidence threshold: {self.confidence_threshold}")
        
        # Initialize YOLO model
        try:
            self.model = YOLO(model_path)
            self.model_loaded = True
            self.logger.info("✅ YOLOv8 model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.model_loaded = False
        
        # Performance tracking
        self.detection_times = []
        self.frame_count = 0
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on input image"""
        if not self.model_loaded:
            return []
        
        start_time = time.time()
        
        try:
            # Run YOLOv8 inference
            results = self.model(image, 
                               conf=self.confidence_threshold,
                               iou=self.nms_threshold,
                               classes=self.target_classes,
                               verbose=False)
            
            detections = []
            
            # Process results
            if results and len(results) > 0:
                result = results[0]  # Single image
                
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Extract box information
                        bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Filter by area
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        area = width * height
                        
                        if self.min_box_area <= area <= self.max_box_area:
                            class_name = self.class_names.get(class_id, f'class_{class_id}')
                            
                            detection = Detection(
                                bbox=bbox.tolist(),
                                confidence=confidence,
                                class_id=class_id,
                                class_name=class_name
                            )
                            detections.append(detection)
            
            # Performance tracking
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            self.frame_count += 1
            
            # Log performance every 30 frames
            if self.frame_count % 30 == 0:
                avg_time = np.mean(self.detection_times[-30:])
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.logger.debug(f"Detection performance: {fps:.1f} FPS, "
                                f"{len(detections)} objects detected")
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get detection performance statistics"""
        if not self.detection_times:
            return {'avg_fps': 0, 'avg_detection_time': 0}
        
        recent_times = self.detection_times[-100:]  # Last 100 detections
        avg_time = np.mean(recent_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_fps': avg_fps,
            'avg_detection_time': avg_time,
            'total_frames': self.frame_count
        }
    
    def cleanup(self):
        """Clean up detector resources"""
        self.logger.info("Cleaning up YOLO detector...")
        if hasattr(self, 'model'):
            del self.model
        self.model_loaded = False
