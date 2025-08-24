"""
Sensor Data Processing Pipeline
Handles image preprocessing and data preparation
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger


class ImageProcessor:
    """Processes camera images for detection and visualization"""
    
    def __init__(self):
        self.logger = get_logger("ImageProcessor")
        self.processing_times = []
        
        # Processing parameters
        self.resize_enabled = False
        self.target_size = None
        self.roi_enabled = False
        self.roi_coordinates = None
        
        self.logger.info("Image processor initialized")
    
    def set_resize_params(self, target_width: int, target_height: int):
        """Enable and configure image resizing"""
        self.resize_enabled = True
        self.target_size = (target_width, target_height)
        self.logger.info(f"Image resizing enabled: {self.target_size}")
    
    def set_roi_params(self, x1: int, y1: int, x2: int, y2: int):
        """Enable and configure region of interest cropping"""
        self.roi_enabled = True
        self.roi_coordinates = (x1, y1, x2, y2)
        self.logger.info(f"ROI cropping enabled: {self.roi_coordinates}")
    
    def preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for object detection"""
        start_time = time.time()
        
        try:
            processed_image = image.copy()
            
            # Apply ROI cropping if enabled
            if self.roi_enabled and self.roi_coordinates:
                x1, y1, x2, y2 = self.roi_coordinates
                processed_image = processed_image[y1:y2, x1:x2]
            
            # Apply resizing if enabled
            if self.resize_enabled and self.target_size:
                processed_image = cv2.resize(processed_image, self.target_size)
            
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return processed_image
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            return image
    
    def preprocess_for_visualization(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for visualization (minimal processing)"""
        try:
            # Just ensure correct format
            if image.dtype != np.uint8:
                image = np.clip(image * 255, 0, 255).astype(np.uint8)
            
            # Ensure 3 channels
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image for visualization: {e}")
            return image
    
    def enhance_image(self, image: np.ndarray, 
                     brightness: float = 0.0, 
                     contrast: float = 1.0,
                     gamma: float = 1.0) -> np.ndarray:
        """Apply image enhancement"""
        try:
            enhanced = image.astype(np.float32)
            
            # Apply brightness adjustment
            if brightness != 0.0:
                enhanced = enhanced + brightness * 255
            
            # Apply contrast adjustment
            if contrast != 1.0:
                enhanced = enhanced * contrast
            
            # Apply gamma correction
            if gamma != 1.0:
                enhanced = 255.0 * np.power(enhanced / 255.0, gamma)
            
            # Clip and convert back to uint8
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error enhancing image: {e}")
            return image
    
    def apply_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction filter"""
        try:
            # Apply bilateral filter for noise reduction while preserving edges
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            return denoised
            
        except Exception as e:
            self.logger.error(f"Error applying noise reduction: {e}")
            return image
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image values"""
        try:
            if image.dtype != np.float32:
                normalized = image.astype(np.float32) / 255.0
            else:
                normalized = image
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing image: {e}")
            return image
    
    def get_processing_stats(self) -> Dict[str, float]:
        """Get image processing performance statistics"""
        if not self.processing_times:
            return {'avg_processing_time_ms': 0, 'processing_fps': 0}
        
        recent_times = self.processing_times[-100:]  # Last 100 operations
        avg_time = np.mean(recent_times)
        processing_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_processing_time_ms': avg_time * 1000,
            'processing_fps': processing_fps,
            'total_processed': len(self.processing_times)
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.processing_times.clear()
        self.logger.info("Processing statistics reset")
