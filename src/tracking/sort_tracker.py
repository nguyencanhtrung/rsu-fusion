"""
SORT Tracker Implementation
Simple Online and Realtime Tracking with 4D Kalman Filter
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import linear_sum_assignment
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger
from core.config_manager import get_config_manager
from .kalman_filter import create_kalman_filter
from detection.yolo_detector import Detection


class Track:
    """Single object track"""
    
    def __init__(self, track_id: int, detection: Detection, dt: float = 1.0):
        self.track_id = track_id
        self.dt = dt
        
        # Initialize Kalman filter with detection center
        self.kalman_filter = create_kalman_filter(dt=dt)
        initial_measurement = np.array([detection.center_x, detection.center_y])
        self.kalman_filter.initialize(initial_measurement)
        
        # Track state
        self.age = 0  # Frames since track creation
        self.hits = 1  # Number of detections associated with this track
        self.hit_streak = 1  # Consecutive hits
        self.time_since_update = 0  # Frames since last update
        
        # Track history for visualization
        self.history = [(detection.center_x, detection.center_y)]
        self.max_history = 30  # Maximum trail length
        
        # Detection info
        self.last_detection = detection
        self.class_name = detection.class_name
        self.confidence_history = [detection.confidence]
        
        # Predicted bounding box (for display)
        self.predicted_bbox = detection.bbox.copy()
        
        self.logger = get_logger("Track")
        self.logger.debug(f"Created track {track_id} for {detection.class_name} "
                         f"at ({detection.center_x:.1f}, {detection.center_y:.1f})")
    
    def predict(self) -> np.ndarray:
        """Predict next position"""
        predicted_pos = self.kalman_filter.predict()
        
        # Update predicted bounding box (assuming same size as last detection)
        if self.last_detection:
            half_width = self.last_detection.width / 2
            half_height = self.last_detection.height / 2
            
            self.predicted_bbox = [
                predicted_pos[0] - half_width,   # x1
                predicted_pos[1] - half_height,  # y1
                predicted_pos[0] + half_width,   # x2
                predicted_pos[1] + half_height   # y2
            ]
        
        self.age += 1
        self.time_since_update += 1
        
        return predicted_pos
    
    def update(self, detection: Detection):
        """Update track with new detection"""
        # Update Kalman filter
        measurement = np.array([detection.center_x, detection.center_y])
        self.kalman_filter.update(measurement)
        
        # Update track state
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        
        # Update history
        self.history.append((detection.center_x, detection.center_y))
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Update detection info
        self.last_detection = detection
        self.confidence_history.append(detection.confidence)
        if len(self.confidence_history) > 10:  # Keep last 10 confidences
            self.confidence_history.pop(0)
        
        # Update predicted bbox with actual detection
        self.predicted_bbox = detection.bbox.copy()
        
        self.logger.debug(f"Updated track {self.track_id} with {detection.class_name} "
                         f"at ({detection.center_x:.1f}, {detection.center_y:.1f})")
    
    def get_current_position(self) -> np.ndarray:
        """Get current position estimate"""
        return self.kalman_filter.get_position()
    
    def get_predicted_bbox(self) -> List[float]:
        """Get predicted bounding box [x1, y1, x2, y2]"""
        return self.predicted_bbox
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get track state as dictionary"""
        position = self.get_current_position()
        velocity = self.kalman_filter.get_velocity()
        
        return {
            'track_id': self.track_id,
            'position': position.tolist(),
            'velocity': velocity.tolist(),
            'bbox': self.predicted_bbox,
            'class_name': self.class_name,
            'confidence': np.mean(self.confidence_history),
            'age': self.age,
            'hits': self.hits,
            'hit_streak': self.hit_streak,
            'time_since_update': self.time_since_update,
            'history': self.history.copy()
        }


class SORTTracker:
    """Simple Online and Realtime Tracking"""
    
    def __init__(self, dt: float = 1.0):
        self.logger = get_logger("SORTTracker")
        self.dt = dt
        
        # Load tracking configuration
        try:
            config_manager = get_config_manager()
            tracking_config = config_manager.get_module_config('tracking')['tracking']['sort']
        except:
            # Fallback configuration
            tracking_config = {
                'max_age': 30,
                'min_hits': 3,
                'iou_threshold': 0.3
            }
        
        # Tracking parameters
        self.max_age = tracking_config['max_age']  # Frames before track deletion
        self.min_hits = tracking_config['min_hits']  # Detections before track confirmation
        self.iou_threshold = tracking_config['iou_threshold']  # Assignment threshold
        
        # Track management
        self.tracks: List[Track] = []
        self.track_count = 0
        self.frame_count = 0
        
        # Performance tracking
        self.processing_times = []
        
        self.logger.info(f"Initialized SORT tracker: max_age={self.max_age}, "
                        f"min_hits={self.min_hits}, iou_threshold={self.iou_threshold}")
    
    def update(self, detections: List[Detection]) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections
        Returns list of confirmed tracks
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Step 1: Predict all existing tracks
        for track in self.tracks:
            track.predict()
        
        # Step 2: Associate detections with tracks
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(
            detections, self.tracks
        )
        
        # Step 3: Update matched tracks
        for m in matched:
            self.tracks[m[1]].update(detections[m[0]])
        
        # Step 4: Create new tracks for unmatched detections
        for i in unmatched_dets:
            self._create_new_track(detections[i])
        
        # Step 5: Delete old tracks
        self.tracks = [track for track in self.tracks if self._should_keep_track(track)]
        
        # Step 6: Get confirmed tracks
        confirmed_tracks = [
            track.get_state_dict() 
            for track in self.tracks 
            if track.hit_streak >= self.min_hits
        ]
        
        # Performance tracking
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Log performance every 30 frames
        if self.frame_count % 30 == 0:
            avg_time = np.mean(self.processing_times[-30:])
            self.logger.debug(f"SORT performance: {avg_time*1000:.1f}ms/frame, "
                             f"{len(confirmed_tracks)} confirmed tracks")
        
        return confirmed_tracks
    
    def _associate_detections_to_tracks(self, detections: List[Detection], 
                                       tracks: List[Track]) -> Tuple[np.ndarray, List[int], List[int]]:
        """Associate detections to existing tracks using Hungarian algorithm"""
        
        if len(tracks) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), []
        
        # Compute IoU cost matrix
        iou_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)
        
        for d, detection in enumerate(detections):
            det_bbox = detection.bbox
            
            for t, track in enumerate(tracks):
                track_bbox = track.get_predicted_bbox()
                iou_matrix[d, t] = self._calculate_iou(det_bbox, track_bbox)
        
        # Convert IoU to cost (1 - IoU)
        cost_matrix = 1.0 - iou_matrix
        
        # Apply Hungarian algorithm
        if cost_matrix.size > 0:
            matched_indices = linear_sum_assignment(cost_matrix)
            matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
        
        # Filter out matches with IoU below threshold
        matches = []
        for match in matched_indices:
            if iou_matrix[match[0], match[1]] >= self.iou_threshold:
                matches.append(match.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        # Find unmatched detections and tracks
        unmatched_detections = []
        for d in range(len(detections)):
            if len(matches) == 0 or d not in matches[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_tracks = []
        for t in range(len(tracks)):
            if len(matches) == 0 or t not in matches[:, 1]:
                unmatched_tracks.append(t)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def _create_new_track(self, detection: Detection):
        """Create new track for unmatched detection"""
        self.track_count += 1
        new_track = Track(self.track_count, detection, dt=self.dt)
        self.tracks.append(new_track)
        
        self.logger.debug(f"Created new track {self.track_count} for {detection.class_name}")
    
    def _should_keep_track(self, track: Track) -> bool:
        """Determine if track should be kept or deleted"""
        # Delete tracks that haven't been updated for too long
        if track.time_since_update > self.max_age:
            self.logger.debug(f"Deleting track {track.track_id} (too old: {track.time_since_update} frames)")
            return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics"""
        return {
            'total_tracks': len(self.tracks),
            'confirmed_tracks': len([t for t in self.tracks if t.hit_streak >= self.min_hits]),
            'tentative_tracks': len([t for t in self.tracks if t.hit_streak < self.min_hits]),
            'total_created_tracks': self.track_count,
            'frame_count': self.frame_count,
            'avg_processing_time_ms': np.mean(self.processing_times[-100:]) * 1000 if self.processing_times else 0
        }
    
    def reset(self):
        """Reset tracker state"""
        self.tracks = []
        self.track_count = 0
        self.frame_count = 0
        self.processing_times = []
        self.logger.info("SORT tracker reset")
    
    def cleanup(self):
        """Cleanup tracker resources"""
        self.logger.info("Cleaning up SORT tracker...")
        self.tracks.clear()
