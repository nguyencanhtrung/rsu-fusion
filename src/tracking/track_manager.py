"""
Track Lifecycle Management
Manages the creation, updating, and deletion of tracking objects
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict, deque
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger
from core.config_manager import get_config_manager


class TrackManager:
    """Manages the lifecycle and metadata of tracking objects"""
    
    def __init__(self):
        self.logger = get_logger("TrackManager")
        
        # Load configuration
        self._load_config()
        
        # Track storage and management
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.track_history: Dict[int, List[Dict[str, Any]]] = {}
        self.deleted_tracks: Dict[int, Dict[str, Any]] = {}
        
        # Track ID management
        self.next_track_id = 1
        self.max_track_id = 10000  # Prevent ID overflow
        
        # Performance tracking
        self.track_birth_times: Dict[int, float] = {}
        self.track_update_counts: Dict[int, int] = defaultdict(int)
        self.track_statistics = {
            'total_created': 0,
            'total_deleted': 0,
            'active_count': 0,
            'avg_track_lifetime': 0.0
        }
        
        self.logger.info("Track manager initialized")
        self.logger.info(f"Max track age: {self.max_track_age} frames")
        self.logger.info(f"Min track hits: {self.min_track_hits}")
    
    def _load_config(self):
        """Load tracking configuration parameters"""
        try:
            config_manager = get_config_manager()
            tracking_config = config_manager.get_module_config('tracking')
            
            track_mgmt_config = tracking_config.get('tracking', {}).get('track_management', {})
            
            self.max_track_age = track_mgmt_config.get('max_age', 30)
            self.min_track_hits = track_mgmt_config.get('min_hits', 3)
            self.max_history_length = track_mgmt_config.get('max_history_length', 100)
            self.cleanup_interval = track_mgmt_config.get('cleanup_interval', 100)  # frames
            
        except Exception as e:
            self.logger.warning(f"Could not load track management config: {e}, using defaults")
            self.max_track_age = 30
            self.min_track_hits = 3
            self.max_history_length = 100
            self.cleanup_interval = 100
        
        self.frame_count = 0
    
    def create_track(self, detection: Dict[str, Any]) -> int:
        """Create a new track from detection"""
        try:
            # Generate new track ID
            track_id = self._get_next_track_id()
            
            # Initialize track with detection data
            track_data = {
                'track_id': track_id,
                'class_name': detection.get('class_name', 'unknown'),
                'class_id': detection.get('class_id', -1),
                'confidence': detection.get('confidence', 0.0),
                'bbox': detection.get('bbox', []),
                'center': detection.get('center', []),
                'area': detection.get('area', 0.0),
                'created_frame': self.frame_count,
                'last_update_frame': self.frame_count,
                'age': 0,
                'hits': 1,
                'hit_streak': 1,
                'time_since_update': 0,
                'state': 'tentative',  # tentative -> confirmed -> deleted
                'is_confirmed': False,
                'velocity': [0.0, 0.0],  # [du, dv] in pixels per frame
                'track_quality': 1.0
            }
            
            # Store track
            self.active_tracks[track_id] = track_data
            self.track_history[track_id] = [track_data.copy()]
            self.track_birth_times[track_id] = time.time()
            self.track_update_counts[track_id] = 1
            
            # Update statistics
            self.track_statistics['total_created'] += 1
            self.track_statistics['active_count'] = len(self.active_tracks)
            
            self.logger.debug(f"Created track {track_id} for {track_data['class_name']}")
            return track_id
            
        except Exception as e:
            self.logger.error(f"Error creating track: {e}")
            return -1
    
    def update_track(self, track_id: int, detection: Dict[str, Any], 
                    predicted_state: Optional[Dict[str, Any]] = None) -> bool:
        """Update existing track with new detection"""
        try:
            if track_id not in self.active_tracks:
                self.logger.warning(f"Attempting to update non-existent track {track_id}")
                return False
            
            track = self.active_tracks[track_id]
            
            # Calculate velocity if we have previous position
            old_center = track.get('center', [])
            new_center = detection.get('center', [])
            
            if len(old_center) == 2 and len(new_center) == 2:
                velocity = [
                    new_center[0] - old_center[0],  # du
                    new_center[1] - old_center[1]   # dv
                ]
                track['velocity'] = velocity
            
            # Update track data
            track.update({
                'confidence': detection.get('confidence', track['confidence']),
                'bbox': detection.get('bbox', track['bbox']),
                'center': new_center,
                'area': detection.get('area', track['area']),
                'last_update_frame': self.frame_count,
                'hits': track['hits'] + 1,
                'hit_streak': track['hit_streak'] + 1,
                'time_since_update': 0
            })
            
            # Update track quality based on hit streak and consistency
            self._update_track_quality(track_id)
            
            # Check if track should be confirmed
            if (track['hit_streak'] >= self.min_track_hits and 
                track['state'] == 'tentative'):
                track['state'] = 'confirmed'
                track['is_confirmed'] = True
                self.logger.debug(f"Track {track_id} confirmed")
            
            # Add to history
            if track_id in self.track_history:
                self.track_history[track_id].append(track.copy())
                
                # Limit history length
                if len(self.track_history[track_id]) > self.max_history_length:
                    self.track_history[track_id].pop(0)
            
            # Update counters
            self.track_update_counts[track_id] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating track {track_id}: {e}")
            return False
    
    def predict_track(self, track_id: int) -> bool:
        """Update track for prediction step (no detection matched)"""
        try:
            if track_id not in self.active_tracks:
                return False
            
            track = self.active_tracks[track_id]
            
            # Update track state for missed detection
            track['age'] += 1
            track['time_since_update'] += 1
            track['hit_streak'] = 0  # Reset hit streak
            
            # Predict next position using velocity
            if 'velocity' in track and 'center' in track:
                velocity = track['velocity']
                center = track['center']
                
                if len(velocity) == 2 and len(center) == 2:
                    predicted_center = [
                        center[0] + velocity[0],
                        center[1] + velocity[1]
                    ]
                    track['center'] = predicted_center
                    
                    # Update bbox based on predicted center (maintaining size)
                    bbox = track.get('bbox', [])
                    if len(bbox) == 4:
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        track['bbox'] = [
                            predicted_center[0] - width / 2,
                            predicted_center[1] - height / 2,
                            predicted_center[0] + width / 2,
                            predicted_center[1] + height / 2
                        ]
            
            # Decrease track quality for missed detections
            track['track_quality'] *= 0.95  # Gradual decay
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error predicting track {track_id}: {e}")
            return False
    
    def delete_track(self, track_id: int, reason: str = "aged_out") -> bool:
        """Delete a track and move it to deleted tracks"""
        try:
            if track_id not in self.active_tracks:
                return False
            
            track = self.active_tracks[track_id]
            
            # Calculate lifetime
            birth_time = self.track_birth_times.get(track_id, time.time())
            lifetime = time.time() - birth_time
            
            # Add deletion metadata
            track.update({
                'deletion_reason': reason,
                'deletion_frame': self.frame_count,
                'lifetime_seconds': lifetime,
                'total_hits': track['hits']
            })
            
            # Move to deleted tracks
            self.deleted_tracks[track_id] = track
            
            # Remove from active tracking
            del self.active_tracks[track_id]
            if track_id in self.track_birth_times:
                del self.track_birth_times[track_id]
            
            # Update statistics
            self.track_statistics['total_deleted'] += 1
            self.track_statistics['active_count'] = len(self.active_tracks)
            
            # Update average lifetime
            self._update_avg_lifetime()
            
            self.logger.debug(f"Deleted track {track_id} ({reason}) after {lifetime:.1f}s, {track['hits']} hits")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting track {track_id}: {e}")
            return False
    
    def get_track(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get track data by ID"""
        return self.active_tracks.get(track_id)
    
    def get_all_tracks(self) -> Dict[int, Dict[str, Any]]:
        """Get all active tracks"""
        return self.active_tracks.copy()
    
    def get_confirmed_tracks(self) -> Dict[int, Dict[str, Any]]:
        """Get only confirmed tracks"""
        return {
            track_id: track for track_id, track in self.active_tracks.items()
            if track.get('is_confirmed', False)
        }
    
    def get_track_history(self, track_id: int, max_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get track history"""
        if track_id not in self.track_history:
            return []
        
        history = self.track_history[track_id]
        if max_length:
            return history[-max_length:]
        return history
    
    def cleanup_old_tracks(self) -> int:
        """Remove tracks that are too old or low quality"""
        deleted_count = 0
        tracks_to_delete = []
        
        try:
            for track_id, track in self.active_tracks.items():
                should_delete = False
                reason = ""
                
                # Check age
                if track['time_since_update'] > self.max_track_age:
                    should_delete = True
                    reason = "aged_out"
                
                # Check track quality
                elif track.get('track_quality', 1.0) < 0.1:
                    should_delete = True
                    reason = "low_quality"
                
                # Check if tentative track hasn't been confirmed
                elif (track['age'] > self.max_track_age // 2 and 
                      track['state'] == 'tentative'):
                    should_delete = True
                    reason = "unconfirmed"
                
                if should_delete:
                    tracks_to_delete.append((track_id, reason))
            
            # Delete marked tracks
            for track_id, reason in tracks_to_delete:
                if self.delete_track(track_id, reason):
                    deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error during track cleanup: {e}")
            return deleted_count
    
    def update_frame(self):
        """Update frame counter and perform periodic cleanup"""
        self.frame_count += 1
        
        # Periodic cleanup
        if self.frame_count % self.cleanup_interval == 0:
            deleted_count = self.cleanup_old_tracks()
            if deleted_count > 0:
                self.logger.debug(f"Cleaned up {deleted_count} old tracks")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics"""
        try:
            stats = self.track_statistics.copy()
            
            # Current state
            stats.update({
                'active_tracks': len(self.active_tracks),
                'confirmed_tracks': len(self.get_confirmed_tracks()),
                'tentative_tracks': len(self.active_tracks) - len(self.get_confirmed_tracks()),
                'frame_count': self.frame_count
            })
            
            # Track age distribution
            if self.active_tracks:
                ages = [track['age'] for track in self.active_tracks.values()]
                stats.update({
                    'avg_track_age': np.mean(ages),
                    'max_track_age': max(ages),
                    'min_track_age': min(ages)
                })
            
            # Class distribution
            class_counts = defaultdict(int)
            for track in self.active_tracks.values():
                class_counts[track.get('class_name', 'unknown')] += 1
            stats['class_distribution'] = dict(class_counts)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return self.track_statistics.copy()
    
    def _get_next_track_id(self) -> int:
        """Generate next available track ID"""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        # Handle ID overflow
        if self.next_track_id > self.max_track_id:
            self.next_track_id = 1
            # Find next unused ID
            while self.next_track_id in self.active_tracks:
                self.next_track_id += 1
                if self.next_track_id > self.max_track_id:
                    self.logger.warning("Track ID space exhausted!")
                    self.next_track_id = 1
                    break
        
        return track_id
    
    def _update_track_quality(self, track_id: int):
        """Update track quality metric based on various factors"""
        try:
            if track_id not in self.active_tracks:
                return
            
            track = self.active_tracks[track_id]
            
            # Factors affecting quality
            hit_ratio = track['hits'] / max(track['age'], 1)
            confidence = track.get('confidence', 0.0)
            consistency_factor = min(track['hit_streak'] / 10.0, 1.0)
            
            # Combined quality score
            quality = (hit_ratio * 0.4 + confidence * 0.4 + consistency_factor * 0.2)
            track['track_quality'] = min(1.0, quality)
            
        except Exception as e:
            self.logger.error(f"Error updating track quality for {track_id}: {e}")
    
    def _update_avg_lifetime(self):
        """Update average track lifetime statistic"""
        try:
            if not self.deleted_tracks:
                return
            
            lifetimes = [
                track.get('lifetime_seconds', 0.0) 
                for track in self.deleted_tracks.values()
            ]
            
            if lifetimes:
                self.track_statistics['avg_track_lifetime'] = np.mean(lifetimes)
            
        except Exception as e:
            self.logger.error(f"Error updating average lifetime: {e}")
    
    def reset(self):
        """Reset track manager state"""
        self.logger.info("Resetting track manager...")
        
        self.active_tracks.clear()
        self.track_history.clear()
        self.deleted_tracks.clear()
        self.track_birth_times.clear()
        self.track_update_counts.clear()
        
        self.next_track_id = 1
        self.frame_count = 0
        
        self.track_statistics = {
            'total_created': 0,
            'total_deleted': 0,
            'active_count': 0,
            'avg_track_lifetime': 0.0
        }
    
    def cleanup(self):
        """Clean up track manager resources"""
        self.logger.info("Cleaning up track manager...")
        self.reset()