"""
Process Communication Utilities for Multi-Process Architecture
Handles shared memory and queue communication between processes
"""

import multiprocessing as mp
import queue
import time
import numpy as np
import pickle
from typing import Any, Optional, Dict, List, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger


class ProcessSafeQueue:
    """Process-safe queue wrapper with timeout support"""
    
    def __init__(self, maxsize: int = 10):
        self.queue = mp.Queue(maxsize=maxsize)
        self.logger = get_logger("ProcessSafeQueue")
    
    def put(self, item: Any, timeout: float = 1.0) -> bool:
        """Put item in queue with timeout"""
        try:
            self.queue.put(item, timeout=timeout)
            return True
        except queue.Full:
            self.logger.warning("Queue full, dropping item")
            return False
    
    def put_nowait(self, item: Any) -> bool:
        """Put item in queue without blocking"""
        try:
            self.queue.put_nowait(item)
            return True
        except queue.Full:
            return False
    
    def get(self, timeout: float = 1.0) -> Optional[Any]:
        """Get item from queue with timeout"""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_nowait(self) -> Optional[Any]:
        """Get item from queue without blocking"""
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None
    
    def size(self) -> int:
        """Get approximate queue size"""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self.queue.empty()
    
    def clear(self):
        """Clear all items from queue"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break


class SharedImageBuffer:
    """Shared memory buffer for images between processes"""
    
    def __init__(self, width: int, height: int, channels: int = 3, buffer_size: int = 2):
        self.width = width
        self.height = height
        self.channels = channels
        self.buffer_size = buffer_size
        
        # Calculate buffer dimensions
        self.image_size = width * height * channels
        self.total_size = self.image_size * buffer_size
        
        # Create shared memory array
        self.shared_array = mp.Array('B', self.total_size)  # Unsigned byte array
        
        # Shared metadata
        self.metadata = mp.Array('d', buffer_size * 3)  # timestamp, width, height for each buffer
        
        # Synchronization
        self.lock = mp.Lock()
        self.write_index = mp.Value('i', 0)
        self.read_index = mp.Value('i', 0)
        self.frame_count = mp.Value('i', 0)
        
        self.logger = get_logger("SharedImageBuffer")
        self.logger.info(f"Created shared image buffer: {width}x{height}x{channels}, {buffer_size} buffers")
    
    def write_image(self, image: np.ndarray, timestamp: float) -> bool:
        """Write image to shared buffer"""
        if image.shape != (self.height, self.width, self.channels):
            self.logger.error(f"Image shape mismatch: {image.shape} != {(self.height, self.width, self.channels)}")
            return False
        
        with self.lock:
            # Get write position
            write_pos = self.write_index.value
            
            # Calculate buffer offset
            buffer_offset = write_pos * self.image_size
            
            # Copy image data to shared memory
            flat_image = image.flatten()
            for i in range(self.image_size):
                self.shared_array[buffer_offset + i] = int(flat_image[i])
            
            # Update metadata
            meta_offset = write_pos * 3
            self.metadata[meta_offset] = timestamp
            self.metadata[meta_offset + 1] = self.width
            self.metadata[meta_offset + 2] = self.height
            
            # Update write index
            self.write_index.value = (write_pos + 1) % self.buffer_size
            self.frame_count.value += 1
            
            return True
    
    def read_image(self) -> Optional[Tuple[np.ndarray, float]]:
        """Read latest image from shared buffer"""
        with self.lock:
            # Check if we have data
            if self.frame_count.value == 0:
                return None
            
            # Find latest valid frame
            write_pos = self.write_index.value
            read_pos = (write_pos - 1) % self.buffer_size
            
            # Calculate buffer offset
            buffer_offset = read_pos * self.image_size
            
            # Read image data from shared memory
            image_data = np.array(self.shared_array[buffer_offset:buffer_offset + self.image_size], dtype=np.uint8)
            image = image_data.reshape((self.height, self.width, self.channels))
            
            # Read metadata
            meta_offset = read_pos * 3
            timestamp = self.metadata[meta_offset]
            
            return image.copy(), timestamp
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            return {
                'frame_count': self.frame_count.value,
                'write_index': self.write_index.value,
                'buffer_size': self.buffer_size,
                'image_dimensions': (self.width, self.height, self.channels)
            }


class ProcessController:
    """Base class for process controllers"""
    
    def __init__(self, process_name: str):
        self.process_name = process_name
        self.logger = get_logger(process_name)
        
        # Process control
        self.running = mp.Value('i', 0)  # 0 = stopped, 1 = running
        self.should_stop = mp.Value('i', 0)  # 0 = continue, 1 = stop
        
        # Performance tracking
        self.process_fps = mp.Value('d', 0.0)
        self.frame_count = mp.Value('i', 0)
        
        # Process handle
        self.process: Optional[mp.Process] = None
    
    def start_process(self, target_function, *args, **kwargs):
        """Start the process"""
        if self.is_running():
            self.logger.warning(f"Process {self.process_name} is already running")
            return
        
        self.should_stop.value = 0
        self.running.value = 1
        
        self.process = mp.Process(
            target=target_function,
            args=args,
            kwargs=kwargs,
            name=self.process_name
        )
        self.process.start()
        
        self.logger.info(f"Started process {self.process_name} (PID: {self.process.pid})")
    
    def stop_process(self, timeout: float = 5.0):
        """Stop the process"""
        if not self.is_running():
            return
        
        self.logger.info(f"Stopping process {self.process_name}...")
        self.should_stop.value = 1
        
        if self.process:
            self.process.join(timeout=timeout)
            
            if self.process.is_alive():
                self.logger.warning(f"Process {self.process_name} did not stop gracefully, terminating...")
                self.process.terminate()
                self.process.join(timeout=2.0)
                
                if self.process.is_alive():
                    self.logger.error(f"Process {self.process_name} could not be terminated")
        
        self.running.value = 0
        self.logger.info(f"Process {self.process_name} stopped")
    
    def is_running(self) -> bool:
        """Check if process is running"""
        return self.running.value == 1 and (self.process is None or self.process.is_alive())
    
    def should_continue(self) -> bool:
        """Check if process should continue running"""
        return self.should_stop.value == 0
    
    def update_performance(self, fps: float):
        """Update performance metrics"""
        self.process_fps.value = fps
        self.frame_count.value += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get process statistics"""
        return {
            'process_name': self.process_name,
            'is_running': self.is_running(),
            'fps': self.process_fps.value,
            'frame_count': self.frame_count.value,
            'pid': self.process.pid if self.process else None
        }


def serialize_detections(detections) -> bytes:
    """Serialize detection list for queue transmission"""
    try:
        # Convert detections to serializable format
        serializable_dets = []
        for det in detections:
            if hasattr(det, 'to_dict'):
                serializable_dets.append(det.to_dict())
            else:
                serializable_dets.append(det)
        
        return pickle.dumps(serializable_dets)
    except Exception as e:
        logger = get_logger("serialize_detections")
        logger.error(f"Failed to serialize detections: {e}")
        return pickle.dumps([])


def deserialize_detections(data: bytes):
    """Deserialize detection list from queue"""
    try:
        return pickle.loads(data)
    except Exception as e:
        logger = get_logger("deserialize_detections")
        logger.error(f"Failed to deserialize detections: {e}")
        return []


def serialize_tracks(tracks: List[Dict]) -> bytes:
    """Serialize track list for queue transmission"""
    try:
        return pickle.dumps(tracks)
    except Exception as e:
        logger = get_logger("serialize_tracks")
        logger.error(f"Failed to serialize tracks: {e}")
        return pickle.dumps([])


def deserialize_tracks(data: bytes) -> List[Dict]:
    """Deserialize track list from queue"""
    try:
        return pickle.loads(data)
    except Exception as e:
        logger = get_logger("deserialize_tracks")
        logger.error(f"Failed to deserialize tracks: {e}")
        return []