"""
Camera Manager for CARLA RGB Camera
Handles camera creation, positioning, and image capture
"""

import carla
import numpy as np
import threading
import time
from typing import Optional, Callable, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger
from core.config_manager import get_config_manager


class CameraManager:
    """Manages CARLA RGB camera sensor"""
    
    def __init__(self, carla_client, world: carla.World):
        self.logger = get_logger("CameraManager")
        self.carla_client = carla_client
        self.world = world
        self.blueprint_library = world.get_blueprint_library()
        
        # Camera objects
        self.camera: Optional[carla.Actor] = None
        self.depth_camera: Optional[carla.Actor] = None
        self.camera_transform: Optional[carla.Transform] = None
        
        # Image handling
        self.current_image: Optional[np.ndarray] = None
        self.image_callback: Optional[Callable] = None
        self.image_lock = threading.Lock()
        
        # Load camera configuration
        self._load_camera_config()
        
        self.logger.info("Camera manager initialized")
    
    def _load_camera_config(self):
        """Load camera configuration from config files"""
        try:
            config_manager = get_config_manager()
            camera_config = config_manager.get_module_config('camera')['camera']
            
            # Camera positioning (exact reference compliance)
            transform_config = camera_config['transform']
            location_config = transform_config['location']
            self.camera_position = (
                location_config['x'],
                location_config['y'], 
                location_config['z']
            )
            
            rotation_config = transform_config['rotation']
            self.camera_rotation = (
                rotation_config['pitch'],
                rotation_config['yaw'],
                rotation_config['roll']
            )
            
            # Camera specifications
            rgb_config = camera_config['attributes']['rgb']
            self.image_width = rgb_config['image_size_x']
            self.image_height = rgb_config['image_size_y']
            self.fov = rgb_config['fov']
            
        except Exception as e:
            self.logger.warning(f"Could not load camera config: {e}, using defaults")
            # EXACT reference parameters from reference/detection/detector.py
            self.camera_position = (0.0, 15.0, 5.0)
            self.camera_rotation = (-10.0, 180.0, 0.0)  # pitch=-10, yaw=180, roll=0
            self.image_width = 960   # VIEW_WIDTH = 1920//2
            self.image_height = 540  # VIEW_HEIGHT = 1080//2
            self.fov = 40.0          # VIEW_FOV = 40
        
        self.logger.info(f"Camera position: {self.camera_position}")
        self.logger.info(f"Camera rotation: {self.camera_rotation}")
        self.logger.info(f"Image size: {self.image_width}x{self.image_height}")
    
    def setup_camera(self) -> bool:
        """Setup RGB camera in CARLA world - EXACT reference implementation"""
        try:
            # Get RGB camera blueprint
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            
            # EXACT reference attributes
            camera_bp.set_attribute('image_size_x', str(self.image_width))
            camera_bp.set_attribute('image_size_y', str(self.image_height))
            camera_bp.set_attribute('fov', str(self.fov))
            camera_bp.set_attribute('sensor_tick', '0.1')  # EXACT: 0.1s = 10Hz
            
            # EXACT reference transform (absolute world position)
            # In reference, camera was at traffic light + (0, 15, 5) offset
            # For Town10HD intersection, use absolute position near center
            camera_transform = carla.Transform(
                carla.Location(x=-50, y=-15, z=8),  # Elevated position overlooking intersection
                carla.Rotation(pitch=-15, yaw=180, roll=0)  # Look down toward traffic
            )
            self.camera_transform = camera_transform
            
            # Try to attach to traffic light if available, otherwise spawn freely
            traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
            if traffic_lights:
                traffic_light = traffic_lights[0]
                self.logger.info(f"Attaching camera to traffic light: {traffic_light.id}")
                
                # Use relative transform from traffic light
                relative_transform = carla.Transform(
                    carla.Location(x=0, y=15, z=5),
                    carla.Rotation(pitch=-10, yaw=180, roll=0)
                )
                
                self.camera = self.world.spawn_actor(
                    camera_bp, 
                    relative_transform, 
                    attach_to=traffic_light
                )
            else:
                # Fallback: spawn camera at absolute position
                self.logger.warning("No traffic lights found - spawning camera at absolute position")
                
                self.camera = self.world.spawn_actor(camera_bp, camera_transform)
            
            if self.camera is None:
                self.logger.error("Failed to spawn camera actor")
                return False
            
            # Register with client for cleanup
            if hasattr(self.carla_client, 'spawned_actors'):
                self.carla_client.spawned_actors.append(self.camera)
            
            self.logger.info(f"✅ Camera spawned successfully at {self.camera_transform.location}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up camera: {e}")
            return False
    
    def start_listening(self, callback: Optional[Callable] = None):
        """Start listening for camera data"""
        if self.camera is None:
            self.logger.error("Camera not setup, cannot start listening")
            return False
        
        self.image_callback = callback
        
        try:
            # Set image processing callback
            self.camera.listen(self._on_image_received)
            self.logger.info("✅ Camera listening started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting camera listen: {e}")
            return False
    
    def _on_image_received(self, carla_image):
        """Process received camera image"""
        try:
            # Convert CARLA image to numpy array
            image_array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
            image_array = image_array.reshape((carla_image.height, carla_image.width, 4))
            
            # Convert BGRA to RGB
            image_rgb = image_array[:, :, :3][:, :, ::-1].copy()
            
            with self.image_lock:
                self.current_image = image_rgb
            
            # Call external callback if provided
            if self.image_callback:
                self.image_callback(image_rgb)
                
        except Exception as e:
            self.logger.error(f"Error processing camera image: {e}")
    
    def get_latest_image(self) -> Optional[np.ndarray]:
        """Get the latest camera image"""
        with self.image_lock:
            return self.current_image.copy() if self.current_image is not None else None
    
    def get_camera_info(self) -> dict:
        """Get camera information"""
        return {
            'position': self.camera_position,
            'rotation': self.camera_rotation,
            'image_size': (self.image_width, self.image_height),
            'fov': self.fov,
            'transform': {
                'location': {
                    'x': self.camera_transform.location.x,
                    'y': self.camera_transform.location.y,
                    'z': self.camera_transform.location.z
                } if self.camera_transform else None,
                'rotation': {
                    'pitch': self.camera_transform.rotation.pitch,
                    'yaw': self.camera_transform.rotation.yaw,
                    'roll': self.camera_transform.rotation.roll
                } if self.camera_transform else None
            }
        }
    
    def stop_listening(self):
        """Stop camera data acquisition"""
        if self.camera:
            try:
                self.camera.stop()
                self.logger.info("Camera listening stopped")
            except Exception as e:
                self.logger.error(f"Error stopping camera: {e}")
    
    def cleanup(self):
        """Clean up camera resources"""
        self.logger.info("Cleaning up camera manager...")
        
        self.stop_listening()
        
        if self.camera:
            try:
                if self.camera.is_alive:
                    self.camera.destroy()
                self.camera = None
                self.logger.info("Camera destroyed")
            except Exception as e:
                self.logger.error(f"Error destroying camera: {e}")
        
        with self.image_lock:
            self.current_image = None
