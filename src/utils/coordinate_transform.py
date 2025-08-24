"""
Coordinate Transformation Utilities
Handles transformations between different coordinate systems (CARLA 3D, camera 2D, world coordinates)
"""

import numpy as np
import carla
from typing import Tuple, List, Optional, Dict, Any
import math
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger


class CoordinateTransform:
    """Utility class for coordinate system transformations"""
    
    def __init__(self):
        self.logger = get_logger("CoordinateTransform")
        
        # Camera intrinsic parameters (will be set when camera is available)
        self.camera_intrinsics = None
        self.camera_transform = None
        self.image_width = 960
        self.image_height = 540
        
        # Cached transformation matrices
        self._world_to_camera_matrix = None
        self._camera_to_world_matrix = None
        
        self.logger.debug("Coordinate transform utility initialized")
    
    def set_camera_parameters(self, camera_actor: carla.Actor, 
                            image_width: int, image_height: int) -> None:
        """Set camera parameters for coordinate transformations"""
        try:
            self.image_width = image_width
            self.image_height = image_height
            self.camera_transform = camera_actor.get_transform()
            
            # Get camera attributes
            camera_bp = camera_actor.get_blueprint_library().find('sensor.camera.rgb')
            fov = float(camera_actor.get_attribute('fov').as_float())
            
            # Calculate intrinsic parameters
            self.camera_intrinsics = self._calculate_camera_intrinsics(
                image_width, image_height, fov
            )
            
            # Calculate transformation matrices
            self._update_transformation_matrices()
            
            self.logger.info(f"Camera parameters set: {image_width}x{image_height}, FOV: {fov}°")
            
        except Exception as e:
            self.logger.error(f"Error setting camera parameters: {e}")
    
    def world_to_camera_2d(self, world_point: carla.Location) -> Optional[Tuple[int, int]]:
        """Transform 3D world point to 2D camera coordinates"""
        try:
            if self.camera_transform is None or self.camera_intrinsics is None:
                self.logger.warning("Camera parameters not set")
                return None
            
            # Convert CARLA location to homogeneous coordinates
            world_point_homo = np.array([
                world_point.x,
                world_point.y,
                world_point.z,
                1.0
            ])
            
            # Transform to camera coordinates
            camera_point = self._world_to_camera_matrix @ world_point_homo
            
            # Check if point is behind camera
            if camera_point[2] <= 0:
                return None
            
            # Project to image coordinates
            image_point = self.camera_intrinsics @ camera_point[:3]
            
            # Convert to pixel coordinates
            if image_point[2] != 0:
                u = int(image_point[0] / image_point[2])
                v = int(image_point[1] / image_point[2])
                
                # Check if point is within image bounds
                if 0 <= u < self.image_width and 0 <= v < self.image_height:
                    return (u, v)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in world to camera transform: {e}")
            return None
    
    def camera_2d_to_world_ray(self, pixel_u: int, pixel_v: int) -> Optional[Tuple[carla.Location, carla.Vector3D]]:
        """Convert 2D pixel coordinates to 3D ray in world coordinates"""
        try:
            if self.camera_transform is None or self.camera_intrinsics is None:
                return None
            
            # Convert pixel to normalized camera coordinates
            camera_point = np.array([
                (pixel_u - self.camera_intrinsics[0, 2]) / self.camera_intrinsics[0, 0],
                (pixel_v - self.camera_intrinsics[1, 2]) / self.camera_intrinsics[1, 1],
                1.0
            ])
            
            # Transform to world coordinates
            world_direction = self._camera_to_world_matrix[:3, :3] @ camera_point
            world_direction = world_direction / np.linalg.norm(world_direction)
            
            # Ray origin is camera location
            ray_origin = self.camera_transform.location
            ray_direction = carla.Vector3D(
                world_direction[0], 
                world_direction[1], 
                world_direction[2]
            )
            
            return (ray_origin, ray_direction)
            
        except Exception as e:
            self.logger.error(f"Error in camera 2D to world ray: {e}")
            return None
    
    def estimate_world_position_from_bbox(self, bbox: List[float], 
                                        assumed_height: float = 2.0) -> Optional[carla.Location]:
        """Estimate 3D world position from 2D bounding box"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Use bottom center of bounding box as ground contact point
            bottom_center_u = int((x1 + x2) / 2)
            bottom_center_v = int(y2)  # Bottom of bbox
            
            # Get ray from bottom center
            ray_result = self.camera_2d_to_world_ray(bottom_center_u, bottom_center_v)
            if ray_result is None:
                return None
            
            ray_origin, ray_direction = ray_result
            
            # Intersect ray with ground plane (assuming z=0 is ground)
            # Ray equation: P = Origin + t * Direction
            # Ground plane: z = 0
            # Solve for t: 0 = Origin.z + t * Direction.z
            
            if abs(ray_direction.z) < 1e-6:  # Ray parallel to ground
                return None
            
            t = -ray_origin.z / ray_direction.z
            
            if t < 0:  # Intersection behind camera
                return None
            
            # Calculate world position
            world_x = ray_origin.x + t * ray_direction.x
            world_y = ray_origin.y + t * ray_direction.y
            world_z = assumed_height / 2  # Center height of vehicle
            
            return carla.Location(world_x, world_y, world_z)
            
        except Exception as e:
            self.logger.error(f"Error estimating world position: {e}")
            return None
    
    def transform_bbox_to_world_bounds(self, bbox: List[float], 
                                     vehicle_dimensions: Tuple[float, float, float] = (4.5, 2.0, 2.0)) -> Optional[Dict[str, carla.Location]]:
        """Transform 2D bounding box to estimated 3D world bounds"""
        try:
            length, width, height = vehicle_dimensions
            
            # Get center position estimate
            center_pos = self.estimate_world_position_from_bbox(bbox, height)
            if center_pos is None:
                return None
            
            # Estimate orientation (simplified - assumes vehicles face forward/backward)
            # In a more complete system, this would use appearance or motion cues
            
            # Create bounding box corners
            half_length = length / 2
            half_width = width / 2
            half_height = height / 2
            
            bounds = {
                'center': center_pos,
                'front_left': carla.Location(
                    center_pos.x + half_length,
                    center_pos.y - half_width,
                    center_pos.z
                ),
                'front_right': carla.Location(
                    center_pos.x + half_length,
                    center_pos.y + half_width,
                    center_pos.z
                ),
                'rear_left': carla.Location(
                    center_pos.x - half_length,
                    center_pos.y - half_width,
                    center_pos.z
                ),
                'rear_right': carla.Location(
                    center_pos.x - half_length,
                    center_pos.y + half_width,
                    center_pos.z
                ),
                'top_center': carla.Location(
                    center_pos.x,
                    center_pos.y,
                    center_pos.z + half_height
                ),
                'bottom_center': carla.Location(
                    center_pos.x,
                    center_pos.y,
                    center_pos.z - half_height
                )
            }
            
            return bounds
            
        except Exception as e:
            self.logger.error(f"Error transforming bbox to world bounds: {e}")
            return None
    
    def pixel_distance_to_world_distance(self, pixel_distance: float, 
                                       world_depth: float) -> float:
        """Convert pixel distance to world distance at given depth"""
        try:
            if self.camera_intrinsics is None:
                return pixel_distance
            
            # Use focal length in pixels
            focal_length = self.camera_intrinsics[0, 0]  # fx
            
            # World distance = pixel_distance * world_depth / focal_length
            world_distance = pixel_distance * world_depth / focal_length
            
            return world_distance
            
        except Exception as e:
            self.logger.error(f"Error converting pixel to world distance: {e}")
            return pixel_distance
    
    def calculate_object_real_size(self, bbox: List[float], 
                                 world_position: carla.Location) -> Tuple[float, float]:
        """Calculate real-world size of object from 2D bbox and world position"""
        try:
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Calculate distance from camera to object
            camera_pos = self.camera_transform.location
            distance = math.sqrt(
                (world_position.x - camera_pos.x) ** 2 +
                (world_position.y - camera_pos.y) ** 2 +
                (world_position.z - camera_pos.z) ** 2
            )
            
            # Convert pixel dimensions to world dimensions
            real_width = self.pixel_distance_to_world_distance(bbox_width, distance)
            real_height = self.pixel_distance_to_world_distance(bbox_height, distance)
            
            return (real_width, real_height)
            
        except Exception as e:
            self.logger.error(f"Error calculating object real size: {e}")
            return (0.0, 0.0)
    
    def _calculate_camera_intrinsics(self, width: int, height: int, fov: float) -> np.ndarray:
        """Calculate camera intrinsic matrix from parameters"""
        try:
            # Convert FOV to focal length
            focal_length = width / (2.0 * np.tan(fov * np.pi / 360.0))
            
            # Principal point at image center
            cx = width / 2.0
            cy = height / 2.0
            
            # Intrinsic matrix
            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ])
            
            return K
            
        except Exception as e:
            self.logger.error(f"Error calculating camera intrinsics: {e}")
            return np.eye(3)
    
    def _update_transformation_matrices(self):
        """Update world-to-camera and camera-to-world transformation matrices"""
        try:
            if self.camera_transform is None:
                return
            
            # Camera transform
            camera_location = self.camera_transform.location
            camera_rotation = self.camera_transform.rotation
            
            # Convert rotation to radians
            pitch = math.radians(camera_rotation.pitch)
            yaw = math.radians(camera_rotation.yaw)
            roll = math.radians(camera_rotation.roll)
            
            # Create rotation matrix (CARLA uses left-handed coordinate system)
            cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            cos_roll, sin_roll = math.cos(roll), math.sin(roll)
            
            # Rotation matrix from world to camera
            R = np.array([
                [cos_yaw * cos_pitch, sin_yaw * cos_pitch, -sin_pitch],
                [-sin_yaw * cos_roll + cos_yaw * sin_pitch * sin_roll,
                 cos_yaw * cos_roll + sin_yaw * sin_pitch * sin_roll,
                 cos_pitch * sin_roll],
                [sin_yaw * sin_roll + cos_yaw * sin_pitch * cos_roll,
                 -cos_yaw * sin_roll + sin_yaw * sin_pitch * cos_roll,
                 cos_pitch * cos_roll]
            ])
            
            # Translation vector
            t = np.array([
                -R[0, 0] * camera_location.x - R[0, 1] * camera_location.y - R[0, 2] * camera_location.z,
                -R[1, 0] * camera_location.x - R[1, 1] * camera_location.y - R[1, 2] * camera_location.z,
                -R[2, 0] * camera_location.x - R[2, 1] * camera_location.y - R[2, 2] * camera_location.z
            ])
            
            # World to camera transformation matrix
            self._world_to_camera_matrix = np.eye(4)
            self._world_to_camera_matrix[:3, :3] = R
            self._world_to_camera_matrix[:3, 3] = t
            
            # Camera to world transformation matrix (inverse)
            self._camera_to_world_matrix = np.eye(4)
            self._camera_to_world_matrix[:3, :3] = R.T
            self._camera_to_world_matrix[:3, 3] = np.array([
                camera_location.x,
                camera_location.y,
                camera_location.z
            ])
            
        except Exception as e:
            self.logger.error(f"Error updating transformation matrices: {e}")
    
    def get_transformation_info(self) -> Dict[str, Any]:
        """Get information about current transformations"""
        info = {
            'camera_set': self.camera_transform is not None,
            'intrinsics_set': self.camera_intrinsics is not None,
            'image_size': (self.image_width, self.image_height)
        }
        
        if self.camera_transform is not None:
            info['camera_location'] = {
                'x': self.camera_transform.location.x,
                'y': self.camera_transform.location.y,
                'z': self.camera_transform.location.z
            }
            info['camera_rotation'] = {
                'pitch': self.camera_transform.rotation.pitch,
                'yaw': self.camera_transform.rotation.yaw,
                'roll': self.camera_transform.rotation.roll
            }
        
        if self.camera_intrinsics is not None:
            info['focal_length'] = float(self.camera_intrinsics[0, 0])
            info['principal_point'] = {
                'cx': float(self.camera_intrinsics[0, 2]),
                'cy': float(self.camera_intrinsics[1, 2])
            }
        
        return info