"""
Route Management for Vehicle Navigation
Handles waypoint generation and path planning
"""

import carla
import random
import math
import json
from typing import List, Optional, Tuple, Dict, Any
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger


class RouteManager:
    """Manages vehicle routes and navigation"""
    
    def __init__(self, world: carla.World):
        self.logger = get_logger("RouteManager")
        self.world = world
        self.map = world.get_map()
        
        # Route storage
        self.active_routes: Dict[int, List[carla.Waypoint]] = {}
        
        # Reference route storage
        self.reference_routes: Dict[str, List[carla.Transform]] = {}
        
        self.logger.info("Route manager initialized")
    
    def load_reference_route(self, route_filename: str) -> Optional[List[carla.Transform]]:
        """Load exact reference route from JSON file"""
        try:
            route_path = Path(__file__).parent.parent.parent / "data" / "routes" / route_filename
            
            if not route_path.exists():
                self.logger.error(f"Route file not found: {route_path}")
                return None
            
            with open(route_path, 'r') as f:
                route_data = json.load(f)
            
            if not isinstance(route_data, list):
                self.logger.error("Route data must be a list of waypoints")
                return None
            
            # Convert JSON data to CARLA transforms
            transforms = []
            for waypoint in route_data:
                if 'location' not in waypoint or 'rotation' not in waypoint:
                    self.logger.warning(f"Skipping invalid waypoint: {waypoint}")
                    continue
                
                location = carla.Location(
                    x=waypoint['location']['x'],
                    y=waypoint['location']['y'],
                    z=waypoint['location']['z']
                )
                
                rotation = carla.Rotation(
                    pitch=waypoint['rotation']['pitch'],
                    yaw=waypoint['rotation']['yaw'],
                    roll=waypoint['rotation']['roll']
                )
                
                transforms.append(carla.Transform(location, rotation))
            
            self.reference_routes[route_filename] = transforms
            
            self.logger.info(f"✅ Loaded reference route '{route_filename}' with {len(transforms)} waypoints")
            
            # Log route details
            if transforms:
                start = transforms[0].location
                end = transforms[-1].location
                distance = self._calculate_total_distance(transforms)
                self.logger.info(f"   Route: ({start.x:.1f}, {start.y:.1f}) → ({end.x:.1f}, {end.y:.1f})")
                self.logger.info(f"   Total distance: {distance:.1f}m")
            
            return transforms
            
        except Exception as e:
            self.logger.error(f"Error loading reference route: {e}")
            return None
    
    def get_reference_route(self, route_filename: str) -> Optional[List[carla.Transform]]:
        """Get loaded reference route or load if not cached"""
        if route_filename not in self.reference_routes:
            return self.load_reference_route(route_filename)
        return self.reference_routes[route_filename]
    
    def _calculate_total_distance(self, transforms: List[carla.Transform]) -> float:
        """Calculate total distance of a route"""
        if len(transforms) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(transforms)):
            distance = self._calculate_distance(
                transforms[i-1].location,
                transforms[i].location
            )
            total_distance += distance
        
        return total_distance
    
    def generate_random_route(self, start_location: carla.Location, 
                            distance_km: float = 2.0) -> List[carla.Waypoint]:
        """Generate random route from starting location"""
        try:
            # Get starting waypoint
            start_waypoint = self.map.get_waypoint(start_location)
            if not start_waypoint:
                self.logger.error("Could not find starting waypoint")
                return []
            
            route = [start_waypoint]
            current_waypoint = start_waypoint
            total_distance = 0.0
            target_distance = distance_km * 1000  # Convert to meters
            
            # Generate waypoints until target distance is reached
            while total_distance < target_distance:
                # Get next waypoints (roads ahead)
                next_waypoints = current_waypoint.next(2.0)  # 2 meter intervals
                
                if not next_waypoints:
                    self.logger.warning("No more waypoints available, ending route")
                    break
                
                # Choose next waypoint (random choice for variety)
                if len(next_waypoints) > 1:
                    # At intersections, make random choice
                    next_waypoint = random.choice(next_waypoints)
                else:
                    next_waypoint = next_waypoints[0]
                
                # Calculate distance to next waypoint
                distance = self._calculate_distance(current_waypoint.transform.location,
                                                  next_waypoint.transform.location)
                
                route.append(next_waypoint)
                total_distance += distance
                current_waypoint = next_waypoint
                
                # Safety check to prevent infinite loops
                if len(route) > 1000:
                    self.logger.warning("Route generation exceeded maximum waypoints")
                    break
            
            self.logger.debug(f"Generated route with {len(route)} waypoints, "
                            f"total distance: {total_distance:.1f}m")
            
            return route
            
        except Exception as e:
            self.logger.error(f"Error generating random route: {e}")
            return []
    
    def generate_loop_route(self, start_location: carla.Location,
                          radius_m: float = 500.0) -> List[carla.Waypoint]:
        """Generate circular/loop route"""
        try:
            start_waypoint = self.map.get_waypoint(start_location)
            if not start_waypoint:
                return []
            
            route = [start_waypoint]
            current_waypoint = start_waypoint
            
            # Try to create a loop by following roads
            visited_locations = set()
            max_attempts = 200
            attempt = 0
            
            while attempt < max_attempts:
                next_waypoints = current_waypoint.next(3.0)
                
                if not next_waypoints:
                    break
                
                # Prefer continuing straight, but allow turns
                best_waypoint = None
                min_angle_diff = float('inf')
                
                for waypoint in next_waypoints:
                    # Calculate angle difference to prefer straight roads
                    angle_diff = abs(current_waypoint.transform.rotation.yaw - 
                                   waypoint.transform.rotation.yaw)
                    angle_diff = min(angle_diff, 360 - angle_diff)  # Normalize to 0-180
                    
                    if angle_diff < min_angle_diff:
                        min_angle_diff = angle_diff
                        best_waypoint = waypoint
                
                if best_waypoint:
                    # Check if we're getting close to start (completing loop)
                    distance_to_start = self._calculate_distance(
                        best_waypoint.transform.location,
                        start_waypoint.transform.location
                    )
                    
                    if distance_to_start < 20.0 and len(route) > 50:
                        # Close enough to start, complete the loop
                        route.append(start_waypoint)
                        self.logger.debug(f"Created loop route with {len(route)} waypoints")
                        return route
                    
                    # Add waypoint to route
                    location_key = (int(best_waypoint.transform.location.x),
                                  int(best_waypoint.transform.location.y))
                    
                    if location_key not in visited_locations:
                        route.append(best_waypoint)
                        visited_locations.add(location_key)
                        current_waypoint = best_waypoint
                
                attempt += 1
            
            # If we couldn't create a perfect loop, return the generated route
            self.logger.debug(f"Generated partial loop route with {len(route)} waypoints")
            return route
            
        except Exception as e:
            self.logger.error(f"Error generating loop route: {e}")
            return []
    
    def create_ambulance_emergency_route(self, start_location: carla.Location) -> List[carla.Waypoint]:
        """Create emergency route for ambulance with priority roads"""
        try:
            # Get major roads and highways for emergency routing
            start_waypoint = self.map.get_waypoint(start_location)
            if not start_waypoint:
                return []
            
            route = []
            current_waypoint = start_waypoint
            
            # Find major roads (highways, arterials)
            for _ in range(100):  # Maximum route length
                next_waypoints = current_waypoint.next(5.0)  # Larger intervals for highways
                
                if not next_waypoints:
                    break
                
                # Prefer major roads for emergency routing
                best_waypoint = None
                best_priority = -1
                
                for waypoint in next_waypoints:
                    priority = self._get_road_priority(waypoint)
                    
                    if priority > best_priority:
                        best_priority = priority
                        best_waypoint = waypoint
                
                if best_waypoint:
                    route.append(best_waypoint)
                    current_waypoint = best_waypoint
                else:
                    break
            
            self.logger.debug(f"Created emergency route with {len(route)} waypoints")
            return route
            
        except Exception as e:
            self.logger.error(f"Error creating emergency route: {e}")
            return []
    
    def _get_road_priority(self, waypoint: carla.Waypoint) -> int:
        """Get road priority for emergency routing"""
        # Higher values = higher priority for emergency vehicles
        if waypoint.is_junction:
            return 1  # Intersections are lower priority
        
        # Check lane type and road type
        lane_type = waypoint.lane_type
        
        if lane_type == carla.LaneType.Driving:
            # Check number of lanes (more lanes = higher priority road)
            lane_width = waypoint.lane_width
            if lane_width > 4.0:  # Wide lanes typically indicate major roads
                return 3
            else:
                return 2
        
        return 0  # Lowest priority
    
    def _calculate_distance(self, loc1: carla.Location, loc2: carla.Location) -> float:
        """Calculate Euclidean distance between two locations"""
        return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2 + (loc1.z - loc2.z)**2)
    
    def register_route(self, vehicle_id: int, route: List[carla.Waypoint]):
        """Register route for a vehicle"""
        self.active_routes[vehicle_id] = route
        self.logger.debug(f"Registered route for vehicle {vehicle_id} with {len(route)} waypoints")
    
    def get_route(self, vehicle_id: int) -> Optional[List[carla.Waypoint]]:
        """Get route for a vehicle"""
        return self.active_routes.get(vehicle_id)
    
    def remove_route(self, vehicle_id: int):
        """Remove route for a vehicle"""
        if vehicle_id in self.active_routes:
            del self.active_routes[vehicle_id]
            self.logger.debug(f"Removed route for vehicle {vehicle_id}")
    
    def get_active_route_count(self) -> int:
        """Get number of active routes"""
        return len(self.active_routes)
    
    def cleanup(self):
        """Clean up route manager"""
        self.logger.info("Cleaning up route manager...")
        self.active_routes.clear()
