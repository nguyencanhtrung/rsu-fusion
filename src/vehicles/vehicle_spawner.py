"""
Vehicle Spawner for CARLA Simulation
Handles spawning and management of vehicles with reference ambulance route
"""

import carla
import random
import time
from typing import List, Optional, Dict, Any, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger
from core.config_manager import get_config_manager
from vehicles.route_manager import RouteManager


class VehicleSpawner:
    """Manages vehicle spawning in CARLA world"""
    
    def __init__(self, carla_client, world: carla.World):
        self.logger = get_logger("VehicleSpawner")
        self.carla_client = carla_client
        self.world = world
        self.blueprint_library = world.get_blueprint_library()
        
        # Route manager for loading reference routes
        self.route_manager = RouteManager(world)
        
        # Vehicle management
        self.spawned_vehicles: List[carla.Actor] = []
        self.vehicle_configs = []
        
        # Reference ambulance (priority target for detection)
        self.ambulance: Optional[carla.Actor] = None
        self.ambulance_route: List[carla.Transform] = []
        
        self.logger.info("Vehicle spawner initialized")
    
    def load_vehicle_configs(self) -> bool:
        """Load vehicle configuration from config files"""
        try:
            config_manager = get_config_manager()
            vehicle_config = config_manager.get_module_config('vehicles')['vehicles']
            
            self.vehicle_configs = vehicle_config['spawn_configs']
            self.logger.info(f"Loaded {len(self.vehicle_configs)} vehicle configurations")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not load vehicle config: {e}, using defaults")
            # Default vehicle configuration
            self.vehicle_configs = [
                {
                    'blueprint_filter': 'vehicle.audi.*',
                    'count': 5,
                    'autopilot': True,
                    'color': None
                },
                {
                    'blueprint_filter': 'vehicle.bmw.*',
                    'count': 3,
                    'autopilot': True,
                    'color': None
                },
                {
                    'blueprint_filter': 'vehicle.ford.*',
                    'count': 2,
                    'autopilot': True,
                    'color': None
                }
            ]
            return True
    
    def spawn_reference_ambulance(self) -> bool:
        """Spawn reference ambulance using EXACT reference route"""
        try:
            # Load the EXACT reference route
            self.ambulance_route = self.route_manager.load_reference_route("vehicle.ford.ambulance.json")
            if not self.ambulance_route:
                self.logger.error("Failed to load reference ambulance route")
                return False
            
            # Get EXACT ambulance blueprint from reference
            ambulance_bps = self.blueprint_library.filter('vehicle.ford.ambulance')
            if not ambulance_bps:
                self.logger.error("vehicle.ford.ambulance not found - using backup")
                # Fallback to any ambulance
                ambulance_bps = self.blueprint_library.filter('*ambulance*')
                if not ambulance_bps:
                    # Emergency fallback
                    ambulance_bps = self.blueprint_library.filter('vehicle.carlamotors.firetruck')
            
            if not ambulance_bps:
                self.logger.error("No suitable ambulance blueprint found")
                return False
            
            ambulance_bp = ambulance_bps[0]  # Use first match, not random
            
            # Set color to white for easy detection (reference visibility)
            if ambulance_bp.has_attribute('color'):
                ambulance_bp.set_attribute('color', '255,255,255')  # White for visibility
            
            # Use FIRST waypoint from reference route as spawn location
            spawn_transform = self.ambulance_route[0]
            self.logger.info(f"Spawning ambulance at reference start: {spawn_transform.location}")
            
            # Clear any existing vehicles at spawn location
            self._clear_spawn_area(spawn_transform.location, radius=5.0)
            
            # Try to spawn ambulance at EXACT reference location
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Spawn at EXACT reference route starting location
                    self.ambulance = self.world.spawn_actor(ambulance_bp, spawn_transform)
                    
                    if self.ambulance:
                        # Register with client for cleanup
                        if hasattr(self.carla_client, 'spawned_actors'):
                            self.carla_client.spawned_actors.append(self.ambulance)
                        
                        self.spawned_vehicles.append(self.ambulance)
                        
                        self.logger.info(f"✅ Reference ambulance spawned at EXACT location:")
                        self.logger.info(f"   Location: ({spawn_transform.location.x:.1f}, {spawn_transform.location.y:.1f}, {spawn_transform.location.z:.1f})")
                        self.logger.info(f"   Blueprint: {ambulance_bp.id}")
                        self.logger.info(f"   Route has {len(self.ambulance_route)} waypoints")
                        return True
                        
                except Exception as e:
                    self.logger.warning(f"Reference spawn attempt {attempt + 1} failed: {e}")
                    # Try slight position adjustment
                    spawn_transform.location.z += 0.1 * (attempt + 1)
                    
                    if attempt == max_attempts - 1:
                        self.logger.error("Failed to spawn ambulance at reference location")
                        return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error spawning reference ambulance: {e}")
            return False
    
    def _clear_spawn_area(self, location: carla.Location, radius: float = 5.0):
        """Clear any existing vehicles near spawn location"""
        try:
            vehicles = self.world.get_actors().filter('vehicle.*')
            cleared_count = 0
            
            for vehicle in vehicles:
                distance = location.distance(vehicle.get_location())
                if distance < radius:
                    try:
                        vehicle.destroy()
                        cleared_count += 1
                    except:
                        pass  # Vehicle might already be destroyed
            
            if cleared_count > 0:
                self.logger.info(f"Cleared {cleared_count} vehicles from spawn area")
                
        except Exception as e:
            self.logger.warning(f"Error clearing spawn area: {e}")
    
    def _generate_ambulance_route(self, start_point: carla.Transform):
        """Generate route for reference ambulance"""
        try:
            # Get available spawn points as potential waypoints
            spawn_points = self.world.get_map().get_spawn_points()
            
            # Create a route with 5-8 waypoints
            num_waypoints = random.randint(5, 8)
            route_waypoints = random.sample(spawn_points, min(num_waypoints, len(spawn_points)))
            
            # Ensure route starts from current position
            self.ambulance_route = [start_point] + route_waypoints
            
            self.logger.info(f"Generated ambulance route with {len(self.ambulance_route)} waypoints")
            
        except Exception as e:
            self.logger.error(f"Error generating ambulance route: {e}")
            self.ambulance_route = [start_point]  # Fallback to stationary
    
    def spawn_traffic_vehicles(self, total_vehicles: int = 15) -> int:
        """Spawn background traffic vehicles"""
        try:
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                self.logger.error("No spawn points available for traffic")
                return 0
            
            spawned_count = 0
            available_spawn_points = spawn_points.copy()
            random.shuffle(available_spawn_points)
            
            # Calculate vehicles per config
            remaining_vehicles = total_vehicles
            
            for config in self.vehicle_configs:
                if remaining_vehicles <= 0:
                    break
                
                vehicles_for_this_config = min(config.get('count', 1), remaining_vehicles)
                
                # Get matching blueprints
                blueprint_filter = config.get('blueprint_filter', 'vehicle.*')
                matching_bps = self.blueprint_library.filter(blueprint_filter)
                
                if not matching_bps:
                    self.logger.warning(f"No blueprints found for filter: {blueprint_filter}")
                    continue
                
                # Spawn vehicles for this configuration
                for _ in range(vehicles_for_this_config):
                    if not available_spawn_points:
                        self.logger.warning("No more spawn points available")
                        break
                    
                    # Select blueprint and spawn point
                    blueprint = random.choice(matching_bps)
                    spawn_point = available_spawn_points.pop()
                    
                    # Configure blueprint
                    if blueprint.has_attribute('color') and config.get('color'):
                        blueprint.set_attribute('color', config['color'])
                    
                    # Attempt to spawn vehicle
                    try:
                        # Adjust spawn height
                        spawn_point.location.z += 0.5
                        
                        vehicle = self.world.spawn_actor(blueprint, spawn_point)
                        
                        if vehicle:
                            # Register with client for cleanup
                            if hasattr(self.carla_client, 'spawned_actors'):
                                self.carla_client.spawned_actors.append(vehicle)
                            
                            self.spawned_vehicles.append(vehicle)
                            spawned_count += 1
                            remaining_vehicles -= 1
                            
                            # Enable autopilot if configured
                            if config.get('autopilot', False):
                                vehicle.set_autopilot(True)
                            
                    except Exception as e:
                        self.logger.debug(f"Failed to spawn vehicle: {e}")
                        # Continue with next attempt
            
            self.logger.info(f"✅ Spawned {spawned_count} traffic vehicles")
            return spawned_count
            
        except Exception as e:
            self.logger.error(f"Error spawning traffic vehicles: {e}")
            return 0
    
    def enable_ambulance_autopilot(self) -> bool:
        """Enable autopilot for reference ambulance"""
        if not self.ambulance:
            self.logger.error("No ambulance to enable autopilot for")
            return False
        
        try:
            # Enable basic autopilot
            self.ambulance.set_autopilot(True)
            
            # Set traffic manager settings for more dynamic behavior
            traffic_manager = self.carla_client.client.get_trafficmanager()
            
            # Make ambulance more aggressive (emergency vehicle behavior)
            traffic_manager.ignore_lights_percentage(self.ambulance, 80)  # Ignore 80% of lights
            traffic_manager.ignore_signs_percentage(self.ambulance, 90)   # Ignore 90% of signs
            traffic_manager.vehicle_percentage_speed_difference(self.ambulance, -20)  # 20% faster
            
            self.logger.info("✅ Ambulance autopilot enabled with emergency behavior")
            return True
            
        except Exception as e:
            self.logger.error(f"Error enabling ambulance autopilot: {e}")
            return False
    
    def get_ambulance_info(self) -> Optional[Dict[str, Any]]:
        """Get current ambulance information"""
        if not self.ambulance:
            return None
        
        try:
            transform = self.ambulance.get_transform()
            velocity = self.ambulance.get_velocity()
            
            return {
                'id': self.ambulance.id,
                'type_id': self.ambulance.type_id,
                'location': {
                    'x': transform.location.x,
                    'y': transform.location.y,
                    'z': transform.location.z
                },
                'rotation': {
                    'pitch': transform.rotation.pitch,
                    'yaw': transform.rotation.yaw,
                    'roll': transform.rotation.roll
                },
                'velocity': {
                    'x': velocity.x,
                    'y': velocity.y,
                    'z': velocity.z
                },
                'speed_kmh': 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
            }
            
        except Exception as e:
            self.logger.error(f"Error getting ambulance info: {e}")
            return None
    
    def get_vehicle_count(self) -> int:
        """Get number of spawned vehicles"""
        return len(self.spawned_vehicles)
    
    def cleanup(self):
        """Clean up all spawned vehicles"""
        self.logger.info("Cleaning up spawned vehicles...")
        
        for vehicle in self.spawned_vehicles:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except Exception as e:
                self.logger.error(f"Error destroying vehicle: {e}")
        
        self.spawned_vehicles.clear()
        self.ambulance = None
        self.ambulance_route.clear()
        
        self.logger.info("Vehicle cleanup completed")
