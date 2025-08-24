"""
CARLA Client for RSU Fusion System
Manages connection to CARLA simulator and world setup
Based on reference but with modular architecture
"""

import carla
import time
import threading
from typing import Optional, List, Callable
from .logger import get_logger
from .config_manager import get_config_manager


class CarlaClient:
    """CARLA client with synchronous mode and Town10HD setup"""
    
    def __init__(self, host: str = "localhost", port: int = 2000):
        self.logger = get_logger("CarlaClient")
        self.host = host
        self.port = port
        self.timeout = 10.0
        
        # CARLA objects
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.blueprint_library: Optional[carla.BlueprintLibrary] = None
        self.spawn_points: List[carla.Transform] = []
        
        # Synchronous mode settings
        self.synchronous_mode = True
        self.fixed_delta_seconds = 0.05  # 20 FPS
        self.original_settings: Optional[carla.WorldSettings] = None
        
        # Actor management
        self.spawned_actors: List[carla.Actor] = []
        self.cleanup_lock = threading.Lock()
        
        # Connection state
        self.connected = False
        self.world_loaded = False
    
    def connect(self) -> bool:
        """Connect to CARLA server"""
        try:
            self.logger.info(f"Connecting to CARLA server at {self.host}:{self.port}")
            
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            
            # Test connection
            server_version = self.client.get_server_version()
            client_version = self.client.get_client_version()
            
            self.logger.info(f"Connected to CARLA server version: {server_version}")
            self.logger.info(f"Client version: {client_version}")
            
            # Get world reference
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            
            self.connected = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to CARLA server: {e}")
            self.connected = False
            return False
    
    def load_world(self, map_name: str = "Town10HD") -> bool:
        """Load specified CARLA world/map"""
        if not self.connected:
            self.logger.error("Not connected to CARLA server")
            return False
        
        try:
            self.logger.info(f"Loading world: {map_name}")
            
            # Store original settings for cleanup
            self.original_settings = self.world.get_settings()
            
            # Load the map
            current_map = self.world.get_map().name
            if current_map != map_name:
                self.logger.info(f"Switching from {current_map} to {map_name}")
                self.world = self.client.load_world(map_name)
                self.blueprint_library = self.world.get_blueprint_library()
                
                # Wait for world to be ready
                time.sleep(2.0)
            else:
                self.logger.info(f"Already in {map_name}")
            
            # Setup synchronous mode
            self._setup_synchronous_mode()
            
            # Get spawn points
            self.spawn_points = self.world.get_map().get_spawn_points()
            self.logger.info(f"Found {len(self.spawn_points)} spawn points")
            
            self.world_loaded = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load world {map_name}: {e}")
            self.world_loaded = False
            return False
    
    def _setup_synchronous_mode(self):
        """Configure synchronous simulation mode"""
        try:
            settings = self.world.get_settings()
            
            if self.synchronous_mode:
                self.logger.info("Enabling synchronous mode")
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.fixed_delta_seconds
                settings.no_rendering_mode = False  # Enable rendering for visualization
            else:
                self.logger.info("Disabling synchronous mode")
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
            
            self.world.apply_settings(settings)
            
            # In synchronous mode, we need to tick the world manually
            if self.synchronous_mode:
                self.world.tick()
                
            self.logger.info(f"Synchronous mode: {settings.synchronous_mode}, "
                           f"Fixed delta: {settings.fixed_delta_seconds}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup synchronous mode: {e}")
            raise
    
    def tick(self):
        """Advance simulation by one tick (synchronous mode only)"""
        if self.synchronous_mode and self.world:
            self.world.tick()
    
    def get_world(self) -> Optional[carla.World]:
        """Get world reference"""
        return self.world
    
    def get_blueprint_library(self) -> Optional[carla.BlueprintLibrary]:
        """Get blueprint library"""
        return self.blueprint_library
    
    def get_spawn_points(self) -> List[carla.Transform]:
        """Get available spawn points"""
        return self.spawn_points
    
    def spawn_actor(self, blueprint: carla.ActorBlueprint, 
                   transform: carla.Transform) -> Optional[carla.Actor]:
        """Spawn actor and track for cleanup"""
        try:
            actor = self.world.spawn_actor(blueprint, transform)
            
            with self.cleanup_lock:
                self.spawned_actors.append(actor)
            
            self.logger.debug(f"Spawned actor: {actor.type_id} at {transform.location}")
            return actor
            
        except Exception as e:
            self.logger.error(f"Failed to spawn actor: {e}")
            return None
    
    def destroy_actor(self, actor: carla.Actor):
        """Destroy actor and remove from tracking"""
        try:
            if actor.is_alive:
                actor.destroy()
            
            with self.cleanup_lock:
                if actor in self.spawned_actors:
                    self.spawned_actors.remove(actor)
                    
            self.logger.debug(f"Destroyed actor: {actor.type_id}")
            
        except Exception as e:
            self.logger.error(f"Error destroying actor: {e}")
    
    def cleanup(self):
        """Clean up all spawned actors and restore settings"""
        self.logger.info("Cleaning up CARLA client...")
        
        # Destroy all spawned actors
        with self.cleanup_lock:
            actors_to_destroy = self.spawned_actors.copy()
        
        for actor in actors_to_destroy:
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception as e:
                self.logger.error(f"Error destroying actor during cleanup: {e}")
        
        self.spawned_actors.clear()
        
        # Restore original settings
        if self.original_settings and self.world:
            try:
                self.world.apply_settings(self.original_settings)
                self.logger.info("Restored original world settings")
            except Exception as e:
                self.logger.error(f"Error restoring world settings: {e}")
        
        self.logger.info("CARLA client cleanup completed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()


def create_carla_client_from_config() -> CarlaClient:
    """Create CARLA client using configuration"""
    config_manager = get_config_manager()
    carla_config = config_manager.get_carla_config()
    
    client = CarlaClient(
        host=carla_config.get('host', 'localhost'),
        port=carla_config.get('port', 2000)
    )
    
    # Update synchronous mode settings from config
    client.synchronous_mode = carla_config.get('synchronous_mode', True)
    client.fixed_delta_seconds = carla_config.get('fixed_delta_seconds', 0.05)
    client.timeout = carla_config.get('timeout', 10.0)
    
    return client