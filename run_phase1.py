#!/usr/bin/env python3
"""
RSU Fusion Phase 1 - Multi-Process Implementation
Main entry point for detection and tracking system
"""

import multiprocessing
import time
import signal
import sys
import queue
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.logger import get_logger
from core.config_manager import get_config_manager
from core.carla_client import create_carla_client_from_config


class Phase1System:
    """Main orchestrator for Phase 1 multi-process system"""
    
    def __init__(self):
        self.logger = get_logger("Phase1System")
        self.processes = {}
        self.shared_data = {}
        self.running = False
        
        # Multi-process communication
        self.manager = multiprocessing.Manager()
        self.detection_image_queue = multiprocessing.Queue(maxsize=5)
        self.visualization_image_queue = multiprocessing.Queue(maxsize=5)
        self.detection_queue = multiprocessing.Queue(maxsize=10)
        self.tracking_queue = multiprocessing.Queue(maxsize=10)
        self.stats_dict = self.manager.dict()
        
        self.logger.info("Phase 1 system initialized")
    
    def start(self):
        """Start all system processes"""
        try:
            self.logger.info("🚀 Starting RSU Fusion Phase 1 System...")
            
            # Load configuration
            config_manager = get_config_manager()
            config_manager.load_main_config()
            
            self.running = True
            
            # Start processes in order
            self._start_camera_process()
            time.sleep(2)  # Allow camera to initialize
            
            self._start_detection_process()
            time.sleep(1)
            
            self._start_tracking_process()
            time.sleep(1)
            
            self._start_visualization_process()
            
            self.logger.info("✅ All processes started successfully")
            
            # Run main control loop
            self._run_main_loop()
            
        except Exception as e:
            self.logger.error(f"Error starting system: {e}")
            self.shutdown()
    
    def _start_camera_process(self):
        """Start camera and CARLA simulation process"""
        def camera_worker():
            try:
                # Set up process-specific logging
                logger = get_logger("CameraProcess")
                logger.info("Camera process started")
                
                # Connect to CARLA
                carla_client = create_carla_client_from_config()
                if not carla_client.connect():
                    logger.error("Failed to connect to CARLA")
                    return
                
                if not carla_client.load_world("Town10HD"):
                    logger.error("Failed to load Town10HD")
                    return
                
                # Setup camera
                from sensors.camera_manager import CameraManager
                camera_manager = CameraManager(carla_client, carla_client.get_world())
                
                if not camera_manager.setup_camera():
                    logger.error("Failed to setup camera")
                    return
                
                # Setup vehicles
                from vehicles.vehicle_spawner import VehicleSpawner
                vehicle_spawner = VehicleSpawner(carla_client, carla_client.get_world())
                vehicle_spawner.load_vehicle_configs()
                
                # Spawn reference ambulance
                vehicle_spawner.spawn_reference_ambulance()
                vehicle_spawner.enable_ambulance_autopilot()
                
                # Spawn traffic vehicles
                vehicle_spawner.spawn_traffic_vehicles(15)
                
                # Start camera listening
                def image_callback(image):
                    try:
                        # Send image to BOTH detection and visualization queues
                        # Detection queue
                        while self.detection_image_queue.full():
                            try:
                                self.detection_image_queue.get_nowait()  # Remove old frame
                            except queue.Empty:
                                break
                        self.detection_image_queue.put_nowait(image)
                        
                        # Visualization queue
                        while self.visualization_image_queue.full():
                            try:
                                self.visualization_image_queue.get_nowait()  # Remove old frame
                            except queue.Empty:
                                break
                        self.visualization_image_queue.put_nowait(image)
                    except:
                        pass  # Queue might be closed during shutdown
                
                camera_manager.start_listening(image_callback)
                
                # Camera process loop
                frame_count = 0
                while self.running:
                    carla_client.tick()  # Advance simulation
                    time.sleep(0.05)  # 20 FPS
                    
                    frame_count += 1
                    if frame_count % 60 == 0:  # Every 3 seconds
                        ambulance_info = vehicle_spawner.get_ambulance_info()
                        if ambulance_info:
                            logger.debug(f"Ambulance speed: {ambulance_info['speed_kmh']:.1f} km/h")
                
                # Cleanup
                logger.info("Cleaning up camera process...")
                camera_manager.cleanup()
                vehicle_spawner.cleanup()
                carla_client.cleanup()
                
            except Exception as e:
                logger.error(f"Camera process error: {e}")
        
        process = multiprocessing.Process(target=camera_worker, name="CameraProcess")
        process.start()
        self.processes['camera'] = process
        self.logger.info("Camera process started")
    
    def _start_detection_process(self):
        """Start YOLO detection process"""
        def detection_worker():
            try:
                logger = get_logger("DetectionProcess")
                logger.info("Detection process started")
                
                # Initialize YOLO detector
                from detection.yolo_detector import YOLODetector
                detector = YOLODetector()
                
                frame_count = 0
                detection_count = 0
                
                while self.running:
                    try:
                        # Get image from detection queue
                        image = self.detection_image_queue.get(timeout=1.0)
                        
                        # Run detection
                        detections = detector.detect(image)
                        detection_count += len(detections)
                        
                        # Convert detections to serializable format
                        serializable_detections = [det.to_dict() for det in detections]
                        
                        # Send to tracking process
                        if not self.detection_queue.full():
                            self.detection_queue.put(serializable_detections)
                        
                        frame_count += 1
                        
                        # Update stats
                        if frame_count % 30 == 0:
                            stats = detector.get_performance_stats()
                            self.stats_dict['detection'] = stats
                            logger.debug(f"Detection: {stats['avg_fps']:.1f} FPS, "
                                       f"{detection_count} detections processed")
                        
                    except queue.Empty:
                        if self.running:
                            time.sleep(0.01)  # Short wait for new images
                        continue
                    except Exception as e:
                        if self.running:  # Only log if not shutting down
                            logger.error(f"Detection error: {e}")
                        time.sleep(0.1)
                
                # Cleanup
                logger.info("Cleaning up detection process...")
                detector.cleanup()
                
            except Exception as e:
                logger.error(f"Detection process error: {e}")
        
        process = multiprocessing.Process(target=detection_worker, name="DetectionProcess")
        process.start()
        self.processes['detection'] = process
        self.logger.info("Detection process started")
    
    def _start_tracking_process(self):
        """Start SORT tracking process"""
        def tracking_worker():
            try:
                logger = get_logger("TrackingProcess")
                logger.info("Tracking process started")
                
                # Initialize SORT tracker
                from tracking.sort_tracker import SORTTracker
                from detection.yolo_detector import Detection
                tracker = SORTTracker()
                
                frame_count = 0
                
                while self.running:
                    try:
                        # Get detections from queue
                        detection_dicts = self.detection_queue.get(timeout=1.0)
                        
                        # Convert back to Detection objects
                        detections = []
                        for det_dict in detection_dicts:
                            detection = Detection(
                                bbox=det_dict['bbox'],
                                confidence=det_dict['confidence'],
                                class_id=det_dict['class_id'],
                                class_name=det_dict['class_name']
                            )
                            detections.append(detection)
                        
                        # Update tracker
                        tracks = tracker.update(detections)
                        
                        # Send to visualization
                        if not self.tracking_queue.full():
                            self.tracking_queue.put({
                                'detections': detection_dicts,
                                'tracks': tracks
                            })
                        
                        frame_count += 1
                        
                        # Update stats
                        if frame_count % 30 == 0:
                            stats = tracker.get_statistics()
                            self.stats_dict['tracking'] = stats
                            logger.debug(f"Tracking: {stats['confirmed_tracks']} confirmed tracks")
                        
                    except queue.Empty:
                        if self.running:
                            time.sleep(0.01)  # Short wait for new detections
                        continue
                    except Exception as e:
                        if self.running:
                            logger.error(f"Tracking error: {e}")
                        time.sleep(0.1)
                
                # Cleanup
                logger.info("Cleaning up tracking process...")
                tracker.cleanup()
                
            except Exception as e:
                logger.error(f"Tracking process error: {e}")
        
        process = multiprocessing.Process(target=tracking_worker, name="TrackingProcess")
        process.start()
        self.processes['tracking'] = process
        self.logger.info("Tracking process started")
    
    def _start_visualization_process(self):
        """Start live visualization process"""
        def visualization_worker():
            try:
                logger = get_logger("VisualizationProcess")
                logger.info("Visualization process started")
                
                # Initialize live viewer
                from visualization.live_viewer import LiveViewer
                viewer = LiveViewer()
                viewer.start()
                
                frame_count = 0
                
                while self.running:
                    try:
                        # Get latest image from visualization queue (consume all available to get the most recent)
                        latest_image = None
                        image_count = 0
                        while True:
                            try:
                                latest_image = self.visualization_image_queue.get_nowait()
                                image_count += 1
                                if image_count > 5:  # Prevent consuming too many at once
                                    break
                            except queue.Empty:
                                break
                        
                        if latest_image is not None:
                            viewer.update_image(latest_image)
                            # Debug: log image updates occasionally
                            if frame_count % 100 == 0:
                                logger.debug(f"Updated image (consumed {image_count} frames)")
                        
                        # Get latest tracking results
                        try:
                            if not self.tracking_queue.empty():
                                tracking_data = self.tracking_queue.get_nowait()
                                viewer.update_detections(tracking_data['detections'])
                                viewer.update_tracks(tracking_data['tracks'])
                        except queue.Empty:
                            pass
                        
                        # Show frame
                        if not viewer.show_frame():
                            logger.info("Visualization window closed by user")
                            self.running = False
                            break
                        
                        frame_count += 1
                        
                        # Update stats
                        if frame_count % 30 == 0:
                            stats = viewer.get_performance_stats()
                            self.stats_dict['visualization'] = stats
                            logger.debug(f"Visualization: {stats['avg_fps']:.1f} FPS")
                        
                        time.sleep(0.01)  # ~100 FPS max
                        
                    except Exception as e:
                        if self.running:
                            logger.error(f"Visualization error: {e}")
                        time.sleep(0.1)
                
                # Cleanup
                logger.info("Cleaning up visualization process...")
                viewer.cleanup()
                
            except Exception as e:
                logger.error(f"Visualization process error: {e}")
        
        process = multiprocessing.Process(target=visualization_worker, name="VisualizationProcess")
        process.start()
        self.processes['visualization'] = process
        self.logger.info("Visualization process started")
    
    def _run_main_loop(self):
        """Main control loop"""
        try:
            start_time = time.time()
            
            while self.running:
                # Monitor process health
                for name, process in self.processes.items():
                    if not process.is_alive():
                        self.logger.error(f"Process {name} has died!")
                        self.running = False
                        break
                
                # Print system stats every 30 seconds
                if int(time.time() - start_time) % 30 == 0 and int(time.time() - start_time) > 0:
                    self._print_system_stats()
                
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
            self.running = False
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
            self.running = False
        
        # Shutdown system
        self.shutdown()
    
    def _print_system_stats(self):
        """Print system performance statistics"""
        self.logger.info("=== System Performance Stats ===")
        
        for process_name, stats in self.stats_dict.items():
            if process_name == 'detection':
                self.logger.info(f"Detection: {stats.get('avg_fps', 0):.1f} FPS")
            elif process_name == 'tracking':
                confirmed = stats.get('confirmed_tracks', 0)
                total = stats.get('total_tracks', 0)
                self.logger.info(f"Tracking: {confirmed}/{total} tracks")
            elif process_name == 'visualization':
                self.logger.info(f"Visualization: {stats.get('avg_fps', 0):.1f} FPS")
        
        # Queue status
        self.logger.info(f"Queue sizes - Detection Images: {self.detection_image_queue.qsize()}, "
                        f"Visualization Images: {self.visualization_image_queue.qsize()}, "
                        f"Detections: {self.detection_queue.qsize()}, "
                        f"Tracking: {self.tracking_queue.qsize()}")
    
    def shutdown(self):
        """Shutdown all processes"""
        self.logger.info("🛑 Shutting down Phase 1 system...")
        
        self.running = False
        
        # Wait for processes to finish gracefully
        for name, process in self.processes.items():
            if process.is_alive():
                self.logger.info(f"Waiting for {name} process to finish...")
                process.join(timeout=5.0)
                
                if process.is_alive():
                    self.logger.warning(f"Force terminating {name} process")
                    process.terminate()
                    process.join(timeout=2.0)
        
        # Clean up queues
        try:
            while not self.detection_image_queue.empty():
                self.detection_image_queue.get_nowait()
            while not self.visualization_image_queue.empty():
                self.visualization_image_queue.get_nowait()
            while not self.detection_queue.empty():
                self.detection_queue.get_nowait()
            while not self.tracking_queue.empty():
                self.tracking_queue.get_nowait()
        except:
            pass
        
        self.logger.info("✅ System shutdown complete")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Received shutdown signal. Cleaning up...")
    if 'system' in globals():
        system.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start system
    system = Phase1System()
    
    try:
        system.start()
    except Exception as e:
        print(f"System startup failed: {e}")
        system.shutdown()
        sys.exit(1)
