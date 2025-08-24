"""
PID Autopilot Controller for RSU Fusion
Based on reference implementation with exact PID parameters
"""

import carla
import math
import time
from typing import List, Optional, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger
from core.config_manager import get_config_manager


class PIDController:
    """Generic PID controller"""
    
    def __init__(self, kp: float, ki: float, kd: float, output_limit: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None
    
    def update(self, error: float, dt: float) -> float:
        """Update PID controller and return control output"""
        # Proportional term
        proportional = self.kp * error
        
        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral
        
        # Derivative term
        if self.previous_time is not None:
            derivative = self.kd * (error - self.previous_error) / dt
        else:
            derivative = 0.0
        
        # PID output
        output = proportional + integral + derivative
        
        # Apply output limits
        output = max(-self.output_limit, min(self.output_limit, output))
        
        # Store for next iteration
        self.previous_error = error
        self.previous_time = time.time()
        
        return output
    
    def reset(self):
        """Reset PID controller state"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None


class AutopilotController:
    """PID-based autopilot controller for vehicle navigation"""
    
    def __init__(self, vehicle: carla.Actor):
        self.logger = get_logger("AutopilotController")
        self.vehicle = vehicle
        self.config_manager = get_config_manager()
        
        # Load control parameters from config
        vehicle_config = self.config_manager.get_module_config('vehicle')
        control_config = vehicle_config['vehicle']['control']
        
        # Initialize PID controllers with reference parameters
        steering_pid = control_config['steering_pid']
        self.steering_controller = PIDController(
            kp=steering_pid['kp'],
            ki=steering_pid['ki'], 
            kd=steering_pid['kd'],
            output_limit=steering_pid.get('output_limit', 1.0)
        )
        
        speed_pid = control_config['speed_pid']
        self.speed_controller = PIDController(
            kp=speed_pid['kp'],
            ki=speed_pid['ki'],
            kd=speed_pid['kd'],
            output_limit=speed_pid.get('output_limit', 1.0)
        )
        
        # Speed settings (from reference)
        self.max_speed = control_config.get('max_speed', 30.0)  # km/h
        self.turn_speed_reduction = control_config.get('turn_speed_reduction', 15.0)  # km/h
        self.reverse_threshold = control_config.get('reverse_threshold', 70.0)  # degrees
        
        self.logger.info(f"Initialized autopilot with reference PID parameters:")
        self.logger.info(f"  Steering PID: kp={steering_pid['kp']}, ki={steering_pid['ki']}, kd={steering_pid['kd']}")
        self.logger.info(f"  Speed PID: kp={speed_pid['kp']}, ki={speed_pid['ki']}, kd={speed_pid['kd']}")
        self.logger.info(f"  Max speed: {self.max_speed} km/h, Turn reduction: {self.turn_speed_reduction} km/h")
    
    def calculate_steering_angle(self, target_location: carla.Location, dt: float) -> float:
        """Calculate steering angle to reach target location"""
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_forward = vehicle_transform.get_forward_vector()
        
        # Vector from vehicle to target
        target_vector = carla.Vector3D(
            x=target_location.x - vehicle_location.x,
            y=target_location.y - vehicle_location.y,
            z=0.0
        )
        
        # Normalize vectors
        target_vector_norm = math.sqrt(target_vector.x**2 + target_vector.y**2)
        if target_vector_norm < 0.01:  # Too close to target
            return 0.0
        
        target_vector.x /= target_vector_norm
        target_vector.y /= target_vector_norm
        
        forward_norm = math.sqrt(vehicle_forward.x**2 + vehicle_forward.y**2)
        vehicle_forward.x /= forward_norm
        vehicle_forward.y /= forward_norm
        
        # Calculate cross product to determine steering direction
        cross_product = vehicle_forward.x * target_vector.y - vehicle_forward.y * target_vector.x
        
        # Calculate angle error
        dot_product = vehicle_forward.x * target_vector.x + vehicle_forward.y * target_vector.y
        angle_error = math.acos(max(-1.0, min(1.0, dot_product)))
        
        # Apply sign based on cross product
        if cross_product < 0:
            angle_error = -angle_error
        
        # Use PID controller
        steering = self.steering_controller.update(angle_error, dt)
        
        return steering
    
    def calculate_target_speed(self, target_location: carla.Location, 
                              next_target_location: Optional[carla.Location] = None) -> float:
        """Calculate target speed based on upcoming turns"""
        base_speed = self.max_speed
        
        if next_target_location is not None:
            # Calculate upcoming turn angle
            vehicle_location = self.vehicle.get_transform().location
            
            # Vector from current to target
            vec1 = carla.Vector3D(
                x=target_location.x - vehicle_location.x,
                y=target_location.y - vehicle_location.y,
                z=0.0
            )
            
            # Vector from target to next target
            vec2 = carla.Vector3D(
                x=next_target_location.x - target_location.x,
                y=next_target_location.y - target_location.y,
                z=0.0
            )
            
            # Calculate angle between vectors
            norm1 = math.sqrt(vec1.x**2 + vec1.y**2)
            norm2 = math.sqrt(vec2.x**2 + vec2.y**2)
            
            if norm1 > 0.01 and norm2 > 0.01:
                dot = (vec1.x * vec2.x + vec1.y * vec2.y) / (norm1 * norm2)
                angle = math.acos(max(-1.0, min(1.0, dot)))
                angle_degrees = math.degrees(angle)
                
                # Reduce speed for sharp turns
                if angle_degrees > self.reverse_threshold:
                    # Very sharp turn - use minimum speed
                    base_speed = self.turn_speed_reduction
                elif angle_degrees > 30.0:
                    # Moderate turn - reduce speed proportionally
                    reduction_factor = min(1.0, angle_degrees / 90.0)
                    base_speed = self.max_speed - (self.turn_speed_reduction * reduction_factor)
        
        return base_speed
    
    def calculate_throttle_brake(self, target_speed_kmh: float, dt: float) -> Tuple[float, float]:
        """Calculate throttle and brake values based on target speed"""
        # Get current speed
        velocity = self.vehicle.get_velocity()
        current_speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        current_speed_kmh = current_speed_ms * 3.6
        
        # Speed error
        speed_error = target_speed_kmh - current_speed_kmh
        
        # Use PID controller
        control_output = self.speed_controller.update(speed_error, dt)
        
        # Convert to throttle/brake
        if control_output > 0:
            # Accelerate
            throttle = min(1.0, control_output)
            brake = 0.0
        else:
            # Brake
            throttle = 0.0
            brake = min(1.0, abs(control_output))
        
        return throttle, brake
    
    def should_reverse(self, target_location: carla.Location) -> bool:
        """Determine if vehicle should reverse based on target angle"""
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_forward = vehicle_transform.get_forward_vector()
        
        # Vector to target
        target_vector = carla.Vector3D(
            x=target_location.x - vehicle_location.x,
            y=target_location.y - vehicle_location.y,
            z=0.0
        )
        
        # Normalize
        target_norm = math.sqrt(target_vector.x**2 + target_vector.y**2)
        if target_norm < 0.01:
            return False
        
        target_vector.x /= target_norm
        target_vector.y /= target_norm
        
        forward_norm = math.sqrt(vehicle_forward.x**2 + vehicle_forward.y**2)
        vehicle_forward.x /= forward_norm
        vehicle_forward.y /= forward_norm
        
        # Calculate angle
        dot_product = vehicle_forward.x * target_vector.x + vehicle_forward.y * target_vector.y
        angle = math.acos(max(-1.0, min(1.0, dot_product)))
        angle_degrees = math.degrees(angle)
        
        return angle_degrees > self.reverse_threshold
    
    def update_control(self, target_location: carla.Location, 
                      next_target_location: Optional[carla.Location] = None,
                      dt: float = 0.05) -> carla.VehicleControl:
        """Calculate vehicle control based on target waypoint"""
        control = carla.VehicleControl()
        
        # Check if should reverse
        if self.should_reverse(target_location):
            # Reverse maneuver
            control.throttle = 0.3
            control.brake = 0.0
            control.steer = -self.calculate_steering_angle(target_location, dt) * 0.5
            control.reverse = True
            control.hand_brake = False
        else:
            # Normal forward driving
            control.reverse = False
            control.hand_brake = False
            
            # Calculate steering
            control.steer = self.calculate_steering_angle(target_location, dt)
            
            # Calculate speed
            target_speed = self.calculate_target_speed(target_location, next_target_location)
            
            # Calculate throttle and brake
            throttle, brake = self.calculate_throttle_brake(target_speed, dt)
            control.throttle = throttle
            control.brake = brake
        
        return control
    
    def reset_controllers(self):
        """Reset PID controller states"""
        self.steering_controller.reset()
        self.speed_controller.reset()
        self.logger.debug("PID controllers reset")