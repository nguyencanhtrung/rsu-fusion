"""
4D Kalman Filter for 2D Object Tracking
State vector: [u, v, du, dv] - position and velocity in pixel coordinates
"""

import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger


class KalmanFilter4D:
    """
    4D Kalman Filter for 2D object tracking
    State vector: [u, v, du, dv]
    - u, v: center coordinates in pixels
    - du, dv: velocity in pixels per frame
    """
    
    def __init__(self, dt: float = 1.0):
        self.logger = get_logger("KalmanFilter4D")
        self.dt = dt  # Time step
        
        # State dimension: [u, v, du, dv]
        self.state_dim = 4
        self.measurement_dim = 2  # [u, v]
        
        # State vector
        self.x = np.zeros((self.state_dim, 1))  # [u, v, du, dv]
        
        # State covariance matrix
        self.P = np.eye(self.state_dim) * 1000.0  # High initial uncertainty
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        # Higher values = more responsive to changes
        q = 1.0  # Process noise parameter
        self.Q = np.array([
            [self.dt**4/4, 0, self.dt**3/2, 0],
            [0, self.dt**4/4, 0, self.dt**3/2],
            [self.dt**3/2, 0, self.dt**2, 0],
            [0, self.dt**3/2, 0, self.dt**2]
        ], dtype=np.float32) * q
        
        # Measurement noise covariance
        # Higher values = trust measurements less
        r = 10.0  # Measurement noise parameter
        self.R = np.eye(self.measurement_dim, dtype=np.float32) * r
        
        # Identity matrix
        self.I = np.eye(self.state_dim, dtype=np.float32)
        
        self.initialized = False
    
    def initialize(self, measurement: np.ndarray):
        """Initialize filter with first measurement"""
        if measurement.shape != (2,):
            measurement = measurement.reshape(2)
        
        # Initialize state [u, v, 0, 0]
        self.x[0, 0] = measurement[0]  # u
        self.x[1, 0] = measurement[1]  # v
        self.x[2, 0] = 0.0             # du (initial velocity = 0)
        self.x[3, 0] = 0.0             # dv (initial velocity = 0)
        
        # Reset covariance matrix
        self.P = np.eye(self.state_dim, dtype=np.float32) * 1000.0
        self.P[2, 2] = 100.0  # Lower uncertainty for velocity
        self.P[3, 3] = 100.0
        
        self.initialized = True
        self.logger.debug(f"Kalman filter initialized at ({measurement[0]:.1f}, {measurement[1]:.1f})")
    
    def predict(self) -> np.ndarray:
        """Predict next state"""
        # Predict state: x = F * x
        self.x = np.dot(self.F, self.x)
        
        # Predict covariance: P = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return self.get_position()
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update filter with new measurement"""
        if measurement.shape != (2, 1):
            measurement = measurement.reshape(2, 1)
        
        # Calculate innovation
        z = measurement
        y = z - np.dot(self.H, self.x)  # Innovation
        
        # Innovation covariance
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update state: x = x + K * y
        self.x = self.x + np.dot(K, y)
        
        # Update covariance: P = (I - K * H) * P
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)
        
        return self.get_position()
    
    def get_position(self) -> np.ndarray:
        """Get current position [u, v]"""
        return self.x[:2].flatten()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity [du, dv]"""
        return self.x[2:4].flatten()
    
    def get_state(self) -> np.ndarray:
        """Get full state [u, v, du, dv]"""
        return self.x.flatten()
    
    def is_initialized(self) -> bool:
        """Check if filter is initialized"""
        return self.initialized


def create_kalman_filter(dt: float = 1.0, 
                        process_noise: float = 1.0,
                        measurement_noise: float = 10.0):
    """
    Factory function to create configured Kalman filter
    
    Args:
        dt: Time step between measurements
        process_noise: Process noise parameter (higher = more responsive)
        measurement_noise: Measurement noise parameter (higher = trust measurements less)
    """
    kf = KalmanFilter4D(dt=dt)
    
    # Update noise parameters
    kf.Q *= process_noise
    kf.R *= measurement_noise
    
    return kf
