"""
Centralized logging system for RSU Fusion
Provides consistent logging across all modules
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import os


class RSULogger:
    """Centralized logger for RSU Fusion system"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_logger()
            RSULogger._initialized = True
    
    def _setup_logger(self):
        """Setup logging configuration"""
        # Create logs directory
        log_dir = Path("data/outputs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"rsu_fusion_{timestamp}.log"
        
        # Configure root logger
        self.logger = logging.getLogger("RSUFusion")
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(name)-12s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"RSU Fusion logging initialized - Log file: {log_file}")
    
    def get_logger(self, name=None):
        """Get a logger instance for a specific module"""
        if name:
            return logging.getLogger(f"RSUFusion.{name}")
        return self.logger
    
    def set_level(self, level):
        """Set logging level"""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.logger.setLevel(level)
        
        # Update all handlers
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(level)


# Global logger instance
_rsu_logger = RSULogger()

def get_logger(name=None):
    """Get logger instance for module"""
    return _rsu_logger.get_logger(name)

def set_log_level(level):
    """Set global logging level"""
    _rsu_logger.set_level(level)