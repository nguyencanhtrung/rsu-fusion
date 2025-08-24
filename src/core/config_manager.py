"""
Configuration Management System for RSU Fusion
Handles loading and validation of YAML configuration files
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from .logger import get_logger


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_dir: str = "config"):
        self.logger = get_logger("ConfigManager")
        self.config_dir = Path(config_dir)
        self.main_config = None
        self.module_configs = {}
        
        # Ensure config directory exists
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")
    
    def load_main_config(self, filename: str = "phase1_config.yaml") -> Dict[str, Any]:
        """Load main configuration file"""
        config_file = self.config_dir / filename
        
        if not config_file.exists():
            raise FileNotFoundError(f"Main configuration file not found: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                self.main_config = yaml.safe_load(f)
            
            self.logger.info(f"Loaded main configuration from {config_file}")
            
            # Load module-specific configs
            if 'modules' in self.main_config:
                self._load_module_configs(self.main_config['modules'])
            
            return self.main_config
            
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML file {config_file}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading configuration {config_file}: {e}")
            raise
    
    def _load_module_configs(self, module_config_files: Dict[str, str]):
        """Load module-specific configuration files"""
        for module_name, config_file in module_config_files.items():
            try:
                config_path = self.config_dir / config_file
                
                if not config_path.exists():
                    self.logger.warning(f"Module config file not found: {config_path}")
                    continue
                
                with open(config_path, 'r') as f:
                    self.module_configs[module_name] = yaml.safe_load(f)
                
                self.logger.debug(f"Loaded {module_name} configuration from {config_file}")
                
            except Exception as e:
                self.logger.error(f"Error loading module config {config_file}: {e}")
                raise
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration section"""
        if self.main_config is None:
            raise RuntimeError("Configuration not loaded. Call load_main_config() first.")
        
        if section is None:
            return self.main_config
        
        if section in self.main_config:
            return self.main_config[section]
        elif section in self.module_configs:
            return self.module_configs[section]
        else:
            raise KeyError(f"Configuration section '{section}' not found")
    
    def get_carla_config(self) -> Dict[str, Any]:
        """Get CARLA-specific configuration"""
        return self.get_config('carla')
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration"""
        return self.get_config('system')
    
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """Get module-specific configuration"""
        if module_name not in self.module_configs:
            raise KeyError(f"Module configuration '{module_name}' not found")
        return self.module_configs[module_name]
    
    def validate_config(self) -> bool:
        """Validate loaded configuration"""
        if self.main_config is None:
            self.logger.error("No configuration loaded")
            return False
        
        required_sections = ['carla', 'system', 'paths']
        
        for section in required_sections:
            if section not in self.main_config:
                self.logger.error(f"Required configuration section '{section}' missing")
                return False
        
        # Validate CARLA config
        carla_config = self.main_config.get('carla', {})
        required_carla_keys = ['host', 'port', 'map']
        
        for key in required_carla_keys:
            if key not in carla_config:
                self.logger.error(f"Required CARLA config key '{key}' missing")
                return False
        
        self.logger.info("Configuration validation passed")
        return True
    
    def get_data_path(self, path_type: str) -> Path:
        """Get data path from configuration"""
        paths_config = self.get_config('paths')
        
        if path_type not in paths_config:
            raise KeyError(f"Path type '{path_type}' not found in configuration")
        
        path = Path(paths_config[path_type])
        path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        
        return path


# Global config manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager