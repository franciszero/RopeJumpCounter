"""
Application configuration management module

This module provides comprehensive configuration management for the RopeJumpCounter
application, including YAML file loading, environment variable support,
and configuration validation.
"""

import os
import yaml
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from ..core.exceptions import ConfigError
from ..ml.data.builders.feature_mode import mode_to_str, get_feature_mode

from ..utils.common.FrameSample import SELECTED_LM


@dataclass
class LogConfig:
    """Logging configuration settings

    Controls application logging behavior including output directory,
    log level, and whether logging is enabled.
    """
    enabled: bool = True
    log_dir: Path = Path("logs")
    level: str = "INFO"


@dataclass
class CameraConfig:
    """Camera capture configuration settings

    Defines video capture parameters including resolution, frame rate,
    and device selection for camera input.
    """
    width: int = 640
    height: int = 480
    fps: int = 30
    device_index: int = 0


@dataclass
class ModelConfig:
    """Machine learning model configuration settings

    Specifies model file selection, inference parameters, and
    automatically constructs model paths based on feature configuration.
    """
    model_name: str = "best_cnn8_ws4_withT.keras"
    threshold: float = 0.5

    @property
    def model_path(self) -> Path:
        """Construct full model file path

        Builds the complete path to the model file based on the current
        feature configuration and selected landmarks.

        Returns:
            Path: Complete path to the model file
        """
        model_dir = f"models_{len(SELECTED_LM)}_{mode_to_str(get_feature_mode())}"
        return Path("model_files") / model_dir / self.model_name

@dataclass
class AppConfig:
    """Main application configuration container

    Aggregates all configuration sections and provides loading methods
    for YAML files and environment variables with fallback defaults.
    """
    camera: CameraConfig = CameraConfig()
    model: ModelConfig = ModelConfig()
    logging: LogConfig = LogConfig()
    save_video_path: str | None = None

    @classmethod
    def load(cls) -> 'AppConfig':
        """Load configuration from file or environment variables

        Attempts to load configuration in the following order:
        1. YAML configuration file (config.yaml or APP_CONFIG env var)
        2. Environment variables with fallback to defaults

        Returns:
            AppConfig: Loaded configuration instance

        Raises:
            ConfigError: If configuration file exists but cannot be parsed
        """
        # 1. Try loading from configuration file
        config_path = os.getenv('APP_CONFIG', 'config.yaml')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
                return cls._from_dict(config_dict)
            except Exception as e:
                raise ConfigError(f"Configuration file loading failed: {e}")

        # 2. Load from environment variables with defaults
        return cls(
            camera=CameraConfig(
                width=int(os.getenv('CAMERA_WIDTH', 640)),
                height=int(os.getenv('CAMERA_HEIGHT', 480)),
                fps=int(os.getenv('CAMERA_FPS', 30)),
                device_index=int(os.getenv('CAMERA_DEVICE', 0))
            ),
            model=ModelConfig(
                model_name=os.getenv('MODEL_NAME', "best_cnn8_ws4_withT.keras"),
                threshold=float(os.getenv('MODEL_THRESHOLD', 0.5))
            ),
            logging=LogConfig(
                enabled=os.getenv('LOG_ENABLED', 'true').lower() == 'true',
                log_dir=Path(os.getenv('LOG_DIR', 'logs')),
                level=os.getenv('LOG_LEVEL', 'INFO')
            ),
            save_video_path=os.getenv('SAVE_VIDEO_PATH')
        )

    @classmethod
    def _from_dict(cls, config_dict: dict) -> 'AppConfig':
        """Create AppConfig from dictionary with nested object construction

        Args:
            config_dict: Configuration dictionary from YAML or other source

        Returns:
            AppConfig: Constructed configuration instance
        """
        # Handle nested configuration objects
        camera_config = CameraConfig(**config_dict.get('camera', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        logging_config = LogConfig(**config_dict.get('logging', {}))

        return cls(
            camera=camera_config,
            model=model_config,
            logging=logging_config,
            save_video_path=config_dict.get('save_video_path')
        )