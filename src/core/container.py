"""
Dependency Injection Container

Provides a centralized container for managing application dependencies
and promoting loose coupling between components.
"""

from typing import Dict, Any, Optional, Type
from dataclasses import dataclass
from pathlib import Path

from ..config.settings import AppConfig
from ..ml.inference.video_predictor import VideoPredictor
from ..interface.gui import PlayerGUI
from ..utils.logging import setup_logger
from .exceptions import AppError


@dataclass
class AppState:
    """Application state management"""
    is_running: bool = False
    current_count: int = 0
    session_start_time: Optional[float] = None
    performance_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


class Container:
    """Dependency injection container for RopeJumpCounter"""

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._config: Optional[AppConfig] = None
        self._state: Optional[AppState] = None

    def register_config(self, config: AppConfig):
        """Register application configuration"""
        self._config = config
        self._services['config'] = config

    def register_service(self, name: str, service: Any):
        """Register a service instance"""
        self._services[name] = service

    def register_singleton(self, name: str, service_class: Type, *args, **kwargs):
        """Register a singleton service"""
        if name not in self._singletons:
            self._singletons[name] = service_class(*args, **kwargs)
        self._services[name] = self._singletons[name]

    def get_service(self, name: str) -> Any:
        """Get a registered service"""
        if name not in self._services:
            raise AppError(f"Service '{name}' not registered")
        return self._services[name]

    def get_config(self) -> AppConfig:
        """Get application configuration"""
        if not self._config:
            raise AppError("Configuration not registered")
        return self._config

    def get_state(self) -> AppState:
        """Get application state"""
        if not self._state:
            self._state = AppState()
        return self._state

    def initialize_services(self):
        """Initialize all application services"""
        try:
            # Get configuration
            config = self.get_config()

            # Initialize logger
            logger = setup_logger(
                "RopeJump",
                config.logging.log_dir if config.logging.enabled else None
            )
            self.register_service('logger', logger)

            # Initialize video predictor
            predictor = VideoPredictor(str(config.model.model_path))
            self.register_service('predictor', predictor)

            # Initialize GUI
            gui = PlayerGUI(
                predictor=predictor,
                width=config.camera.width,
                height=config.camera.height,
                fps=config.camera.fps,
                save_path=config.save_video_path
            )
            self.register_service('gui', gui)

            logger.info("All services initialized successfully")

        except Exception as e:
            if 'logger' in self._services:
                self._services['logger'].error(f"Service initialization failed: {e}")
            raise AppError(f"Service initialization failed: {e}")

    def cleanup(self):
        """Cleanup all services"""
        try:
            if 'logger' in self._services:
                self._services['logger'].info("Cleaning up services")

            # Cleanup specific services if needed
            if 'predictor' in self._services:
                self._services['predictor'].reset()

            self._services.clear()
            self._singletons.clear()

        except Exception as e:
            if 'logger' in self._services:
                self._services['logger'].error(f"Cleanup failed: {e}")


# Global container instance
container = Container()


def get_container() -> Container:
    """Get the global container instance"""
    return container
