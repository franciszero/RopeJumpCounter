#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced main application entry point for RopeJumpCounter v2.0

This module provides the new architecture-based application logic with
dependency injection, event bus, and plugin system integration.

Features:
- Dependency injection container for service management
- Event bus for decoupled component communication
- Plugin system for modular extensions
- Comprehensive error handling and logging
- GPU acceleration setup with mixed precision
"""

import sys
import os
import asyncio
import logging
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ..config.settings import AppConfig
from ..ml.inference.video_predictor import VideoPredictor
from ..interface.gui import PlayerGUI
from ..utils.logging import setup_logger
from ..core.exceptions import AppError
from ..core.container import get_container, Container
from ..core.event_bus import get_event_bus, EventType
from ..core.plugin_manager import get_plugin_manager

def setup_gpu():
    # Enable mixed precision for better performance
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus, 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 必须在任何TF相关import之前调用
setup_gpu()

def setup_event_handlers(container: Container):
    """Setup event handlers for application events"""
    event_bus = get_event_bus()
    
    def on_jump_detected(event):
        """Handle jump detected events"""
        state = container.get_state()
        state.current_count = event.data.get('count', 0)
        logger = container.get_service('logger')
        logger.info(f"Jump detected! Count: {state.current_count}")
    
    def on_error_occurred(event):
        """Handle error events"""
        logger = container.get_service('logger')
        logger.error(f"Error occurred: {event.data.get('error', 'Unknown error')}")
    
    def on_performance_update(event):
        """Handle performance update events"""
        state = container.get_state()
        state.performance_metrics.update(event.data)
    
    # Register event handlers
    event_bus.subscribe(EventType.JUMP_DETECTED, on_jump_detected)
    event_bus.subscribe(EventType.ERROR_OCCURRED, on_error_occurred)
    event_bus.subscribe(EventType.PERFORMANCE_UPDATE, on_performance_update)


async def main_async():
    container = get_container()
    event_bus = get_event_bus()
    plugin_manager = get_plugin_manager()
    
    try:
        # 1. Load application configuration
        config = AppConfig.load()
        container.register_config(config)
        
        # 2. Initialize services
        container.initialize_services()
        logger = container.get_service('logger')
        logger.info("Application starting up with new architecture")
        
        # 3. Setup event handlers
        setup_event_handlers(container)
        
        # 4. Start event bus
        event_bus.start()
        logger.info("Event bus started")
        
        # 5. Load and enable plugins
        loaded_plugins = plugin_manager.load_all_plugins()
        for plugin_name in loaded_plugins:
            if plugin_manager.enable_plugin(plugin_name):
                logger.info(f"Plugin {plugin_name} enabled")
        
        # 6. Publish application start event
        event_bus.publish(EventType.APPLICATION_START, {}, "main")
        
        # 7. Get GUI and run
        gui = container.get_service('gui')
        
        # Update application state
        state = container.get_state()
        state.is_running = True
        state.session_start_time = asyncio.get_event_loop().time()
        
        # Run GUI (this will block until GUI is closed)
        gui.run()
        
    except AppError as e:
        if 'logger' in container._services:
            container.get_service('logger').error(f"Application error: {e}")
        event_bus.publish(EventType.ERROR_OCCURRED, {"error": str(e)}, "main")
        sys.exit(1)
    except Exception as e:
        if 'logger' in container._services:
            container.get_service('logger').exception(f"Unexpected error: {e}")
        event_bus.publish(EventType.ERROR_OCCURRED, {"error": str(e)}, "main")
        sys.exit(1)
    finally:
        # Cleanup
        try:
            # Publish application stop event
            event_bus.publish(EventType.APPLICATION_STOP, {}, "main")
            
            # Stop event bus
            event_bus.stop()
            
            # Cleanup plugins
            plugin_manager.cleanup()
            
            # Cleanup container
            container.cleanup()
            
            if 'logger' in container._services:
                container.get_service('logger').info("Application shutdown complete")
                
        except Exception as e:
            if 'logger' in container._services:
                container.get_service('logger').error(f"Cleanup error: {e}")


def main():
    """Main application entry point

    Orchestrates the complete application startup sequence using the new
    architecture with dependency injection, event bus, and plugin system.

    The function handles all major error scenarios and provides
    appropriate logging and exit codes.
    """
    # Run the async main function
    asyncio.run(main_async())


if __name__ == "__main__":
    main() 