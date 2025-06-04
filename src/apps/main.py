#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main application entry point for RopeJumpCounter

This module provides the primary entry point for the jump rope counting application.
It handles configuration loading, GPU setup, model initialization, and GUI startup
with comprehensive error handling and logging.

Features:
- Automatic configuration loading from files or environment
- GPU acceleration setup with mixed precision
- Model loading and validation
- GUI initialization and startup
- Comprehensive error handling and logging
"""

import sys
import tensorflow as tf
from tensorflow.keras import mixed_precision

from src.config.settings import AppConfig
from src.core.video_predictor import VideoPredictor
from src.interface.gui import PlayerGUI
from src.utils.logging import setup_logger
from src.core.exceptions import AppError


def setup_gpu():
    """Configure GPU acceleration for optimal performance

    Sets up TensorFlow GPU configuration with mixed precision training
    and memory growth to optimize performance and memory usage.

    Features:
    - Mixed precision (float16) for faster inference
    - Memory growth to prevent GPU memory allocation issues
    - Multi-GPU support with proper device visibility
    """
    # Enable mixed precision for better performance
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # Configure GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus, 'GPU')

    # Enable memory growth to prevent allocation issues
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    """Main application entry point

    Orchestrates the complete application startup sequence including
    configuration loading, logging setup, GPU configuration, model
    initialization, and GUI startup.

    The function handles all major error scenarios and provides
    appropriate logging and exit codes.
    """
    try:
        # 1. Load application configuration
        config = AppConfig.load()

        # 2. Initialize logging system
        logger = setup_logger("RopeJump", config.logging.log_dir if config.logging.enabled else None)
        logger.info("Application starting up")

        # 3. Configure GPU acceleration
        setup_gpu()
        logger.info("GPU configuration completed")

        # 4. Initialize machine learning model
        predictor = VideoPredictor(str(config.model.model_path))
        logger.info("Model loading completed")

        # 5. Start graphical user interface
        gui = PlayerGUI(
            predictor=predictor,
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
            save_path=config.save_video_path
        )
        gui.run()

    except AppError as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        logger.info("Application shutdown")

if __name__ == "__main__":
    main()