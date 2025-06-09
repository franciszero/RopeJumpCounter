#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RopeJumpCounter main application entry point

Simple entry point that uses the configured version from src/apps/main.py
This file provides backward compatibility and easy access to the main application.
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
    """Configure GPU acceleration for optimal performance"""
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus, 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    try:
        # 1. Load configuration
        config = AppConfig.load()

        # 2. Initialize logging
        logger = setup_logger("RopeJump", config.logging.log_dir if config.logging.enabled else None)
        logger.info("Application starting")

        # 3. Setup GPU
        setup_gpu()
        logger.info("GPU configuration completed")

        # 4. Initialize model
        predictor = VideoPredictor(str(config.model.model_path))
        logger.info("Model loading completed")

        # 5. Start GUI
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
