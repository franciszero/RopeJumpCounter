#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
from tensorflow.keras import mixed_precision

from src.config.settings import AppConfig
from src.core.video_predictor import VideoPredictor
from src.interface.gui import PlayerGUI
from src.utils.logging import setup_logger
from src.core.exceptions import AppError

def setup_gpu():
    """设置GPU加速"""
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus, 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def main():
    try:
        # 1. 加载配置
        config = AppConfig.load()
        
        # 2. 初始化日志
        logger = setup_logger("RopeJump", config.logging.log_dir if config.logging.enabled else None)
        logger.info("应用启动")
        
        # 3. 设置GPU
        setup_gpu()
        logger.info("GPU设置完成")
        
        # 4. 初始化模型
        predictor = VideoPredictor(str(config.model.model_path))
        logger.info("模型加载完成")
        
        # 5. 启动GUI
        gui = PlayerGUI(
            predictor=predictor,
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
            save_path=config.save_video_path
        )
        gui.run()
        
    except AppError as e:
        logger.error(f"应用错误: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"未知错误: {e}")
        sys.exit(1)
    finally:
        logger.info("应用退出")

if __name__ == "__main__":
    main() 