from dataclasses import dataclass
from pathlib import Path
import os
import yaml
from ..core.exceptions import ConfigError
from ..ml.data.builders.feature_mode import mode_to_str, get_feature_mode


@dataclass
class LogConfig:
    enabled: bool = True
    log_dir: Path = Path("logs")
    level: str = "INFO"

@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    device_index: int = 0

@dataclass
class ModelConfig:
    model_name: str = "best_cnn8_ws4_withT.keras"
    threshold: float = 0.5
    
    @property
    def model_path(self) -> Path:
        from utils.FrameSample import SELECTED_LM
        return Path("model_files") / f"models_{len(SELECTED_LM)}_{mode_to_str(get_feature_mode())}" / self.model_name

@dataclass
class AppConfig:
    camera: CameraConfig = CameraConfig()
    model: ModelConfig = ModelConfig()
    logging: LogConfig = LogConfig()
    save_video_path: str | None = None
    
    @classmethod
    def load(cls) -> 'AppConfig':
        """从配置文件或环境变量加载配置"""
        # 1. 尝试从配置文件加载
        config_path = os.getenv('APP_CONFIG', 'config.yaml')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                return cls(**config_dict)
            except Exception as e:
                raise ConfigError(f"配置文件加载失败: {e}")
        
        # 2. 从环境变量加载
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
            save_video_path=os.getenv('SAVE_VIDEO_PATH', 'data/raw_videos_3')
        ) 