import numpy as np
from collections import deque
from tensorflow.keras import models
import logging
from pathlib import Path
from ..core.exceptions import ModelError
from ..ml.models.ModelParams.ThresholdHolder import ThresholdHolder


logger = logging.getLogger(__name__)

class VideoPredictor:
    """封装模型 + 滑动窗口推理逻辑"""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ModelError(f"模型文件不存在: {model_path}")
            
        try:
            logger.info(f"加载模型: {model_path}")
            self.model = models.load_model(model_path, compile=False)
        except Exception as e:
            raise ModelError(f"模型加载失败: {e}")

        # (batch, timestamps, feature_dim)
        _, self.window_size, feat_dim = self.model.input_shape
        logger.debug(f"模型参数: window_size={self.window_size}, feature_dim={feat_dim}")
        self.threshold = 0.5

        # 用 deque 维护最近 window_size 帧特征
        self.buffer = deque(maxlen=self.window_size)
        # 在首次喂满窗口前，无推理结果
        self._warmup = self.window_size

    def predict(self, feature_dim: np.ndarray) -> float:
        """
        传入 BGR frame → 更新窗口 → 若已满返回正例概率，否则 None
        """
        try:
            self.buffer.append(feature_dim)

            if len(self.buffer) < self.window_size:
                return 0.0  # still warming‑up

            window = np.stack(self.buffer, axis=0)  # (win, feat_dim)
            self.model.run_eagerly = True
            prob = float(self.model(np.expand_dims(window, axis=0), training=False)[0])
            logger.debug(f"预测概率: {prob:.3f}")
            return prob
        except Exception as e:
            raise ModelError(f"模型推理失败: {e}") 