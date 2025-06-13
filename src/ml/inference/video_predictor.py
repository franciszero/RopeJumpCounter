"""
Video prediction module

Encapsulates deep learning model inference with sliding window logic for
temporal sequence prediction in jump rope counting applications.
"""

import numpy as np
from collections import deque
from tensorflow.keras import models
import logging
from pathlib import Path
from ...core.exceptions import ModelError
from ..models.ModelParams.ThresholdHolder import ThresholdHolder


logger = logging.getLogger(__name__)


class VideoPredictor:
    """Video-based jump prediction using sliding window inference

    This class wraps a trained deep learning model and implements sliding window
    inference for temporal sequence prediction. It maintains a buffer of recent
    feature vectors and performs inference when the window is full.

    The predictor handles model loading, input validation, and provides
    robust error handling for production use.
    """

    def __init__(self, model_path: str, threshold: float = 0.5):
        """Initialize the video predictor

        Loads the trained model and sets up the sliding window buffer.
        The model is expected to have input shape (batch, window_size, feature_dim).

        Args:
            model_path: Path to the trained Keras model file
            threshold: Decision threshold for binary classification (default: 0.5)

        Raises:
            ModelError: If model file doesn't exist or loading fails
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ModelError(f"Model file not found: {model_path}")

        try:
            logger.info(f"Loading model: {model_path}")
            self.model = models.load_model(model_path, compile=False)
        except Exception as e:
            raise ModelError(f"Model loading failed: {e}")

        # Extract model parameters from input shape (batch, timestamps, feature_dim)
        _, self.window_size, feat_dim = self.model.input_shape
        logger.debug(f"Model parameters: window_size={self.window_size}, feature_dim={feat_dim}")
        self.threshold = threshold

        # Sliding window buffer using deque for efficient operations
        self.buffer = deque(maxlen=self.window_size)

        # Track warmup period before window is full
        self._warmup_remaining = self.window_size

    def predict(self, feature_vector: np.ndarray) -> float:
        """Predict jump probability from feature vector

        Adds the feature vector to the sliding window and performs inference
        if the window is full. During warmup period, returns 0.0.

        Args:
            feature_vector: Feature vector extracted from current frame

        Returns:
            float: Jump probability (0.0-1.0), or 0.0 during warmup

        Raises:
            ModelError: If model inference fails
        """
        try:
            # Add new feature vector to sliding window
            self.buffer.append(feature_vector)

            # Check if window is full (warmup complete)
            if len(self.buffer) < self.window_size:
                return 0.0  # Still in warmup period

            # Prepare input tensor: (window_size, feature_dim) -> (1, window_size, feature_dim)
            window = np.stack(self.buffer, axis=0)
            input_tensor = np.expand_dims(window, axis=0)

            # Perform inference
            self.model.run_eagerly = True
            prediction = self.model(input_tensor, training=False)
            prob = float(prediction[0])

            logger.debug(f"Prediction probability: {prob:.3f}")
            return prob

        except Exception as e:
            raise ModelError(f"Model inference failed: {e}")

    def is_ready(self) -> bool:
        """Check if predictor is ready for inference

        Returns:
            bool: True if sliding window is full and ready for prediction
        """
        return len(self.buffer) >= self.window_size

    def reset(self):
        """Reset the sliding window buffer

        Clears the buffer and resets to warmup state. Useful when
        starting a new video or recovering from errors.
        """
        self.buffer.clear()
        self._warmup_remaining = self.window_size
        logger.debug("Predictor buffer reset")