# src/ml/models/ModelParams/ThresholdHolder.py
import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="custom")
class ThresholdHolder(keras.layers.Layer):
    """
    in SavedModel / .keras persist a scalar inthreshold t
    """

    def __init__(self, t=0.5, **kwargs):
        super().__init__(**kwargs)  # ← no longer repeatedly pass trainable
        self.trainable = False  # still remain non-training
        self.t_init = float(t)

    def build(self, input_shape):
        self.t = self.add_weight(
            name="threshold",
            shape=(),
            initializer=tf.constant_initializer(self.t_init),
            trainable=False,
        )

    def call(self, inputs):
        return inputs  # pass through ── only forsave t

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"t": self.t_init})
        return cfg
