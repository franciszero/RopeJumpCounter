# PoseDetection/models/ModelParams/ThresholdHolder.py
import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="custom")
class ThresholdHolder(keras.layers.Layer):
    """在 SavedModel / .keras 里持久化一个标量阈值 t"""

    def __init__(self, t=0.5, **kwargs):
        super().__init__(**kwargs)  # ← 不再重复传 trainable
        self.trainable = False  # 依然保持不可训练
        self.t_init = float(t)

    def build(self, input_shape):
        self.t = self.add_weight(
            name="threshold",
            shape=(),
            initializer=tf.constant_initializer(self.t_init),
            trainable=False,
        )

    def call(self, inputs):
        return inputs  # 透传 ── 只为保存 t

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"t": self.t_init})
        return cfg
