import tensorflow as tf

import tensorflow as tf


class ThresholdHolder(tf.keras.layers.Layer):
    """
    把最佳阈值 τ 保存进 SavedModel。
    纯 passthrough，不修改张量，只负责在权重里存一标量。
    """

    def __init__(self, t: float, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self._t_init = float(t)

    def build(self, input_shape):
        # 把 t 作为不可训练权重保存
        self.t = self.add_weight(
            name="threshold",
            shape=[],
            dtype=tf.float32,
            initializer=tf.constant_initializer(self._t_init),
            trainable=False,
        )
        super().build(input_shape)  # 标记已 build

    def call(self, inputs, **kwargs):
        return inputs  # 原样透传

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"t": float(self.t.numpy())})
        return cfg
