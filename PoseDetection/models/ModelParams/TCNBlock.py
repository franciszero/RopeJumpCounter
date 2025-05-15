from tensorflow import keras
import tensorflow as tf


@keras.utils.register_keras_serializable(package="PoseDetection")
class TCNBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.conv = tf.keras.layers.Conv1D(
            filters, kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation="relu")
        self.norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv(inputs)
        return self.norm(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
        })
        return config
