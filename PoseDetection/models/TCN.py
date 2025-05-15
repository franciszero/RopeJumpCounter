from PoseDetection.models.BaseModel import TrainMyModel
import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="PoseDetection")
class TCNBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(
            filters,
            kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu'
        )
        self.norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv(inputs)
        return self.norm(x)


class TCNModel(TrainMyModel):
    def __init__(self, name="tcn"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        input_shape = self.X_train.shape[1:]
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for dilation in [1, 2, 4, 8]:
            x = TCNBlock(filters=64, kernel_size=3, dilation_rate=dilation)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),  # ROC‑AUC
                tf.keras.metrics.AUC(curve='PR', name='pr_auc'),  # PR‑AUC
            ],
            **self.compile_kwargs
        )
        return model
