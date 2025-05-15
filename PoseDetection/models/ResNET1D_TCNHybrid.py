# PoseDetection/models/ResNET1D.py
import tensorflow as tf
from tensorflow.keras import layers, models

from PoseDetection.models.BaseModel import TrainMyModel


def res_block(x, filters, name):
    y = layers.Conv1D(filters, 3, padding="same",
                      kernel_initializer="he_normal",
                      name=f"{name}_conv1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.Activation("relu", name=f"{name}_act1")(y)

    y = layers.Conv1D(filters, 3, padding="same",
                      kernel_initializer="he_normal",
                      name=f"{name}_conv2")(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)

    if x.shape[-1] != filters:
        x = layers.Conv1D(filters, 1, padding="same",
                          name=f"{name}_resize")(x)
    out = layers.Add(name=f"{name}_add")([x, y])
    return layers.Activation("relu", name=f"{name}_act2")(out)


def tcn_stack(x, filters, name):
    for i, d in enumerate([1, 2, 4, 8]):
        y = layers.Conv1D(filters, 3, padding="causal",
                          dilation_rate=d,
                          kernel_initializer="he_normal",
                          name=f"{name}_conv{d}")(x)
        y = layers.BatchNormalization(name=f"{name}_bn{d}")(y)
        y = layers.Activation("gelu", name=f"{name}_act{d}")(y)
        x = layers.Add(name=f"{name}_add{d}")([x, y])
    return x


class ResNET1DTcnHybridModel(TrainMyModel):
    def __init__(self, name="resnet1d_tcn"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        input_shape = self.X_train.shape[1:]
        inputs = layers.Input(shape=input_shape, name="inputs")
        x = inputs

        # ResNet stage 1 & 2
        for i, f in enumerate([64, 128], start=1):
            for j in range(2):
                x = res_block(x, f, name=f"res{i}_{j + 1}")
            x = layers.MaxPooling1D(2, name=f"pool{i}")(x)

        # Hybrid stage: TCN stack
        x = tcn_stack(x, 256, name="tcn")

        x = layers.GlobalAveragePooling1D(name="gap")(x)
        x = layers.Dense(128, activation="relu", name="fc")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

        model = tf.keras.Model(inputs, outputs, name=self.model_name)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),          # ROC‑AUC
                tf.keras.metrics.AUC(curve='PR', name='pr_auc'),  # PR‑AUC
            ],
            **self.compile_kwargs  # 透传可能的参数
        )
        return model
