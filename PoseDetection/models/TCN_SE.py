# PoseDetection/models/TCN.py
import tensorflow as tf
from tensorflow.keras import layers, models

from PoseDetection.models.BaseModel import TrainMyModel


# ----------- 小组件 ----------- #
def squeeze_excite(inputs, reduction=16, name=None):
    """1D SE block."""
    channels = inputs.shape[-1]
    se = layers.GlobalAveragePooling1D(name=f"{name}_gap")(inputs)
    se = layers.Dense(channels // reduction, activation="relu",
                      name=f"{name}_fc1")(se)
    se = layers.Dense(channels, activation="sigmoid",
                      name=f"{name}_fc2")(se)
    se = layers.Reshape((1, channels))(se)
    return layers.Multiply(name=f"{name}_scale")([inputs, se])


def tcn_block(x, filters, dilation_rate, name):
    """Dilated causal conv + BN + GELU + SE + residual."""
    y = layers.Conv1D(
        filters,
        kernel_size=3,
        padding="causal",
        dilation_rate=dilation_rate,
        kernel_initializer="he_normal",
        name=f"{name}_conv")(x)
    y = layers.BatchNormalization(name=f"{name}_bn")(y)
    y = layers.Activation("gelu", name=f"{name}_act")(y)
    y = squeeze_excite(y, name=f"{name}_se")
    # 如果通道数不同，用 1×1 调整 residual
    if x.shape[-1] != filters:
        x = layers.Conv1D(filters, 1, padding="same",
                          name=f"{name}_resize")(x)
    return layers.Add(name=f"{name}_add")([x, y])


# ----------- 模型 ----------- #
class TCNSEModel(TrainMyModel):
    def __init__(self, name="tcn_se"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        # 输入：[B, 24, n_features]
        input_shape = self.X_train.shape[1:]
        inputs = layers.Input(shape=input_shape, name="inputs")
        x = inputs

        # 3 级堆叠，每级 2 个 dilation(1,2)->(4,8)->(16,32)
        filters = [64, 128, 256]
        for i, f in enumerate(filters):
            for j, d in enumerate([1 << (2 * i), 1 << (2 * i + 1)]):
                x = tcn_block(x, f, d, name=f"block{i + 1}_{j + 1}")
            x = layers.Dropout(0.2, name=f"drop{i + 1}")(x)

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
