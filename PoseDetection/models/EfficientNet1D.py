# models/EfficientNet1D.py
import tensorflow as tf
from tensorflow.keras import layers, models
from PoseDetection.models.BaseModel import TrainMyModel


def depthwise_conv1d(x, point_filters: int, kernel_size: int, stride: int):
    """
    Depth‑wise + point‑wise separable 1‑D convolution block.

    Uses Keras `SeparableConv1D`, which internally performs a depth‑wise
    convolution along the temporal dimension followed by a point‑wise
    projection, and supports arbitrary strides without the row/column
    constraint of `DepthwiseConv2D`.

    Args:
        x            : input tensor of shape (B, W, C)
        point_filters: number of output channels after point‑wise projection
        kernel_size  : temporal kernel size
        stride       : temporal stride
    Returns:
        Tensor of shape (B, new_W, point_filters)
    """
    x = layers.SeparableConv1D(
        filters=point_filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=False
    )(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(x)
    x = layers.Activation("swish")(x)
    return x


class EfficientNet1DModel(TrainMyModel):
    """Simplified EfficientNet‑V2 stem adapted to 1‑D sequences."""

    def __init__(self, name="efficientnet1d"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        inputs = layers.Input(shape=self.X_train.shape[1:])  # (W, D)
        x = depthwise_conv1d(inputs, 32, 3, 1)
        x = depthwise_conv1d(x, 64, 3, 1)
        x = depthwise_conv1d(x, 128, 3, 2)
        x = depthwise_conv1d(x, 256, 3, 1)

        # --- head ---
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='swish')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=['accuracy',tf.keras.metrics.AUC(name='auc')],
            **self.compile_kwargs  # 透传额外参数
        )
        return model

    def get_callbacks(self):
        import tensorflow as tf
        return [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                             patience=6,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.5,
                                                 patience=3,
                                                 verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{self.dest_root}/best_{self.model_name}.keras",
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            )
        ]
