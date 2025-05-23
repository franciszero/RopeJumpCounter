# models/EfficientNet1D.py
from PoseDetection.models.BaseModel import TrainMyModel
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


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
    x = SeparableConv1D(
        filters=point_filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=False
    )(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-3)(x)
    x = Activation("swish")(x)
    return x


class EfficientNet1DModel(TrainMyModel):
    """Simplified EfficientNet‑V2 stem adapted to 1‑D sequences."""

    def __init__(self, name="efficientnet1d"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])  # (W, D)
        x = depthwise_conv1d(inputs, 32, 3, 1)
        x = depthwise_conv1d(x, 64, 3, 1)
        x = depthwise_conv1d(x, 128, 3, 2)
        x = depthwise_conv1d(x, 256, 3, 1)

        # --- head ---
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='swish')(x)
        x = Dropout(0.4)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.lr_schedule),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                AUC(name='auc'),
                AUC(curve='PR', name='pr_auc')
            ],
            **self.compile_kwargs  # 透传额外参数
        )
        return model
