# PoseDetection/models/ResNET1D.py
from PoseDetection.models.BaseModel import TrainMyModel
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def res_block(x, filters, name):
    y = Conv1D(filters, 3, padding="same", kernel_initializer="he_normal", name=f"{name}_conv1")(x)
    y = BatchNormalization(name=f"{name}_bn1")(y)
    y = Activation("relu", name=f"{name}_act1")(y)

    y = Conv1D(filters, 3, padding="same", kernel_initializer="he_normal", name=f"{name}_conv2")(y)
    y = BatchNormalization(name=f"{name}_bn2")(y)

    if x.shape[-1] != filters: x = Conv1D(filters, 1, padding="same", name=f"{name}_resize")(x)
    out = Add(name=f"{name}_add")([x, y])
    return Activation("relu", name=f"{name}_act2")(out)


def tcn_stack(x, filters, name):
    for i, d in enumerate([1, 2, 4, 8]):
        y = Conv1D(filters, 3, padding="causal", dilation_rate=d, kernel_initializer="he_normal",
                   name=f"{name}_conv{d}")(x)
        y = BatchNormalization(name=f"{name}_bn{d}")(y)
        y = Activation("gelu", name=f"{name}_act{d}")(y)
        x = Add(name=f"{name}_add{d}")([x, y])
    return x


class ResNET1DTcnHybridModel(TrainMyModel):
    def __init__(self, name="resnet1d_tcn"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        input_shape = self.X_train.shape[1:]
        inputs = Input(shape=input_shape, name="inputs")
        x = inputs

        # ResNet stage 1 & 2
        for i, f in enumerate([64, 128], start=1):
            for j in range(2):
                x = res_block(x, f, name=f"res{i}_{j + 1}")
            x = MaxPooling1D(2, name=f"pool{i}")(x)

        # TCN stack – keep same channel width (128) to allow residual add
        x = tcn_stack(x, 128, name="tcn")

        x = GlobalAveragePooling1D(name="gap")(x)
        x = Dense(128, activation="relu", name="fc")(x)
        outputs = Dense(1, activation="sigmoid", name="output")(x)

        model = Model(inputs, outputs, name=self.model_name)

        model.compile(
            optimizer=Adam(learning_rate=self.lr_schedule),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                AUC(name='auc'),  # ROC‑AUC
                AUC(curve='PR', name='pr_auc'),  # PR‑AUC
            ],
            **self.compile_kwargs  # 透传可能的参数
        )
        return model
