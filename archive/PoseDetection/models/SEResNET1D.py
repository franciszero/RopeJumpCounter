# PoseDetection/models/SEResNET1D.py
from PoseDetection.models.BaseModel import TrainMyModel
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class SEResNET1DModel(TrainMyModel):
    """ResNet-style blocks + SE attention."""

    def __init__(self, name="seresnet1d"):
        super().__init__(name)
        self._init_model()

    def se_block(self, inputs, r=8):
        ch = inputs.shape[-1]
        s = GlobalAveragePooling1D()(inputs)
        s = Dense(ch // r, activation='relu')(s)
        s = Dense(ch, activation='sigmoid')(s)
        s = Reshape((1, ch))(s)
        return Multiply()([inputs, s])

    def residual_block(self, x, filters, stride=1):
        shortcut = x
        x = Conv1D(filters, 3, stride, padding='same', use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-3)(x)
        x = Activation('relu')(x)

        x = Conv1D(filters, 3, 1, padding='same', use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-3)(x)

        x = self.se_block(x)

        if shortcut.shape[-1] != filters or stride != 1:
            shortcut = Conv1D(filters, 1, stride, padding='same', use_bias=False)(shortcut)
            shortcut = BatchNormalization(momentum=0.9, epsilon=1e-3)(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])
        x = self.residual_block(inputs, 64, 1)
        x = self.residual_block(x, 128, 2)
        x = self.residual_block(x, 128, 1)

        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.4)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.lr_schedule),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                AUC(name='auc'),  # ROC‑AUC
                AUC(curve='PR', name='pr_auc'),  # PR‑AUC
            ],
            **self.compile_kwargs
        )
        return model
