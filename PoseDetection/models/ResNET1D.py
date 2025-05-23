from PoseDetection.models.BaseModel import TrainMyModel
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class ResNET1DModel(TrainMyModel):
    def __init__(self, name="resnet1d"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        def residual_block(x, filters, stride=1):
            shortcut = x
            # --- Conv-BN-ReLU ---
            x = Conv1D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
            x = BatchNormalization(momentum=0.9, epsilon=1e-3)(x)
            x = Activation('relu')(x)

            x = Conv1D(filters, 3, strides=1, padding='same', use_bias=False)(x)
            x = BatchNormalization(momentum=0.9, epsilon=1e-3)(x)

            # --- shortcut ---
            if shortcut.shape[-1] != filters or stride != 1:
                shortcut = Conv1D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
                shortcut = BatchNormalization(momentum=0.9, epsilon=1e-3)(shortcut)

            x = Add()([shortcut, x])
            x = Activation('relu')(x)
            return x

        inputs = Input(shape=self.X_train.shape[1:])
        x = residual_block(inputs, 64, stride=1)
        x = residual_block(x, 128, stride=2)
        x = residual_block(x, 128, stride=1)

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
