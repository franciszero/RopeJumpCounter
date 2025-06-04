# PoseDetection/models/WaveNet.py
from PoseDetection.models.BaseModel import TrainMyModel
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class WaveNetModel(TrainMyModel):
    """Dilated causal convolutions à la WaveNet."""

    def __init__(self, name="wavenet"):
        super().__init__(name)
        self._init_model()

    def residual_block(self, x, dilation_rate, filters):
        tanh_out = Conv1D(filters, 3, padding='causal',
                                 dilation_rate=dilation_rate,
                                 activation='tanh')(x)
        sigm_out = Conv1D(filters, 3, padding='causal',
                                 dilation_rate=dilation_rate,
                                 activation='sigmoid')(x)
        gated = Multiply()([tanh_out, sigm_out])
        skip = Conv1D(filters, 1, padding='same')(gated)
        res = Add()([skip, x])
        return res, skip

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])  # (T, D)

        x = Conv1D(64, 1, padding='same')(inputs)
        skips = []
        for d in [1, 2, 4, 8, 16, 32]:
            x, s = self.residual_block(x, d, 64)
            skips.append(s)

        x = Add()(skips)
        x = Activation('relu')(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                AUC(name='auc'),  # ROC‑AUC
                AUC(curve='PR', name='pr_auc'),  # PR‑AUC
            ],
            **self.compile_kwargs
        )
        return model
