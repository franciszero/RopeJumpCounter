# models/InceptionTime.py
from PoseDetection.models.BaseModel import TrainMyModel
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class InceptionTimeModel(TrainMyModel):
    """Inception-Time block × 3  (simplified, 1-D)."""

    def __init__(self, name="inception"):
        super().__init__(name)
        self._init_model()

    @staticmethod
    def inception_module(x, filters):
        branch1 = Conv1D(filters, 1, padding='same', activation='relu')(x)

        branch3 = Conv1D(filters, 1, padding='same', activation='relu')(x)
        branch3 = Conv1D(filters, 3, padding='same', activation='relu')(branch3)

        branch5 = Conv1D(filters, 1, padding='same', activation='relu')(x)
        branch5 = Conv1D(filters, 5, padding='same', activation='relu')(branch5)

        pool = MaxPooling1D(3, strides=1, padding='same')(x)
        pool = Conv1D(filters, 1, padding='same', activation='relu')(pool)

        x = Concatenate()([branch1, branch3, branch5, pool])
        x = BatchNormalization(momentum=0.9, epsilon=1e-3)(x)
        x = Activation('relu')(x)
        return x

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])  # (T, D)
        x = inputs
        for f in [16, 32, 64]:
            x = self.inception_module(x, f)
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
