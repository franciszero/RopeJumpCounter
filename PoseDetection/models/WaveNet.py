# PoseDetection/models/WaveNet.py
import tensorflow as tf
from tensorflow.keras import layers, models
from PoseDetection.models.BaseModel import TrainMyModel


class WaveNetModel(TrainMyModel):
    """Dilated causal convolutions à la WaveNet."""
    def __init__(self, name="wavenet"):
        super().__init__(name)
        self._init_model()

    def residual_block(self, x, dilation_rate, filters):
        tanh_out = layers.Conv1D(filters, 3, padding='causal',
                                 dilation_rate=dilation_rate,
                                 activation='tanh')(x)
        sigm_out = layers.Conv1D(filters, 3, padding='causal',
                                 dilation_rate=dilation_rate,
                                 activation='sigmoid')(x)
        gated = layers.Multiply()([tanh_out, sigm_out])
        skip = layers.Conv1D(filters, 1, padding='same')(gated)
        res = layers.Add()([skip, x])
        return res, skip

    def _build(self):
        inputs = layers.Input(shape=self.X_train.shape[1:])      # (T, D)

        x = layers.Conv1D(64, 1, padding='same')(inputs)
        skips = []
        for d in [1, 2, 4, 8, 16, 32]:
            x, s = self.residual_block(x, d, 64)
            skips.append(s)

        x = layers.Add()(skips)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
            **self.compile_kwargs
        )
        return model