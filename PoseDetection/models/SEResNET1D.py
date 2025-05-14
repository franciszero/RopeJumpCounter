# PoseDetection/models/SEResNET1D.py
import tensorflow as tf
from tensorflow.keras import layers, models
from PoseDetection.models.BaseModel import TrainMyModel


class SEResNET1DModel(TrainMyModel):
    """ResNet-style blocks + SE attention."""
    def __init__(self, name="seresnet1d"):
        super().__init__(name)
        self._init_model()

    def se_block(self, inputs, r=8):
        ch = inputs.shape[-1]
        s = layers.GlobalAveragePooling1D()(inputs)
        s = layers.Dense(ch // r, activation='relu')(s)
        s = layers.Dense(ch, activation='sigmoid')(s)
        s = layers.Reshape((1, ch))(s)
        return layers.Multiply()([inputs, s])

    def residual_block(self, x, filters, stride=1):
        shortcut = x
        x = layers.Conv1D(filters, 3, stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv1D(filters, 3, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(x)

        x = self.se_block(x)

        if shortcut.shape[-1] != filters or stride != 1:
            shortcut = layers.Conv1D(filters, 1, stride, padding='same', use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def _build(self):
        inputs = layers.Input(shape=self.X_train.shape[1:])
        x = self.residual_block(inputs, 64, 1)
        x = self.residual_block(x, 128, 2)
        x = self.residual_block(x, 128, 1)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
            **self.compile_kwargs
        )
        return model