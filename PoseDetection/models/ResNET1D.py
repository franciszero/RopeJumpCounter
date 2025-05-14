from PoseDetection.models.BaseModel import TrainMyModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, Dense
import tensorflow as tf


class ResNET1DModel(TrainMyModel):
    def __init__(self, name="resnet1d"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        def residual_block(x, filters, stride=1):
            shortcut = x
            # --- Conv-BN-ReLU ---
            x = tf.keras.layers.Conv1D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv1D(filters, 3, strides=1, padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(x)

            # --- shortcut ---
            if shortcut.shape[-1] != filters or stride != 1:
                shortcut = tf.keras.layers.Conv1D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
                shortcut = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(shortcut)

            x = tf.keras.layers.Add()([shortcut, x])
            x = tf.keras.layers.Activation('relu')(x)
            return x

        inputs = tf.keras.layers.Input(shape=self.X_train.shape[1:])
        x = residual_block(inputs, 64, stride=1)
        x = residual_block(x, 128, stride=2)
        x = residual_block(x, 128, stride=1)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
            **self.compile_kwargs
        )
        return model
