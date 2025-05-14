# models/InceptionTime.py
import tensorflow as tf
from tensorflow.keras import layers, models
from PoseDetection.models.BaseModel import TrainMyModel


def inception_module(x, filters=32):
    # 四条并行分支：1x1、3x3、5x5、3x3‑pool‑1x1
    b1 = layers.Conv1D(filters, 1, padding='same', activation='relu')(x)

    b2 = layers.Conv1D(filters, 1, padding='same', activation='relu')(x)
    b2 = layers.Conv1D(filters, 3, padding='same', activation='relu')(b2)

    b3 = layers.Conv1D(filters, 1, padding='same', activation='relu')(x)
    b3 = layers.Conv1D(filters, 5, padding='same', activation='relu')(b3)

    b4 = layers.MaxPooling1D(3, strides=1, padding='same')(x)
    b4 = layers.Conv1D(filters, 1, padding='same', activation='relu')(b4)

    return layers.concatenate([b1, b2, b3, b4], axis=-1)


class InceptionTimeModel(TrainMyModel):
    """Inception-Time block × 3  (simplified, 1-D)."""

    def __init__(self, name="inception"):
        super().__init__(name)
        self._init_model()

    @staticmethod
    def inception_module(x, filters):
        branch1 = layers.Conv1D(filters, 1, padding='same', activation='relu')(x)

        branch3 = layers.Conv1D(filters, 1, padding='same', activation='relu')(x)
        branch3 = layers.Conv1D(filters, 3, padding='same', activation='relu')(branch3)

        branch5 = layers.Conv1D(filters, 1, padding='same', activation='relu')(x)
        branch5 = layers.Conv1D(filters, 5, padding='same', activation='relu')(branch5)

        pool = layers.MaxPooling1D(3, strides=1, padding='same')(x)
        pool = layers.Conv1D(filters, 1, padding='same', activation='relu')(pool)

        x = layers.Concatenate()([branch1, branch3, branch5, pool])
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(x)
        x = layers.Activation('relu')(x)
        return x

    def _build(self):
        inputs = layers.Input(shape=self.X_train.shape[1:])  # (T, D)
        x = inputs
        for f in [16, 32, 64]:
            x = self.inception_module(x, f)
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

    def _get_callbacks(self):
        import tensorflow as tf
        return [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{self.dest_root}/best_{self.model_name}.keras",
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            )
        ]
