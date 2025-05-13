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
    """轻量版 InceptionTime‑1D"""
    def __init__(self, name="inception"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        inputs = layers.Input(shape=self.X_train.shape[1:])
        x = inputs
        # 堆叠 3 个 Inception block
        for _ in range(3):
            x = inception_module(x, 32)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs, outputs)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def get_callbacks(self):
        import tensorflow as tf
        return [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=6,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.5,
                                                 patience=3,
                                                 verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{self.dest_root}/best_{self.model_name}.keras",
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            )
        ]