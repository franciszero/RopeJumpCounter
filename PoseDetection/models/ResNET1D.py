from PoseDetection.models.BaseModel import TrainMyModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, Dense
import tensorflow as tf


class ResNET1DModel(TrainMyModel):
    def __init__(self, name="resnet1d"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        def residual_block(x, filters, kernel_size, stride):
            shortcut = x
            x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv1D(filters, kernel_size, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            if shortcut.shape[-1] != filters:
                shortcut = Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
                shortcut = BatchNormalization()(shortcut)
            x = Add()([shortcut, x])
            x = ReLU()(x)
            return x

        input_shape = self.X_train.shape[1:]
        inputs = Input(shape=input_shape)
        x = residual_block(inputs, 64, 3, 1)
        x = residual_block(x, 128, 3, 1)
        x = residual_block(x, 128, 3, 1)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def get_callbacks(self):
        return [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=2, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"models/best_resnet1d.keras",
                monitor="val_accuracy", save_best_only=True, verbose=1
            )
        ]