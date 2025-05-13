from PoseDetection.models.BaseModel import TrainMyModel
import tensorflow as tf
from tensorflow.keras import layers, models


class LSTMAttentionModel(TrainMyModel):
    def __init__(self, name="lstm_attention"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        input_shape = self.X_train.shape[1:]
        inputs = layers.Input(shape=input_shape)
        x = layers.LSTM(64, return_sequences=True)(inputs)
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(64)(attention)
        attention = layers.Permute([2, 1])(attention)
        attended = layers.multiply([x, attention])
        x = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(attended)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def get_callbacks(self):
        return [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.4, patience=4, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"models/best_lstm_attention.keras",
                monitor="val_accuracy", save_best_only=True, verbose=1
            )
        ]