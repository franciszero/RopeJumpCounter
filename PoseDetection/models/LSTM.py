from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from PoseDetection.models.BaseModel import TrainMyModel
import tensorflow as tf


class LSTMModel(TrainMyModel):
    def __init__(self, name="lstm"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        model = Sequential()
        model.add(tf.keras.Input(shape=(self.window_size, self.X_train.shape[2])))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def get_callbacks(self):
        return [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{self.dest_root}/best_lstm.keras",
                monitor="val_accuracy", save_best_only=True, verbose=1
            )
        ]
