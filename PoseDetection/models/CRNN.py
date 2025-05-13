import tensorflow as tf
from PoseDetection.models.BaseModel import TrainMyModel


class CRNNModel(TrainMyModel):
    def __init__(self, name="crnn"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        input_shape = self.X_train.shape[1:]

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def get_callbacks(self):
        return [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"models/best_crnn.keras",
                monitor="val_accuracy", save_best_only=True, verbose=1
            )
        ]