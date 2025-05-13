import tensorflow as tf
from PoseDetection.models.BaseModel import TrainMyModel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class CNNModel(TrainMyModel):
    def __init__(self, name="cnn"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        # print(f"[DEBUG] self.X_train.shape = {self.X_train.shape}")
        # input_shape = self.X_train.shape[1:]

        model = tf.keras.models.Sequential([
            # tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def get_callbacks(self):
        return [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            ModelCheckpoint(
                filepath=f"{self.dest_root}/best_{self.model_name}_w{self.window_size}.keras",
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            )
        ]

class CNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self._build()

    def _build(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

    def call(self, inputs):
        return self.model(inputs)
