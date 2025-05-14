import tensorflow as tf
from PoseDetection.models.BaseModel import TrainMyModel


class CRNNModel(TrainMyModel):
    def __init__(self, name="crnn"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        inputs = tf.keras.Input(shape=self.X_train.shape[1:])

        # --- Conv block 1 ---
        x = tf.keras.layers.Conv1D(64, 3, padding='same', use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv1D(64, 3, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)

        # --- Conv block 2 ---
        x = tf.keras.layers.Conv1D(128, 3, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv1D(128, 3, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)

        # --- Bidirectional LSTM ---
        lstm = tf.keras.layers.LSTM(64, return_sequences=True, recurrent_dropout=0.25)
        x = tf.keras.layers.Bidirectional(lstm)(x)

        # sequence → vector
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # --- dense head ---
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
            **self.compile_kwargs
        )
        return model

    def _get_callbacks(self):
        return [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{self.dest_root}/best_crnn.keras",
                monitor="val_accuracy", save_best_only=True, verbose=1
            )
        ]
