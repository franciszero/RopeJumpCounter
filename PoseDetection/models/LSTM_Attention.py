from PoseDetection.models.BaseModel import TrainMyModel
import tensorflow as tf
from tensorflow.keras import layers, models


class LSTMAttentionModel(TrainMyModel):
    def __init__(self, name="lstm_attention"):
        super().__init__(name)
        self._init_model()
        assert self.window_size >= 12, (f"LSTM_Attention needs window_size>=12, got {self.window_size}")

    def _build(self):
        inputs = layers.Input(shape=self.X_train.shape[1:])  # (T, D)

        # LSTM encoder
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, recurrent_dropout=0.25))(inputs)  # (T, 128)

        # Scaled dot‑product self‑attention
        context = layers.Attention(use_scale=True)([x, x])  # (T, 128)

        # Aggregate
        x = layers.GlobalAveragePooling1D()(context)

        # Dense head
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
            **self.compile_kwargs
        )
        return model

    def _get_callbacks(self):
        return [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.4, patience=4, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{self.dest_root}/best_lstm_attention.keras",
                monitor="val_accuracy", save_best_only=True, verbose=1
            )
        ]
