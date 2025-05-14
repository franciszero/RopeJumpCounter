# PoseDetection/models/TFTLite.py
import tensorflow as tf
from tensorflow.keras import layers, models
from PoseDetection.models.BaseModel import TrainMyModel


class TFTLiteModel(TrainMyModel):
    """Lightweight Temporal Fusion Transformer."""
    def __init__(self, name="tftlite"):
        super().__init__(name)
        self._init_model()

    def gated_residual(self, x, y):
        gate = layers.Activation('sigmoid')(y)
        return layers.Add()([x * gate, y * (1 - gate)])

    def _build(self):
        d_model = 64
        inputs = layers.Input(shape=self.X_train.shape[1:])        # (T, D)
        x = layers.LayerNormalization()(inputs)
        x = layers.Dense(d_model)(x)

        # GRN + Skip connections
        lstm_out = layers.Bidirectional(
            layers.LSTM(d_model // 2, return_sequences=True, recurrent_dropout=0.25))(x)
        x = self.gated_residual(x, lstm_out)

        # Multi-head attn with gating
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=d_model // 4,
                                         dropout=0.1)(x, x)
        attn = layers.LayerNormalization()(attn)
        x = self.gated_residual(x, attn)

        # Static enrichment (dense)
        x = layers.Dense(d_model, activation='relu')(x)

        # Output head
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
            **self.compile_kwargs
        )
        return model