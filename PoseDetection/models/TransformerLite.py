# models/TransformerLite.py
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from PoseDetection.models.BaseModel import TrainMyModel


class PositionalEncoding(layers.Layer):
    """简化版正余弦位置编码"""

    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        pos = tf.cast(tf.range(seq_len)[:, None], tf.float32)
        i = tf.cast(tf.range(d_model)[None, :], tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return x + pos_encoding[None, :, :]


class TransformerLiteModel(TrainMyModel):
    """Encoder‑only Transformer，用于短时序"""

    def __init__(self, name="transformer"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        d_model = 64
        inputs = layers.Input(shape=self.X_train.shape[1:])  # (T, D)

        # --- Positional Encoding via Conv1D ---
        x = layers.Conv1D(d_model, 1, padding='same')(inputs)

        # --- Transformer Encoder blocks (lite) ---
        for _ in range(2):
            # Multi-head attention
            attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=d_model // 4,
                                                 dropout=0.1)(x, x)
            x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)

            # Feed-forward
            ff = layers.Dense(d_model * 2, activation='relu')(x)
            ff = layers.Dense(d_model)(ff)
            x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

        # --- Classification head ---
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
