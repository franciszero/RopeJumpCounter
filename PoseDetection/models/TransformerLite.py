# models/TransformerLite.py
import tensorflow as tf
from tensorflow.keras import layers, models
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
        # Use the original feature dimension as embedding size so that the
        # residual connection shapes match (x + ffn).
        embed_dim = self.X_train.shape[2]  # e.g. 403
        # Choose a num_heads that divides embed_dim; fall back to 1 otherwise.
        num_heads = 4 if embed_dim % 4 == 0 else 1
        key_dim = embed_dim // num_heads
        inputs = layers.Input(shape=self.X_train.shape[1:])  # (W, D)
        x = PositionalEncoding()(inputs)
        # 单层 MHSA + FFN
        x = layers.MultiHeadAttention(num_heads=num_heads,
                                      key_dim=key_dim)(x, x)
        x = layers.LayerNormalization()(x)
        ffn = layers.Dense(embed_dim * 4, activation='relu')(x)
        ffn = layers.Dense(embed_dim)(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization()(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def get_callbacks(self):
        import tensorflow as tf
        return [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                             patience=8,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.5,
                                                 patience=4,
                                                 verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{self.dest_root}/best_{self.model_name}.keras",
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            )
        ]
