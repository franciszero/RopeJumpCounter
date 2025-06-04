# PoseDetection/models/TFTLite.py
from PoseDetection.models.BaseModel import TrainMyModel
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class TFTLiteModel(TrainMyModel):
    """Lightweight Temporal Fusion Transformer."""
    def __init__(self, name="tftlite"):
        super().__init__(name)
        self._init_model()

    def gated_residual(self, x, y):
        gate = Activation('sigmoid')(y)
        return Add()([x * gate, y * (1 - gate)])

    def _build(self):
        d_model = 64
        inputs = Input(shape=self.X_train.shape[1:])        # (T, D)
        x = LayerNormalization()(inputs)
        x = Dense(d_model)(x)

        # GRN + Skip connections
        lstm_out = Bidirectional(
            LSTM(d_model // 2, return_sequences=True, recurrent_dropout=0.25))(x)
        x = self.gated_residual(x, lstm_out)

        # Multi-head attn with gating
        attn = MultiHeadAttention(num_heads=4, key_dim=d_model // 4,
                                         dropout=0.1)(x, x)
        attn = LayerNormalization()(attn)
        x = self.gated_residual(x, attn)

        # Static enrichment (dense)
        x = Dense(d_model, activation='relu')(x)

        # Output head
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.lr_schedule, clipnorm=1.0),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                AUC(name='auc'),  # ROC‑AUC
                AUC(curve='PR', name='pr_auc'),  # PR‑AUC
            ],
            **self.compile_kwargs
        )
        return model