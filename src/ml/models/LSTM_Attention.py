from PoseDetection.models.BaseModel import TrainMyModel
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class LSTMAttentionModel(TrainMyModel):
    def __init__(self, name="lstm_attention"):
        super().__init__(name)
        self._init_model()
        # assert self.window_size >= 12, (f"LSTM_Attention needs window_size>=12, got {self.window_size}")

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])  # (T, D)

        # LSTM encoder
        x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.25))(inputs)  # (T, 128)

        # Scaled dot‑product self‑attention
        context = Attention(use_scale=True)([x, x])  # (T, 128)

        # Aggregate
        x = GlobalAveragePooling1D()(context)

        # Dense head
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.4)(x)
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
