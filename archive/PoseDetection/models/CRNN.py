from PoseDetection.models.BaseModel import TrainMyModel
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class CRNNModel(TrainMyModel):
    def __init__(self, name="crnn"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])

        # --- Conv block 1 ---
        x = Conv1D(64, 3, padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(64, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)

        # --- Conv block 2 ---
        x = Conv1D(128, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(128, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)

        # --- Bidirectional LSTM ---
        lstm = LSTM(64, return_sequences=True, recurrent_dropout=0.25)
        x = Bidirectional(lstm)(x)

        # sequence → vector
        x = GlobalAveragePooling1D()(x)

        # --- dense head ---
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.4)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)

        model.compile(
            optimizer=Adam(learning_rate=self.lr_schedule),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                AUC(name='auc'),
                AUC(curve='PR', name='pr_auc')
            ],
            **self.compile_kwargs
        )
        return model
