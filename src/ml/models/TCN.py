from PoseDetection.models.BaseModel import TrainMyModel
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PoseDetection.models.ModelParams.TCNBlock import TCNBlock


class TCNModel(TrainMyModel):
    def __init__(self, name="tcn"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        input_shape = self.X_train.shape[1:]
        inputs = Input(shape=input_shape)
        x = inputs
        for dilation in [1, 2, 4, 8]:
            x = TCNBlock(filters=64, kernel_size=3, dilation_rate=dilation)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.lr_schedule),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                AUC(name='auc'),  # ROC‑AUC
                AUC(curve='PR', name='pr_auc'),  # PR‑AUC
            ],
            **self.compile_kwargs
        )
        return model
