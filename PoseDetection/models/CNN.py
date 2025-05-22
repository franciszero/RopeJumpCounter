import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, GlobalAveragePooling1D, Dense, Activation, \
    Dropout, SeparableConv1D, SpatialDropout1D, Reshape, Concatenate, Add, Multiply
from tensorflow.keras.metrics import AUC
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from PoseDetection.models.BaseModel import TrainMyModel


class CNNModel(TrainMyModel):
    def __init__(self, name="cnn"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])
        x = Conv1D(64, 3, padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(64, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)

        x = Conv1D(128, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(128, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)

        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                AUC(name='auc'),  # ROC‑AUC
                AUC(curve='PR', name='pr_auc'),  # PR‑AUC
            ],
            **self.compile_kwargs  # 透传可能的参数
        )

        return model


# Variant 1: Depthwise-separable convolutions (like MobileNet)
class CNN1(TrainMyModel):
    def __init__(self, name="cnn1"): super().__init__(name); self._init_model()

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])
        x = SeparableConv1D(64, 3, padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv1D(64, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)

        x = SeparableConv1D(128, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv1D(128, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                AUC(name='auc'),
                AUC(curve='PR', name='pr_auc')
            ],
            **self.compile_kwargs
        )
        return model


# Variant 2: Add Squeeze-and-Excitation block after each block
class CNN2(TrainMyModel):
    def __init__(self, name="cnn2"): super().__init__(name); self._init_model()

    def se_block(self, inputs, ratio=16):
        filters = inputs.shape[-1]
        se = GlobalAveragePooling1D()(inputs)
        se = Dense(filters // ratio, activation='relu')(se)
        se = Dense(filters, activation='sigmoid')(se)
        se = Reshape([1, filters])(se)
        return Multiply([inputs, se])

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])
        # Block 1
        x = Conv1D(64, 3, padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(64, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = self.se_block(x)
        x = MaxPooling1D(2)(x)
        # Block 2
        x = Conv1D(128, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(128, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = self.se_block(x)
        x = MaxPooling1D(2)(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=[
                'accuracy', AUC(name='auc'),
                AUC(curve='PR', name='pr_auc')
            ],
            **self.compile_kwargs
        )
        return model


# Variant 3: Dilated convolutions to expand receptive field
class CNN3(TrainMyModel):
    def __init__(self, name="cnn3"):
        super().__init__(name)
        self._init_model()

    def conv_block(self, x, filters, dilation_rate):
        x = Conv1D(filters, 3, padding='same', dilation_rate=dilation_rate, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])
        # Stack with increasing dilation
        x = inputs
        for d in [1, 2, 4, 8]:
            x = self.conv_block(x, 64, d)
        x = MaxPooling1D(2)(x)
        for d in [1, 2, 4, 8]:
            x = self.conv_block(x, 128, d)
        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=[
                'accuracy', AUC(name='auc'),
                AUC(curve='PR', name='pr_auc')
            ],
            **self.compile_kwargs
        )
        return model


class CNN4(TrainMyModel):
    """
    Residual CNN: 添加残差连接的卷积网络。
    """

    def __init__(self, name="cnn4"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])
        # Initial conv
        x = Conv1D(64, 3, padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Residual block
        def res_block(x, filters):
            shortcut = x
            x = Conv1D(filters, 3, padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv1D(filters, 3, padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Add([shortcut, x])
            x = Activation('relu')(x)
            return x

        x = res_block(x, 64)
        x = MaxPooling1D(2)(x)

        # Second stage
        x = Conv1D(128, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = res_block(x, 128)
        x = MaxPooling1D(2)(x)

        # Head
        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid', dtype='float32')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                AUC(name='auc'),
                AUC(curve='PR', name='pr_auc')
            ],
            **self.compile_kwargs
        )
        return model


class CNN5(TrainMyModel):
    """
    Depthwise-Separable CNN: 使用可分离卷积减少参数量。
    """

    def __init__(self, name="cnn5"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])
        # Separable conv blocks
        x = SeparableConv1D(64, 3, padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv1D(64, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)

        x = SeparableConv1D(128, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv1D(128, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid', dtype='float32')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                AUC(name='auc'),
                AUC(curve='PR', name='pr_auc')
            ],
            **self.compile_kwargs
        )
        return model


class CNN6(TrainMyModel):
    """
    Multi-Scale CNN: 并行多尺度卷积后拼接特征。
    """

    def __init__(self, name="cnn6"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])

        # Multi-scale conv block
        def ms_block(x, filters):
            conv3 = Conv1D(filters, 3, padding='same', use_bias=False)(x)
            conv5 = Conv1D(filters, 5, padding='same', use_bias=False)(x)
            conv7 = Conv1D(filters, 7, padding='same', use_bias=False)(x)
            x = Concatenate([conv3, conv5, conv7], axis=-1)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x

        x = ms_block(inputs, 64)
        x = MaxPooling1D(2)(x)
        x = ms_block(x, 128)
        x = MaxPooling1D(2)(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid', dtype='float32')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                AUC(name='auc'),
                AUC(curve='PR', name='pr_auc')
            ],
            **self.compile_kwargs
        )
        return model


# Variant 1: Wider receptive field with larger kernels and an extra conv block
class CNN7(TrainMyModel):
    def __init__(self, name="cnn7"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])
        # First block: larger kernel
        x = Conv1D(64, 5, padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(64, 5, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)

        # Extra conv block
        x = Conv1D(128, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', AUC(name='auc'), AUC(curve='PR', name='pr_auc')]
        )
        return model


class CNN8(TrainMyModel):
    def __init__(self, name="cnn8"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])
        x = SeparableConv1D(64, 3, padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(0.2)(x)
        x = SeparableConv1D(64, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)

        x = SeparableConv1D(128, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(0.2)(x)
        x = MaxPooling1D(2)(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', AUC(name='auc'), AUC(curve='PR', name='pr_auc')]
        )
        return model


# Variant 2: Add spatial dropout and transition to depthwise separable convs
class CNN8_1(TrainMyModel):
    def __init__(self, name="cnn8_1"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])
        x = SeparableConv1D(64, 3, padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(0.2)(x)
        x = SeparableConv1D(64, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)

        x = SeparableConv1D(128, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(0.2)(x)
        x = MaxPooling1D(2)(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        # --- Warm‑up + cosine‑restart learning‑rate schedule ---
        steps_per_epoch = 1000  # fallback; you can overwrite this attribute later
        lr_schedule = CosineDecayRestarts(
            initial_learning_rate=3e-4,
            first_decay_steps=3 * steps_per_epoch,
            t_mul=2.0,
            m_mul=0.8
        )

        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                AUC(name='auc'),
                AUC(curve='PR', name='pr_auc')
            ]
        )
        return model


# Variant 3: Residual connections between conv blocks
class CNN9(TrainMyModel):
    def __init__(self, name="cnn9"):
        super().__init__(name)
        self._init_model()

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])
        # Residual block 1
        x = Conv1D(64, 3, padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(64, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        skip = Conv1D(64, 1, padding='same')(inputs)
        x = Add()([x, skip])
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)

        # Residual block 2
        y = Conv1D(128, 3, padding='same', use_bias=False)(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv1D(128, 3, padding='same', use_bias=False)(y)
        y = BatchNormalization()(y)
        skip2 = Conv1D(128, 1, padding='same')(x)
        y = Add()([y, skip2])
        y = Activation('relu')(y)
        y = MaxPooling1D(2)(y)

        x = GlobalAveragePooling1D()(y)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', AUC(name='auc'), AUC(curve='PR', name='pr_auc')]
        )
        return model


class CNNHybridModel(TrainMyModel):
    """
    A 1D CNN that in each block:
      – projects down to a small ‘bottleneck’ channel
      – runs parallel convs (kernels 3,5,7 + one dilated conv)
      – concatenates, applies SE, projects back up
      – adds a residual skip + dropout
    """

    def __init__(self,
                 name="cnn_hybrid",
                 num_blocks=4,
                 filters=64,
                 bottleneck_ratio=4,
                 kernel_sizes=(3, 5, 7),
                 dilation_rate=4,
                 dropout_rate=0.3):
        super().__init__(name)
        self.num_blocks = num_blocks
        self.filters = filters
        self.bottleneck_filters = max(1, filters // bottleneck_ratio)
        self.kernel_sizes = kernel_sizes
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self._init_model()

    def _build(self):
        inputs = Input(shape=self.X_train.shape[1:])
        x = inputs

        for _ in range(self.num_blocks):
            shortcut = x
            # 1×1 bottleneck
            x = Conv1D(self.bottleneck_filters, 1, padding="same", use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            # parallel convs: different kernel sizes
            branches = []
            for k in self.kernel_sizes:
                b = Conv1D(self.bottleneck_filters, k, padding="same", use_bias=False)(x)
                b = BatchNormalization()(b)
                b = Activation("relu")(b)
                branches.append(b)
            # plus one dilated conv branch
            d = Conv1D(self.bottleneck_filters, 3,
                       padding="same",
                       dilation_rate=self.dilation_rate,
                       use_bias=False)(x)
            d = BatchNormalization()(d)
            d = Activation("relu")(d)
            branches.append(d)

            # merge
            x = Concatenate()(branches)

            # squeeze-and-excitation gate
            se = GlobalAveragePooling1D()(x)
            se = Dense(self.filters // 8, activation="relu")(se)
            se = Dense(self.filters, activation="sigmoid")(se)
            se = Reshape((1, self.filters))(se)
            x = Multiply()([x, se])

            # project back to full filter count
            x = Conv1D(self.filters, 1, padding="same", use_bias=False)(x)
            x = BatchNormalization()(x)

            # project shortcut to match channels
            proj = Conv1D(self.filters, 1, padding="same", use_bias=False)(shortcut)
            proj = BatchNormalization()(proj)

            # residual + activation + dropout
            x = Add()([proj, x])
            x = Activation("relu")(x)
            x = Dropout(self.dropout_rate)(x)

        # final classifier head
        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1, activation="sigmoid")(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(3e-4),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                AUC(name="auc"),
                AUC(curve="PR", name="pr_auc"),
            ],
            **self.compile_kwargs
        )
        return model
