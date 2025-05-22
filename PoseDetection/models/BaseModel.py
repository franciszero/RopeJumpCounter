import tensorflow as tf
from abc import ABC, abstractmethod
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve
import io
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc, precision_recall_curve
import datetime, os

from PoseDetection.data_builder_utils.feature_mode import mode_to_str, get_feature_mode
from PoseDetection.models.ModelParams.ThresholdHolder import ThresholdHolder
from utils.FrameSample import SELECTED_LM


class TrainMyModel(ABC):
    def __init__(self, name, dest_root='../model_files', source_root='../data', *, class_weight_dict=None,
                 **compile_kwargs):
        self.is_training = None
        self.class_weight_dict = class_weight_dict
        self.compile_kwargs = compile_kwargs

        self.model_name = name
        mode = mode_to_str(get_feature_mode())
        self.dest_root = f"{dest_root}/models_{len(SELECTED_LM)}_{mode}"
        self.source_root = f"{source_root}/dataset_{len(SELECTED_LM)}_{mode}"
        self.num_classes = 2
        self.random_state = 42
        self.epochs = 100
        self.batch_size = 2048
        self.TEST_RATIO = 0.15
        self.VAL_RATIO = 0.15

        # 配置: 各模型对应的 window_size
        self.MODEL_WINDOW_SIZES = {
            "xgb": 1,
            "cnn": 4,
            "cnn1": 4,
            "cnn2": 4,
            "cnn3": 4,
            "cnn4": 4,
            "cnn5": 4,
            "cnn6": 4,
            "cnn7": 4,
            "cnn8": 4,
            "cnn8_1": 4,
            "cnn9": 4,
            "cnn_hybrid": 4,
            "crnn": 12,
            "efficientnet1d": 4,
            "inception": 4,
            "lstm_attention": 16,
            "resnet1d": 16,
            'resnet1d_tcn': 16,
            "seresnet1d": 16,
            "tcn": 24,
            "tcn_se": 24,
            "tftlite": 16,
            "transformerlite": 16,
            "wavenet": 8,
        }
        self._need_aug = name in {"cnn", "efficientnet1d"}

        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.y_true \
            = None, None, None, None, None, None, None
        self.history = None
        self.window_size = None
        self.y_prob = None  # self.model.predict(self.X_test).flatten()
        self.y_pred = None  # self.y_pred = (self.y_prob > 0.5).astype(int)
        self.report = None  # self.report = classification_report(self.y_test, self.y_pred, output_dict=True)
        self.model = None

    def _init_model(self):
        self.window_size = self.MODEL_WINDOW_SIZES[self.model_name]
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test \
            = self.__load_window_npz(self.window_size)

        print("X_train NaNs:", np.isnan(self.X_train).sum())
        print("X_val   NaNs:", np.isnan(self.X_val).sum())
        print("X_test  NaNs:", np.isnan(self.X_test).sum())

        self.y_true = self.y_test
        weight = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        self.class_weight_dict = dict(enumerate(weight))

        if self.class_weight_dict is None:
            neg, pos = np.bincount(self.y_train.astype(int))
            self.class_weight_dict = {0: (neg + pos) / (2.0 * neg), 1: (neg + pos) / (2.0 * pos)}

        self.model = self._build()

    @abstractmethod
    def _build(self):
        """子类必须实现：构建模型结构"""
        pass

    # ------- 新增 / 替换 -------
    def _get_callbacks(self):
        """通用回调：EarlyStopping + ReduceLROnPlateau + ModelCheckpoint + PRCurve + TensorBoard"""
        ckpt_path = os.path.join(self.dest_root, f"best_{self.model_name}_ws{self.window_size}.keras")

        # --- 核心回调 ---
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                monitor="val_pr_auc" if "pr_auc" in self.model.metrics_names else "val_accuracy",
                save_best_only=True,
                verbose=1)
        ]

        # --- PR‑AUC 曲线监控 & TensorBoard ---
        log_dir = os.path.join(
            self.dest_root,
            "logs",
            self.model_name,
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        # 自定义 PR 曲线回调
        callbacks.append(
            PRCurveCallback(
                val_data=(self.X_val, self.y_val),
                log_dir=log_dir,
                prefix="val"
            )
        )

        # TensorBoard 可视化
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=False,
                update_freq="epoch"
            )
        )

        return callbacks

    def train(self):
        print("\n======================================================")
        print(f"\n============ Training {self.model_name} ============")
        print("\n======================================================")
        print(f"Class‑weight: {self.class_weight_dict}")
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self._get_callbacks(),
            class_weight=self.class_weight_dict,
            verbose=2
        )
        print(f"Class‑weight: {self.class_weight_dict}")
        pass

    def evaluate(self):
        self.y_prob = self.model.predict(self.X_test).flatten()
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_prob)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        best_t = float(thresholds[np.argmax(f1)])
        print("best F1 threshold =", best_t)
        # 动态加属性
        best_t = float(thresholds[np.argmax(f1)])

        holder = ThresholdHolder(best_t, name="f1_threshold")
        new_output = holder(self.model.output)
        self.model = tf.keras.Model(self.model.input, new_output, name=f"{self.model.name}_with_t")
        # 再附一个 Python 属性，双保险（可选）
        self.model.best_threshold = best_t

        self.y_pred = (self.y_prob > best_t).astype(int)

        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        self.report = {
            'classification': classification_report(self.y_test, self.y_pred, output_dict=True),
            'roc_auc': roc_auc_score(self.y_test, self.y_prob),
            'average_precision': average_precision_score(self.y_test, self.y_prob),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
        }

    def _augment_window(self, window):
        if not (self.is_training and self._need_aug):
            return window
        if self.is_training:  # 只在训练集增强
            if tf.random.uniform([]) < 0.5:  # jitter
                window += tf.random.normal(tf.shape(window), stddev=0.01)
            if tf.random.uniform([]) < 0.5:  # scaling
                scale = tf.random.uniform([], 0.8, 1.2)
                window *= scale
            if tf.random.uniform([]) < 0.3:  # flip
                window = tf.concat([window[..., :1],  # 时间
                                    -window[..., 1:]], axis=-1)
            # time-warp 省略，可用 random time stretch + interpolation
        return window

    def __load_window_npz(self, window_size):
        """
        Load window‑level datasets for a given window size.

        Preferred layout (produced by the updated builder):

            {source_root}/
                size{window_size}/
                    train/*.npz
                    val/*.npz
                    test/*.npz

        Each .npz file is expected to contain
            X : shape (n, W, D)
            y : shape (n,)
        for a single video.

        If the new directory structure is not found, we fall back to the
        legacy flat layout and create the train/val/test splits on‑the‑fly
        (previous behaviour).
        """
        import glob
        import os

        def _load_split(split_dir):
            """Stack all .npz files under `split_dir`. Returns (X, y) or (None, None) if absent."""
            if not os.path.isdir(split_dir):
                return None, None
            Xs, ys = [], []
            for f in glob.glob(os.path.join(split_dir, "*.npz")):
                data = np.load(f)
                n, p = np.bincount(data["y"])
                print(f'{f}: neg={n}, pos={p}, pos_ratio={p / len(data["y"]):.2%}')
                Xs.append(data["X"])
                ys.append(data["y"])
            if not Xs:  # directory exists but empty
                return None, None
            return np.vstack(Xs), np.concatenate(ys)

        # ---------- try new directory layout ----------
        base_dir = os.path.join(self.source_root, f"size{window_size}")
        X_train, y_train = _load_split(os.path.join(base_dir, "train"))
        X_val, y_val = _load_split(os.path.join(base_dir, "val"))
        X_test, y_test = _load_split(os.path.join(base_dir, "test"))

        for split, y in [('train', y_train), ('val', y_val), ('test', y_test)]:
            neg, pos = np.bincount(y)
            print(f'{split}: neg={neg}, pos={pos}, pos_ratio={pos / len(y):.2%}')

        # --- data augmentation only on the training split ---
        if X_train is not None:
            self.is_training = True
            X_train = np.array([self._augment_window(w) for w in X_train])
            self.is_training = False

        return X_train, y_train, X_val, y_val, X_test, y_test

    def save_model(self):
        self.model.save(f"{self.dest_root}/best_{self.model_name}_ws{self.window_size}_withT.keras",
                        include_optimizer=False)

    def export_tflite(self, representative_data_gen, tflite_path="model_int8.tflite"):
        """
        Export the trained Keras model as an INT8‑quantised TFLite model.

        Parameters
        ----------
        representative_data_gen : callable
            A generator that yields batches of representative input tensors
            for post‑training quantisation calibration.
        tflite_path : str
            Destination path for the .tflite file.
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"✅ INT8 TFLite model written to: {tflite_path}")


class PRCurveCallback(tf.keras.callbacks.Callback):
    """
    每个 epoch 结束后：
    1. 在验证集上跑一次预测
    2. 计算 PR-AUC
    3. 使用 matplotlib 画 PR 曲线
    4. 写到 TensorBoard（Scalars + Images）
    """

    def __init__(self, val_data, log_dir, prefix="val"):
        super().__init__()
        self.val_data = val_data  # tf.data 或 (X_val, y_val)
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        # 1. 收集预测结果
        if isinstance(self.val_data, tf.data.Dataset):
            y_true = np.concatenate([y.numpy() for _, y in self.val_data], axis=0)
            y_pred = np.concatenate([self.model.predict(X) for X, _ in self.val_data], axis=0)
        else:
            X_val, y_true = self.val_data
            y_pred = self.model.predict(X_val, verbose=0)

        # 2. precision-recall
        print("[DEBUG] y_true NaNs:", np.isnan(y_true).sum())
        print("[DEBUG] y_pred NaNs:", np.isnan(y_pred).sum())
        if np.isnan(y_true).sum() > 0 or np.isnan(y_pred).sum():
            print("[DEBUG] contains zero.")
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)

        # 3. 画图
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f"PR curve (AUC={pr_auc:.4f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        ax.grid(True)

        # 4. 写入 TensorBoard
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        with self.file_writer.as_default():
            tf.summary.scalar(f"{self.prefix}_pr_auc", pr_auc, step=epoch)
            tf.summary.image(f"{self.prefix}_pr_curve", image, step=epoch)
