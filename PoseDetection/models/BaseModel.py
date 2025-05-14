import numpy as np
import joblib
from abc import ABC, abstractmethod
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, precision_recall_curve


class TrainMyModel(ABC):
    def __init__(self, name, dest_root="model_files", source_root="./dataset_3"):
        self.model_name = name
        self.dest_root = dest_root
        self.source_root = source_root
        self.num_classes = 2
        self.random_state = 42
        self.epochs = 100
        self.batch_size = 32

        # 配置: 各模型对应的 window_size
        self.MODEL_WINDOW_SIZES = {
            "cnn": 4,  # 8,
            "cnn_w8": 8,
            "lstm_attention": 4,  # 6,
            "lstm": 4,  # 6,
            "crnn": 4,  # 12,
            "resnet1d": 4,  # 16,
            "tcn": 4,  # 24,
            "inception": 4,
            "transformer": 4,  # 8,
            "efficientnet1d": 4,
        }

        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.y_true \
            = None, None, None, None, None, None, None
        self.history = None
        self.window_size = None
        self.y_prob = None  # self.model.predict(self.X_test).flatten()
        self.y_pred = None  # self.y_pred = (self.y_prob > 0.5).astype(int)
        self.report = None  # self.report = classification_report(self.y_test, self.y_pred, output_dict=True)
        self.model = None
        self.class_weight_dict = None

    def _init_model(self):
        self.window_size = self.MODEL_WINDOW_SIZES[self.model_name]
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test \
            = self.__load_window_npz(self.window_size)
        self.y_true = self.y_test
        weight = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        self.class_weight_dict = dict(enumerate(weight))
        self.model = self._build()

    @abstractmethod
    def _build(self):
        """子类必须实现：构建模型结构"""
        pass

    @abstractmethod
    def get_callbacks(self):
        """子类必须实现：callback"""
        pass

    def train(self):
        print("\n======================================================")
        print(f"\n============ Training {self.model_name} ============")
        print("\n======================================================")
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self.get_callbacks(),
            class_weight=self.class_weight_dict,
            verbose=2
        )
        self.model.save(f"{self.dest_root}/{self.model_name}_ws{self.window_size}.keras")
        pass

    def evaluate(self):
        self.y_prob = self.model.predict(self.X_test).flatten()
        self.y_pred = (self.y_prob > 0.5).astype(int)
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_prob)
        self.report = {
            'classification': classification_report(self.y_test, self.y_pred, output_dict=True),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred),
            'average_precision': average_precision_score(self.y_test, self.y_pred),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
        }
        self.save_report()

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

        # # ---------- fallback to legacy flat layout ----------
        # if X_train is None or X_val is None or X_test is None:
        #     files = glob.glob(os.path.join(self.source_root, f"*windows_size{window_size}.npz"))
        #     Xs, ys = [], []
        #     for f in files:
        #         data = np.load(f)
        #         Xs.append(data["X"])
        #         ys.append(data["y"])
        #     X = np.vstack(Xs)
        #     y = np.concatenate(ys)
        #
        #     # recreate splits as before
        #     X_trainval, X_test, y_trainval, y_test = train_test_split(
        #         X, y,
        #         test_size=self.test_size,
        #         random_state=self.random_state,
        #         stratify=y
        #     )
        #     val_fraction = self.val_size / (1 - self.test_size)
        #     X_train, X_val, y_train, y_val = train_test_split(
        #         X_trainval, y_trainval,
        #         test_size=val_fraction,
        #         random_state=self.random_state,
        #         stratify=y_trainval
        #     )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def save_model(self):
        self.model.save(f"{self.dest_root}/{self.model_name}.h5")

    def save_report(self):
        joblib.dump(self.report, f"{self.dest_root}/{self.model_name}_report.pkl")
