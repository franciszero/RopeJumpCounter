import xgboost as xgb
from sklearn.metrics import precision_recall_curve, classification_report, roc_auc_score, average_precision_score, \
    roc_curve

from PoseDetection.models.BaseModel import TrainMyModel


class XGBModel(TrainMyModel):
    def __init__(self, name="xgb"):
        self.evals_result = None
        super().__init__(name)
        self._init_model()

    def _build(self):
        # 不需要构建神经网络，留空即可
        return None

    def train(self, **kwargs):
        # 1. 加载 frame-level 数据（size=1）
        #    加载后会得到 self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test

        # 2. 初始化 XGBoost 分类器
        self.model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric=["logloss", "auc"],
            **kwargs  # 比如 n_estimators, max_depth, learning_rate
        )

        # 3. 训练
        #    self.X_train 形状 (N_train, 403)，self.y_train 形状 (N_train,)
        self.history = self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)],
            verbose=True
        )

        # 获取训练过程指标
        self.evals_result = self.model.evals_result()

    def evaluate(self):
        self.y_prob = self.model.predict(self.X_test).flatten()
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_prob)
        self.y_pred = self.y_prob

        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        self.report = {
            'classification': classification_report(self.y_test, self.y_pred, output_dict=True),
            'roc_auc': roc_auc_score(self.y_test, self.y_prob),
            'average_precision': average_precision_score(self.y_test, self.y_prob),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
        }

    def save_model(self):
        dest_path = f"{self.dest_root}/best_{self.model_name}.json"
        self.model.save_model(dest_path)
