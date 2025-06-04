import xgboost as xgb
from sklearn.metrics import precision_recall_curve, classification_report, roc_auc_score, average_precision_score, \
    roc_curve

from src.ml.models.BaseModel import TrainMyModel


class XGBModel(TrainMyModel):
    def __init__(self, name="xgb"):
        self.evals_result = None
        super().__init__(name)
        self._init_model()

    def _build(self):
        # No need to build neural network, leave empty
        return None

    def train(self, **kwargs):
        # 1. Load frame-level data (size=1)
        # After loading will get self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test

        # 2. Initialize XGBoost classifier
        self.model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric=["logloss", "auc"],
            **kwargs  # such as n_estimators, max_depth, learning_rate
        )

        # 3. Training
        # self.X_train shape (N_train, 403), self.y_train shape (N_train,)
        self.history = self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)],
            verbose=True
        )

        # get training process metrics
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
