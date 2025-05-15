from PoseDetection.models.ModelReportGenerator import ModelReportGenerator
from PoseDetection.models.CNN import CNNModel
from PoseDetection.models.ResNET1D_TCNHybrid import ResNET1DTcnHybridModel
from PoseDetection.models.SEResNET1D import SEResNET1DModel
from PoseDetection.models.TCN import TCNModel
from PoseDetection.models.CRNN import CRNNModel
from PoseDetection.models.LSTM_Attention import LSTMAttentionModel
from PoseDetection.models.ResNET1D import ResNET1DModel
from PoseDetection.models.EfficientNet1D import EfficientNet1DModel
from PoseDetection.models.InceptionTime import InceptionTimeModel
from PoseDetection.models.TCN_SE import TCNSEModel
from PoseDetection.models.TFTLite import TFTLiteModel
from PoseDetection.models.TransformerLite import TransformerLiteModel

# 汇总各模型的评估数据
import pandas as pd
import matplotlib.pyplot as plt

from PoseDetection.models.WaveNet import WaveNetModel


class Trainer:
    def __init__(self):
        # Set up models with their respective window sizes
        self.models = [
            CNNModel(),
            # TCNModel(),
            # CRNNModel(),
            # LSTMAttentionModel(),
            # ResNET1DModel(),
            # EfficientNet1DModel(),
            # InceptionTimeModel(),
            # TransformerLiteModel(),
            # TFTLiteModel(),
            # SEResNET1DModel(),
            # WaveNetModel(),
            # TCNSEModel(),
            # ResNET1DTcnHybridModel(),
        ]

    def train(self):
        for m in self.models:
            m.train()
            m.evaluate()
            m.save_model()
        return


if __name__ == "__main__":
    tr = Trainer()
    tr.train()

    summary_rows = []
    loss_figs = []
    roc_pr_figs = []

    for model in tr.models:
        rep = model.report
        clf = rep.get("classification", {})
        summary_rows.append({
            "model": model.model_name,
            "window_size": model.window_size,
            "roc_auc": rep.get("roc_auc", None),
            "average_precision": rep.get("average_precision", None),
            "accuracy": clf.get("accuracy", None),
            "precision_1": clf.get("1", {}).get("precision", None),
            "recall_1": clf.get("1", {}).get("recall", None),
            "f1_1": clf.get("1", {}).get("f1-score", None),
            "support_1": clf.get("1", {}).get("support", None),
            "train_time_sec": getattr(model, "train_time", None)
        })

        # loss fig
        fig, ax = plt.subplots()
        ax.plot(model.history.history["loss"], label="train_loss")
        if "val_loss" in model.history.history:
            ax.plot(model.history.history["val_loss"], label="val_loss")
        ax.set_title(f"{model.model_name} Loss")
        ax.legend()
        loss_figs.append(fig)

        # roc/pr fig（如果有）
        if "roc_curve" in rep:
            fig, ax = plt.subplots()
            ax.plot(rep["roc_curve"]["fpr"], rep["roc_curve"]["tpr"], label="ROC")
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_title(f"{model.model_name} ROC")
            roc_pr_figs.append(fig)

    summary_df = pd.DataFrame(summary_rows)

    # 构造报告生成器
    dest_root = tr.models[0].dest_root
    reporter = ModelReportGenerator(
        summary_df=summary_df,
        loss_figs=loss_figs,
        roc_pr_figs=roc_pr_figs,
        models=tr.models
    )
    reporter.export_html(f"{dest_root}/report_summary.html")
