from PoseDetection.models.ModelReportGenerator import ModelReportGenerator
from PoseDetection.models.CNN import *
from PoseDetection.models.XGBoost import XGBModel
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

import tensorflow as tf

# Mixed precision on GPU
from tensorflow.keras.mixed_precision import set_global_policy

set_global_policy('mixed_float16')

# 打印一下可见设备
print("Physical devices:", tf.config.list_physical_devices())
# # 打开 placement 日志
# tf.debugging.set_log_device_placement(True)

# Enable Metal GPU (MPS) if available and allow memory growth
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus, 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(f"Could not configure GPU devices: {e}")


class Trainer:
    def __init__(self):
        # Set up models with their respective window sizes
        self.models = [
            CRNNModel(),
            EfficientNet1DModel(),
            InceptionTimeModel(),
            LSTMAttentionModel(),
            ResNET1DModel(),
            ResNET1DTcnHybridModel(),
            SEResNET1DModel(),
            TCNModel(),
            TCNSEModel(),
            TFTLiteModel(),
            TransformerLiteModel(),
            WaveNetModel(),
            CNN8_1(),
            # XGBModel(),
            CNNModel(),
            CNNHybridModel(),
            CNN1(),
            CNN2(),
            CNN3(),
            CNN4(),
            CNN5(),
            CNN6(),
            CNN7(),
            CNN8(),
            CNN9(),
        ]

    def train(self):
        for mo in self.models:
            mo.train()
            mo.evaluate()
            mo.save_model()
        return


if __name__ == "__main__":
    tr = Trainer()
    tr.train()

    summary_rows = []
    loss_figs = []
    roc_pr_figs = []

    for m in tr.models:
        rep = m.report
        clf = rep.get("classification", {})
        summary_rows.append({
            "model": m.model_name,
            "window_size": m.window_size,
            "roc_auc": rep.get("roc_auc", None),
            "average_precision": rep.get("average_precision", None),
            "accuracy": clf.get("accuracy", None),
            "precision_1": clf.get("1", {}).get("precision", None),
            "recall_1": clf.get("1", {}).get("recall", None),
            "f1_1": clf.get("1", {}).get("f1-score", None),
            "support_1": clf.get("1", {}).get("support", None),
            "train_time_sec": getattr(m, "train_time", None)
        })

        # loss fig
        fig, ax = plt.subplots()
        if hasattr(m, "history") and hasattr(m.history, "history"):
            ax.plot(m.history.history["loss"], label="train_loss")
            if "val_loss" in m.history.history:
                ax.plot(m.history.history["val_loss"], label="val_loss")
        else:
            ax.plot(m.evals_result["validation_0"]["logloss"], label="train_loss")
            ax.plot(m.evals_result["validation_1"]["logloss"], label="val_loss")
        ax.set_title(f"{m.model_name} Loss")
        ax.legend()
        loss_figs.append(fig)

        # roc/pr fig（如果有）
        if "roc_curve" in rep:
            fig, ax = plt.subplots()
            ax.plot(rep["roc_curve"]["fpr"], rep["roc_curve"]["tpr"], label="ROC")
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_title(f"{m.model_name} ROC")
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
