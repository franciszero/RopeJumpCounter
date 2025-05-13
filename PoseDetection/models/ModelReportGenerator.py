"""
ModelReportGenerator: A class for generating evaluation reports for a list of trained models.

Features:
- Summarizes per-model metrics (AUC, AP, accuracy, F1, etc.)
- Outputs:
  1. Summary table
  2. Training and validation loss curves
  3. ROC and PR curves
  4. Confusion matrices
- Exports to HTML with base64 inline charts
- Exports CSV summary
- Optionally displays per-model training time

Review Checklist:
✅ __init__: Accepts trained model list
✅ generate_summary: Collects evaluation metrics per model
✅ display: Prints sorted table
✅ export_csv: Saves summary
✅ export_html:
    - Table: via pandas DataFrame
    - Loss curves: from history
    - ROC/PR curves: from report
    - Confusion matrix: from y_true/y_pred
    - Inline PNGs: via base64
    - Layout/styling via inline CSS
"""

import pandas as pd


class ModelReportGenerator:
    def __init__(self, summary_df, loss_figs, roc_pr_figs, models=None):
        """
        Args:
            summary_df (pd.DataFrame): 模型评估摘要信息表格
            loss_figs (List[matplotlib.figure.Figure]): 每个模型的训练/验证 loss 曲线图
            roc_pr_figs (List[matplotlib.figure.Figure]): 每个模型的 ROC 和 PR 曲线图
            models (List[object], optional): 模型对象列表（可用于混淆矩阵生成）
        """
        self.summary_df = summary_df
        self.loss_figs = loss_figs
        self.roc_pr_figs = roc_pr_figs
        self.models = models if models is not None else []

    def generate_summary(self):
        rows = []
        for model in self.models:
            rep = model.report
            clf = rep.get("classification", {})
            row = {
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
            }
            rows.append(row)
        self.summary_df = pd.DataFrame(rows)

    def display(self):
        if self.summary_df is None:
            self.generate_summary()
        print(self.summary_df.sort_values(by="roc_auc", ascending=False))

    def export_csv(self, path="report_summary.csv"):
        if self.summary_df is None:
            self.generate_summary()
        self.summary_df.to_csv(path, index=False)

    def export_html(self, path="model_files/report_summary.html"):
        import matplotlib.pyplot as plt
        import seaborn as sns
        if self.summary_df is None:
            self.generate_summary()

        # ---- Generate loss curves for each model (if available) ----
        import base64
        from io import BytesIO

        figs_html = ""
        for model in self.models:
            if hasattr(model, "history") and model.history and "loss" in getattr(model.history, "history", {}):
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(model.history.history["loss"], label="train_loss")
                if "val_loss" in model.history.history:
                    ax.plot(model.history.history["val_loss"], label="val_loss")
                ax.set_title(f"Loss Curve: {getattr(model, 'model_name', 'Model')}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend()
                buf = BytesIO()
                plt.tight_layout()
                fig.savefig(buf, format="png")
                buf.seek(0)
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                plt.close(fig)
                figs_html += f'<h4>{getattr(model, "model_name", "Model")}</h4><img src="data:image/png;base64,{img_b64}"><hr>'

        # ---- Generate ROC and PR curves for each model (if available) ----
        roc_pr_html = ""
        for model in self.models:
            rep = getattr(model, "report", {})
            # ROC curve image or data
            if "roc_curve" in rep:
                roc_data = rep["roc_curve"]
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(roc_data["fpr"], roc_data["tpr"], label=f'ROC curve (area = {rep.get("roc_auc", 0):.2f})')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC Curve: {getattr(model, "model_name", "Model")}')
                ax.legend(loc="lower right")
                buf = BytesIO()
                plt.tight_layout()
                fig.savefig(buf, format="png")
                buf.seek(0)
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                plt.close(fig)
                roc_pr_html += f'<h4>{getattr(model, "model_name", "Model")} ROC Curve</h4><img src="data:image/png;base64,{img_b64}"><hr>'

            # PR curve image or data
            if "pr_curve" in rep:
                pr_data = rep["pr_curve"]
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(pr_data["recall"], pr_data["precision"],
                        label=f'PR curve (AP = {rep.get("average_precision", 0):.2f})')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(f'Precision-Recall Curve: {getattr(model, "model_name", "Model")}')
                ax.legend(loc="lower left")
                buf = BytesIO()
                plt.tight_layout()
                fig.savefig(buf, format="png")
                buf.seek(0)
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                plt.close(fig)
                roc_pr_html += f'<h4>{getattr(model, "model_name", "Model")} PR Curve</h4><img src="data:image/png;base64,{img_b64}"><hr>'

        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        import base64
        from io import BytesIO

        cm_html = ""
        for model in self.models:
            if hasattr(model, "y_true") and hasattr(model,
                                                    "y_pred") and model.y_true is not None and model.y_pred is not None:
                fig, ax = plt.subplots(figsize=(4, 4))
                ConfusionMatrixDisplay.from_predictions(model.y_true, model.y_pred, ax=ax)
                ax.set_title(f"Confusion Matrix: {model.model_name}")
                buf = BytesIO()
                plt.tight_layout()
                fig.savefig(buf, format="png")
                buf.seek(0)
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                plt.close(fig)
                cm_html += f'<h4>{model.model_name}</h4><img src="data:image/png;base64,{img_b64}"><hr>'

        # Evaluation Summary
        html = self.summary_df.sort_values(by="roc_auc", ascending=False).to_html(index=False)

        # Header + styles
        style = """
        <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        h2 {
            color: #333;
        }
        table {
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            border: 1px solid #dddddd;
            text-align: center;
            padding: 8px;
        }
        th {
            background-color: #f2f2f2;
        }
        img {
            max-width: 90%;
            height: auto;
            margin-bottom: 20px;
        }
        </style>
        """

        # Final HTML export
        with open(path, "w", encoding="utf-8") as f:
            f.write("<html><head>")
            f.write(style)
            f.write("</head><body>")
            f.write("<h1>Model Evaluation Report</h1>")
            f.write("<h2>1. Summary Table</h2>")
            f.write("<p><i>Sorted by ROC AUC</i></p>")
            f.write(html)
            f.write("<h2>2. Loss Curves</h2>")
            f.write(figs_html)
            f.write("<h2>3. ROC and PR Curves</h2>")
            f.write(roc_pr_html)
            f.write("<h2>4. Confusion Matrices</h2>")
            f.write(cm_html)
            f.write("</body></html>")
        print(f"HTML report exported to: {path}")
