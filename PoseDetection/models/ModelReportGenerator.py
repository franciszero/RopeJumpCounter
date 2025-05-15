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
import numpy as np
from sklearn.metrics import f1_score


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

    def export_html(self, path):
        if self.summary_df is None:
            self.generate_summary()

        import plotly.graph_objs as go
        from plotly.io import to_html

        # Prepare summary table HTML
        summary_html = self.summary_df.sort_values(by="roc_auc", ascending=False).to_html(index=False)

        # Prepare visual comparison table rows
        rows_html = ""
        for model in self.models:
            model_name = getattr(model, "model_name", "Model")

            # Confusion Matrices at τ=0.50 and best τ*
            cm_block_html = ""
            if hasattr(model, "y_true") and hasattr(model, "y_prob"):
                y_true = model.y_true
                y_prob = model.y_prob

                # default τ = 0.50
                from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
                import matplotlib.pyplot as plt
                import seaborn as sns
                from io import BytesIO
                import base64

                def _cm_img(y_pred, title):
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(2.6, 2.2))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    disp.plot(ax=ax, cmap="viridis", colorbar=False)
                    ax.set_title(title, fontsize=10)
                    plt.tight_layout(pad=0.4)
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    img64 = base64.b64encode(buf.read()).decode("utf-8")
                    plt.close(fig)
                    return f'<img src="data:image/png;base64,{img64}" style="width:100%;">'

                # τ = 0.5
                cm_default_html = _cm_img((y_prob >= 0.5), "τ=0.50")

                # best τ* by F1 if available / cache on model
                best_tau = getattr(model, "best_threshold", None)
                if best_tau is None:
                    # brute search 0.05–0.95 step 0.01
                    taus = np.linspace(0.05, 0.95, 91)
                    f1s = [f1_score(y_true, y_prob >= t) for t in taus]
                    best_tau = taus[int(np.argmax(f1s))]
                cm_best_html = _cm_img((y_prob >= best_tau), f"τ={best_tau:.2f}")

                cm_block_html = (
                    '<div style="display:flex;flex-direction:column;gap:4px">'
                    f'{cm_default_html}{cm_best_html}</div>'
                )
            cm_img_html = cm_block_html

            # ROC Curve
            roc_img_html = ""
            rep = getattr(model, "report", {})
            if "roc_curve" in rep:
                roc_data = rep["roc_curve"]
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=roc_data["fpr"], y=roc_data["tpr"], mode='lines', name='ROC'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
                fig_roc.update_layout(title="ROC Curve",
                                      xaxis_title="FPR",
                                      yaxis_title="TPR",
                                      height=300,
                                      autosize=True)
                roc_img_html = to_html(fig_roc, include_plotlyjs=False, full_html=False, config={"responsive": True})

            # PR Curve
            pr_img_html = ""
            if "pr_curve" in rep:
                pr_data = rep["pr_curve"]
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=pr_data["recall"], y=pr_data["precision"], mode='lines', name='PR'))
                fig_pr.update_layout(title="Precision-Recall Curve",
                                     xaxis_title="Recall",
                                     yaxis_title="Precision",
                                     height=300,
                                     autosize=True)
                pr_img_html = to_html(fig_pr, include_plotlyjs=False, full_html=False, config={"responsive": True})

            # Loss Curve
            loss_img_html = ""
            if hasattr(model, "history") and model.history and "loss" in getattr(model.history, "history", {}):
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(y=model.history.history["loss"], mode='lines', name='train_loss'))
                if "val_loss" in model.history.history:
                    fig_loss.add_trace(go.Scatter(y=model.history.history["val_loss"], mode='lines', name='val_loss'))
                fig_loss.update_layout(title="Loss Curve",
                                       xaxis_title="Epoch",
                                       yaxis_title="Loss",
                                       height=300,
                                       autosize=True)
                loss_img_html = to_html(fig_loss, include_plotlyjs=False, full_html=False, config={"responsive": True})

            rows_html += f"""
            <tr>
                <td style="text-align:center; vertical-align:middle;">{model_name}</td>
                <td style="text-align:center; vertical-align:middle;">{cm_img_html}</td>
                <td style="text-align:center; vertical-align:middle;">{roc_img_html}</td>
                <td style="text-align:center; vertical-align:middle;">{pr_img_html}</td>
                <td style="text-align:center; vertical-align:middle;">{loss_img_html}</td>
            </tr>
            """

            visual_table_html = f"""
            <h2>Model Visual Comparisons</h2>
            <table class="viz-table">
                <colgroup>
                    <col style="width:10%;">
                    <col style="width:22%;">
                    <col style="width:24%;">
                    <col style="width:24%;">
                    <col style="width:24%;">
                </colgroup>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Confusion Matrix</th>
                        <th>ROC Curve</th>
                        <th>PR Curve</th>
                        <th>Loss Curve</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            """

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
            table-layout: fixed;
            margin-bottom: 40px;
        }
        th, td {
            border: 1px solid #dddddd;
            text-align: center;
            padding: 8px;
            vertical-align: middle;
            overflow: hidden;
        }
        th {
            background-color: #f2f2f2;
        }
        img {
            width: 100%;
            max-height: 240px;
            height: auto;
            margin-bottom: 20px;
        }
        .plotly-graph-div { width: 100% !important; }
        </style>
        """

        # Final HTML export
        with open(path, "w", encoding="utf-8") as f:
            f.write("<html><head>")
            f.write(style)
            f.write('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>')
            f.write("</head><body>")
            f.write("<h1>Model Evaluation Report</h1>")
            f.write("<h2>1. Summary Table</h2>")
            f.write("<p><i>Sorted by ROC AUC</i></p>")
            f.write(summary_html)
            f.write(visual_table_html)
            f.write("</body></html>")
        print(f"HTML report exported to: {path}")
