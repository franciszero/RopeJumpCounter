import matplotlib

matplotlib.use('Agg')
# train_deep_models.py

"""
使用 1D-CNN 和 LSTM 对跳绳序列做分类训练

用法：
python train_deep_models.py \
  --dataset_dir ./dataset \
  --mode window \
  --window_size 32 \
  --batch_size 32 \
  --epochs 100
"""

import os
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# --- Plotly for interactive plots ---
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# === 通用绘图函数 plot_curve ===
import matplotlib.pyplot as plt

def plot_curve(x_values_dict, y_values_dict, title, xlabel, ylabel):
    """
    通用绘图函数，可用于绘制 ROC 或 PR 曲线。

    Args:
        x_values_dict (dict): 每个模型的 x 轴数值（如 FPR 或 Recall）。
        y_values_dict (dict): 每个模型的 y 轴数值（如 TPR 或 Precision）。
        title (str): 图表标题。
        xlabel (str): x 轴标签。
        ylabel (str): y 轴标签。
    """
    plt.figure(figsize=(8, 6))
    for model_name in x_values_dict:
        plt.plot(x_values_dict[model_name], y_values_dict[model_name], label=model_name)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plt.ioff()
import itertools
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.base import BaseEstimator, ClassifierMixin

# === Additional reporting tools ===
from yellowbrick.classifier import ClassificationReport as YBClassificationReport
from imblearn.metrics import classification_report_imbalanced
import mlflow
import wandb
import logging

logger = logging.getLogger(__name__)
try:
    from pycaret.classification import setup as pyc_setup, compare_models as pyc_compare_models, pull as pyc_pull

    has_pycaret = True
except Exception as e:
    has_pycaret = False
    logger.warning(f"PyCaret import failed, skipping PyCaret reports: {e}")
from lime.lime_tabular import LimeTabularExplainer
import shap


# Sklearn wrapper for Keras models so they can be used with Yellowbrick
class SklearnKerasWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        import numpy as _np
        # Record class labels for compatibility
        self.classes_ = _np.unique(y)
        return self

    def predict(self, X):
        import numpy as _np
        preds = self.model.predict(X)
        return _np.argmax(preds, axis=1)


def load_frame_npz(dataset_dir, test_size=0.2, val_size=0.1, random_state=42):
    import glob
    from sklearn.model_selection import train_test_split
    files = glob.glob(os.path.join(dataset_dir, "*_labeled.npz"))
    Xs, ys = [], []
    for f in files:
        data = np.load(f)
        # 假设我们在 .npz 里保存了 'X' (n×D) 和 'y' (n,) 两个数组
        Xs.append(data["X"])
        ys.append(data["y"])
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    # 然后跟原来的流程一样做 train/test/val 划分
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction,
        random_state=random_state, stratify=y_trainval)
    # 调整形状以匹配模型输入 (samples, 1, D)
    D = X.shape[1]
    X_train = X_train.reshape(-1, 1, D)
    X_val = X_val.reshape(-1, 1, D)
    X_test = X_test.reshape(-1, 1, D)
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_window_npz(dataset_dir, window_size, test_size=0.2, val_size=0.1, random_state=42):
    import glob
    from sklearn.model_selection import train_test_split
    files = glob.glob(os.path.join(dataset_dir, "*_windows.npz"))
    Xs, ys = [], []
    for f in files:
        data = np.load(f)
        # 假设 .npz 里保存了 'X' 形状 (n, W, D) 和 'y' (n,)
        Xs.append(data["X"])
        ys.append(data["y"])
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    # 同样划分
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction,
        random_state=random_state, stratify=y_trainval)
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_window_data(dataset_dir, window_size, test_size=0.2, val_size=0.1, random_state=42):
    import glob, pandas as pd
    from sklearn.model_selection import train_test_split
    files = glob.glob(os.path.join(dataset_dir, "*_windows.csv"))
    Xs, ys = [], []
    for f in files:
        df = pd.read_csv(f)
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        X = df[feat_cols].values
        y = df['label'].values
        Xs.append(X)
        ys.append(y)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    D = X.shape[1] // window_size
    X = X.reshape(-1, window_size, D)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction,
        random_state=random_state, stratify=y_trainval)
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_cnn_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(64, 3, activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(256, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_lstm_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_crnn_model(input_shape, num_classes):
    """
    构建 CRNN 模型：多层 Conv1D + BatchNorm + MaxPool, 后接双向 LSTM, 再 Dense.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(256, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_resnet1d_model(input_shape, num_classes):
    """
    构建 1D ResNet 模型：Residual blocks with Conv1D + Add, followed by GlobalAveragePooling.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    # Residual blocks
    for filters in [64, 128, 256]:
        # Main path
        y = tf.keras.layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
        y = tf.keras.layers.Conv1D(filters, 3, padding='same')(y)
        # Shortcut path: project if channel dimension differs
        shortcut = x
        in_channels = tf.keras.backend.int_shape(x)[-1]
        if in_channels != filters:
            shortcut = tf.keras.layers.Conv1D(filters, 1, padding='same')(shortcut)
        # Combine
        x = tf.keras.layers.Add()([shortcut, y])
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
    # Classification head
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_tcn_model(input_shape, num_classes):
    """
    构建简易 1D-TCN 模型：使用多尺度膨胀卷积 (dilation conv) 捕捉时序依赖
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    # 膨胀率序列，可根据需要调整
    for d in [1, 2, 4, 8]:
        x = tf.keras.layers.Conv1D(64, 3, padding='causal', dilation_rate=d, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="./dataset_3")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--window_size", type=int, default=32)
    p.add_argument('--mode', choices=['frame', 'window'], default='window',
                   help='选择训练模式：frame（单帧级）或 window（滑窗级）')
    args = p.parse_args()

    os.makedirs('models', exist_ok=True)

    # 1. Load pre-split data
    if args.mode == 'frame':
        X_train, y_train_raw, X_val, y_val_raw, X_test, y_test_raw = \
            load_frame_npz(args.dataset_dir)
    else:
        X_train, y_train_raw, X_val, y_val_raw, X_test, y_test_raw = \
            load_window_npz(args.dataset_dir, args.window_size)

    print(f"Loaded: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    if args.mode == 'window':
        # Confirm window size
        assert X_train.shape[1] == args.window_size, "window_size mismatch"

    # 2. Encode labels to integers and one-hot
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw)
    y_val_enc = le.transform(y_val_raw)
    y_test_enc = le.transform(y_test_raw)
    # Calculate class weights for imbalanced data
    num_classes = len(le.classes_)
    cw = class_weight.compute_class_weight(
        'balanced',
        classes=np.arange(num_classes),
        y=y_train_enc
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(cw)}
    print("Class weights:", class_weight_dict)
    y_train = np.eye(num_classes, dtype=np.float32)[y_train_enc]
    y_val = np.eye(num_classes, dtype=np.float32)[y_val_enc]
    y_test = np.eye(num_classes, dtype=np.float32)[y_test_enc]
    print("Classes:", list(le.classes_))

    N, W, D = X_train.shape
    print(f"Train samples: {N}, window={W}, dim={D}, classes={num_classes}")

    input_shape = (W, D)
    # 回调：早停 + 保存最优
    cb = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ]
    cb.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, verbose=1
        )
    )
    cb.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/best_crnn.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    )

    # === 训练并评估 LSTM+Attention v1（如存在），以及 lstm_v1, cnn_v1, rf_v1, xgb_v1 ===
    # 保留 LSTM+Attention v1（如存在），以及 lstm_v1, cnn_v1, rf_v1, xgb_v1 的训练、评估与结果输出
    # 其余 *_v0 相关的训练、评估、路径定义均已移除
    # 训练 LSTM+Attention v1
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Multiply, Flatten, Softmax, Lambda
    from tensorflow.keras.models import Model
    import tensorflow.keras.backend as K

    def build_lstm_attention_model(input_shape, num_classes):
        inputs = Input(shape=input_shape)
        x = LSTM(128, return_sequences=True)(inputs)
        x = LSTM(64, return_sequences=True)(x)
        attention_scores = Dense(1, activation='tanh')(x)
        attention_scores = Flatten()(attention_scores)
        attention_weights = Softmax()(attention_scores)
        attention_weights = Lambda(lambda w: K.expand_dims(w, axis=-1))(attention_weights)
        context = Multiply()([x, attention_weights])
        context = Lambda(lambda x: K.sum(x, axis=1))(context)
        context = Dense(64, activation="relu")(context)
        context = Dropout(0.5)(context)
        outputs = Dense(num_classes, activation="softmax")(context)
        model = Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    y_true = np.argmax(y_test, axis=1)
    model_reports = []

    # LSTM+Attention v1
    print("\n=== Training LSTM+Attention v1 ===")
    lstm_att_v1 = build_lstm_attention_model(input_shape, num_classes)
    history_lstm_att_v1 = lstm_att_v1.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=cb,
        class_weight=class_weight_dict
    )
    lstm_att_v1.evaluate(X_test, y_test, verbose=2)
    lstm_att_v1.save("models/lstm_attention_v1.keras")
    print("Saved LSTM+Attention v1 model to models/lstm_attention_v1.keras")
    pred_lstm_att_v1 = np.argmax(lstm_att_v1.predict(X_test), axis=1)
    model_reports.append(("lstm_attention_v1", (y_true, pred_lstm_att_v1)))

    # LSTM v1
    print("\n=== Training LSTM v1 ===")
    lstm_v1 = build_lstm_model(input_shape, num_classes)
    history_lstm_v1 = lstm_v1.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=cb,
        class_weight=class_weight_dict,
        verbose=2
    )
    pred_lstm_v1 = np.argmax(lstm_v1.predict(X_test), axis=1)
    model_reports.append(("lstm_v1", (y_true, pred_lstm_v1)))

    # CNN v1
    print("\n=== Training CNN v1 ===")
    cnn_v1 = build_cnn_model(input_shape, num_classes)
    history_cnn_v1 = cnn_v1.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=cb,
        class_weight=class_weight_dict,
        verbose=2
    )
    pred_cnn_v1 = np.argmax(cnn_v1.predict(X_test), axis=1)
    model_reports.append(("cnn_v1", (y_true, pred_cnn_v1)))

    # CRNN v1
    print("\n=== Training CRNN v1 ===")
    crnn = build_crnn_model(input_shape, num_classes)
    history_crnn = crnn.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=cb,
        class_weight=class_weight_dict,
        verbose=2
    )
    pred_crnn = np.argmax(crnn.predict(X_test), axis=1)
    model_reports.append(("crnn_v1", (y_true, pred_crnn)))

    # ResNet1D v1
    print("\n=== Training ResNet1D v1 ===")
    resnet = build_resnet1d_model(input_shape, num_classes)
    history_resnet = resnet.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=cb,
        class_weight=class_weight_dict,
        verbose=2
    )
    pred_resnet = np.argmax(resnet.predict(X_test), axis=1)
    model_reports.append(("resnet1d_v1", (y_true, pred_resnet)))

    # TCN v1
    print("\n=== Training TCN v1 ===")
    tcn = build_tcn_model(input_shape, num_classes)
    history_tcn = tcn.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=cb,
        class_weight=class_weight_dict,
        verbose=2
    )
    pred_tcn = np.argmax(tcn.predict(X_test), axis=1)
    model_reports.append(("tcn_v1", (y_true, pred_tcn)))

    # === 训练 RF v1 和 XGB v1（如需要，可补充实现） ===
    # 这里只保留接口，具体实现可根据实际项目补充
    # 例如:
    # from sklearn.ensemble import RandomForestClassifier
    # from xgboost import XGBClassifier
    # ... 训练 rf_v1, xgb_v1 并添加到 model_reports

    # 9. 汇总模型分类报告和混淆矩阵等详细指标
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
    print("\n=== All Models Classification Reports ===")
    model_metric_table = []
    model_confusion_info = {}
    model_curve_info = {}
    model_class_reports = {}
    for name, (y_true_flat, y_pred) in model_reports:
        print(f"\n--- {name} classification report ---")
        print(classification_report(y_true_flat, y_pred, target_names=[str(c) for c in le.classes_]))
        # 分类报告
        rep = classification_report(y_true_flat, y_pred, output_dict=True)
        model_class_reports[name] = rep
        # 混淆矩阵
        cm = confusion_matrix(y_true_flat, y_pred)
        # 只支持二分类和多分类下的每类TP,TN,FP,FN
        confusion_stats = {}
        for idx, cls in enumerate(le.classes_):
            # TP: 预测为i且真实为i
            TP = cm[idx, idx]
            # FP: 预测为i但真实不为i
            FP = cm[:, idx].sum() - TP
            # FN: 真实为i但预测不为i
            FN = cm[idx, :].sum() - TP
            # TN: 其他
            TN = cm.sum() - TP - FP - FN
            confusion_stats[cls] = dict(TP=int(TP), TN=int(TN), FP=int(FP), FN=int(FN))
        model_confusion_info[name] = confusion_stats
        # 主要指标
        acc = rep['accuracy'] if 'accuracy' in rep else None
        # 支持数量
        support = {cls: rep[str(cls)]['support'] for cls in le.classes_}
        # 取macro avg
        macro_f1 = rep['macro avg']['f1-score']
        macro_precision = rep['macro avg']['precision']
        macro_recall = rep['macro avg']['recall']
        # ROC AUC & PR AUC
        # 模型名映射
        model_name_map = {
            "lstm_attention_v1": "LSTM+Att",
            "cnn_v1": "CNN",
            "lstm_v1": "LSTM",
            "crnn_v1": "CRNN",
            "resnet1d_v1": "ResNet1D",
            "tcn_v1": "TCN"
        }
        display_name = model_name_map.get(name, name)
        # 获取概率分数
        model_obj = None
        if display_name in ["LSTM+Att", "CNN", "LSTM", "CRNN", "ResNet1D", "TCN"]:
            model_obj = {
                "LSTM+Att": lstm_att_v1,
                "CNN": cnn_v1,
                "LSTM": lstm_v1,
                "CRNN": crnn,
                "ResNet1D": resnet,
                "TCN": tcn
            }[display_name]
        y_score = None
        y_scores = None
        if model_obj is not None:
            y_scores = model_obj.predict(X_test)
            if y_scores.ndim == 2 and y_scores.shape[1] > 1:
                # 二分类取正类
                y_score = y_scores[:, 1]
            else:
                y_score = y_scores.ravel()
        # ROC AUC 和 PR AUC
        roc_auc = None
        pr_auc = None
        if y_score is not None:
            try:
                roc_auc = roc_auc_score(y_true_flat, y_score)
                pr_auc = average_precision_score(y_true_flat, y_score)
            except Exception:
                roc_auc, pr_auc = None, None
        # 保存曲线
        if y_score is not None:
            fpr, tpr, _ = roc_curve(y_true_flat, y_score)
            precision, recall, _ = precision_recall_curve(y_true_flat, y_score)
            model_curve_info[name] = dict(
                fpr=fpr, tpr=tpr, precision=precision, recall=recall,
                roc_auc=roc_auc, pr_auc=pr_auc
            )
        # 汇总表
        model_metric_table.append({
            'model': display_name,
            'accuracy': acc,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'support': sum(support.values()) if hasattr(support, 'values') else None,
            'name_key': name
        })

    # 9. Interactive summary plot with Plotly
    os.makedirs('models', exist_ok=True)
    # === 汇总训练曲线 curves ===
    curves = [("lstm_attention_v1", history_lstm_att_v1)]
    curves.append(("lstm_v1", history_lstm_v1))
    curves.append(("cnn_v1", history_cnn_v1))
    curves.append(("crnn_v1", history_crnn))
    curves.append(("resnet1d_v1", history_resnet))
    curves.append(("tcn_v1", history_tcn))
    fig_plot = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=('Training & Validation Loss', 'Training & Validation Accuracy'))
    for key, hist in curves:
        epochs = list(range(1, len(hist.history['loss']) + 1))
        fig_plot.add_trace(go.Scatter(x=epochs, y=hist.history['loss'], name=f"{key}-train"), row=1, col=1)
        fig_plot.add_trace(go.Scatter(x=epochs, y=hist.history['val_loss'], name=f"{key}-val", line=dict(dash='dash')),
                           row=1, col=1)
        fig_plot.add_trace(go.Scatter(x=epochs, y=hist.history['accuracy'], name=f"{key}-train"), row=2, col=1)
        fig_plot.add_trace(
            go.Scatter(x=epochs, y=hist.history['val_accuracy'], name=f"{key}-val", line=dict(dash='dash')), row=2,
            col=1)
    fig_plot.update_layout(height=700, showlegend=True)
    interactive_summary = fig_plot.to_html(full_html=False, include_plotlyjs='cdn')

    # 10a. 生成交互式分类指标对比
    metrics = []
    for name, (y_true_flat, y_pred) in model_reports:
        rep = model_class_reports[name]
        cls = '1'  # 仅保留正例指标
        metrics.append({
            'model': name,
            'precision': rep[cls]['precision'],
            'recall': rep[cls]['recall'],
            'f1': rep[cls]['f1-score'],
            'support': rep[cls]['support']
        })
    df_metrics = pd.DataFrame(metrics)
    fig_metrics = go.Figure()
    for metric in ['precision', 'recall', 'f1']:
        fig_metrics.add_trace(go.Bar(
            x=df_metrics['model'],
            y=df_metrics[metric],
            name=metric,
            customdata=df_metrics['support'],
            hovertemplate='Model: %{x}<br>'+metric+': %{y:.3f}<br>Support: %{customdata}<extra></extra>'
        ))
    fig_metrics.update_layout(barmode='group', title='分类指标对比')
    interactive_metrics = fig_metrics.to_html(full_html=False, include_plotlyjs='cdn')

    # 10b. 生成交互式 ROC & PR 曲线 (Plotly)
    fig_roc = go.Figure()
    fig_pr = go.Figure()
    auc_dict = {}
    ap_dict = {}
    # 缩放坐标轴默认视野，便于观察左下角/右上角
    roc_xrange = [0, 0.2]
    roc_yrange = [0.8, 1.0]
    pr_xrange = [0.7, 1.0]
    pr_yrange = [0.7, 1.0]
    for name, info in model_curve_info.items():
        display_name = [m['model'] for m in model_metric_table if m['name_key'] == name][0]
        auc_val = info['roc_auc']
        ap_val = info['pr_auc']
        auc_dict[display_name] = auc_val
        ap_dict[display_name] = ap_val
        fig_roc.add_trace(go.Scatter(
            x=info['fpr'], y=info['tpr'], mode='lines',
            name=f"{display_name} (AUC={auc_val:.3f})",
            hovertemplate='FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra>'+display_name+'</extra>'
        ))
        fig_pr.add_trace(go.Scatter(
            x=info['recall'], y=info['precision'], mode='lines',
            name=f"{display_name} (AP={ap_val:.3f})",
            hovertemplate='Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra>'+display_name+'</extra>'
        ))
    fig_roc.update_layout(
        title='ROC Curve for All Models',
        xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
        xaxis=dict(range=roc_xrange), yaxis=dict(range=roc_yrange),
        legend_title='Model'
    )
    fig_pr.update_layout(
        title='Precision-Recall Curve for All Models',
        xaxis_title='Recall', yaxis_title='Precision',
        xaxis=dict(range=pr_xrange), yaxis=dict(range=pr_yrange),
        legend_title='Model'
    )
    interactive_roc = fig_roc.to_html(full_html=False, include_plotlyjs='cdn')
    interactive_pr = fig_pr.to_html(full_html=False, include_plotlyjs='cdn')

    # 本地生成 HTML 报告
    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='UTF-8'><title>本地模型评估报告</title>",
        "<style>body{font-family:sans-serif;padding:2em;}h2{margin-top:1.5em;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:0.5em;text-align:center;}.best-row{background:#ffe082;font-weight:bold;} .metric-scores {margin-top: 5px; font-size: 14px;}</style>",
        "</head><body>",
        "<h1>RopeJumpCounter 本地模型评估报告</h1>"
    ]
    # --- 指标表格(高亮最佳) ---
    # 排序主指标
    main_metric = "macro_f1"
    best_row_idx = None
    if main_metric in ["macro_f1", "roc_auc"]:
        sorted_table = sorted(model_metric_table, key=lambda x: (x[main_metric] if x[main_metric] is not None else -1), reverse=True)
        best_row_idx = 0
    else:
        sorted_table = model_metric_table
    # 汇总表格
    html.append("<h2>模型主要指标总览</h2>")
    html.append("<table><tr><th>模型</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC AUC</th><th>PR AUC</th><th>Support</th></tr>")
    for idx, row in enumerate(sorted_table):
        is_best = idx == best_row_idx
        tr_cls = ' class="best-row"' if is_best else ''
        html.append(f"<tr{tr_cls}><td>{row['model']}</td><td>{row['accuracy']:.3f}</td><td>{row['macro_precision']:.3f}</td><td>{row['macro_recall']:.3f}</td><td>{row['macro_f1']:.3f}</td><td>{row['roc_auc']:.3f}</td><td>{row['pr_auc']:.3f}</td><td>{row['support']}</td></tr>")
    html.append("</table>")
    # --- 每模型详细报告 ---
    for idx, row in enumerate(sorted_table):
        name = row['name_key']
        display_name = row['model']
        html.append(f"<h2>{display_name} 评估报告</h2>")
        # 指标表格
        rep = model_class_reports[name]
        df = pd.DataFrame(rep).transpose().round(3)
        html.append("<h3>分类报告</h3>")
        html.append(df.to_html(border=0))
        # 输出AUC/AP
        auc_val = row['roc_auc']
        ap_val = row['pr_auc']
        # 新增：插入统一格式的AUC/AP信息
        html.append(f"""
<div class="metric-scores">
    <p><strong>ROC AUC:</strong> {auc_val:.3f}</p>
    <p><strong>Average Precision:</strong> {ap_val:.3f}</p>
</div>
""")
        # 混淆矩阵
        html.append("<h3>混淆矩阵 (TP/TN/FP/FN)</h3>")
        confusion = model_confusion_info[name]
        html.append("<table><tr><th>类</th><th>TP</th><th>TN</th><th>FP</th><th>FN</th></tr>")
        for cls, vals in confusion.items():
            html.append(f"<tr><td>{cls}</td><td>{vals['TP']}</td><td>{vals['TN']}</td><td>{vals['FP']}</td><td>{vals['FN']}</td></tr>")
        html.append("</table>")
        # 曲线
        if name in model_curve_info:
            # 单模型ROC/PR曲线
            info = model_curve_info[name]
            # ROC
            fig_single_roc = go.Figure()
            fig_single_roc.add_trace(go.Scatter(
                x=info['fpr'], y=info['tpr'], mode='lines',
                name=f"{display_name} (AUC={auc_val:.3f})",
                hovertemplate='FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra></extra>'
            ))
            fig_single_roc.update_layout(title=f"{display_name} ROC Curve", xaxis_title='FPR', yaxis_title='TPR',
                                         xaxis=dict(range=[0, 0.2]), yaxis=dict(range=[0.8, 1.0]))
            html.append(fig_single_roc.to_html(full_html=False, include_plotlyjs='cdn'))
            # 新增：插入AUC/AP信息到曲线下方
            html.append(f"""
<div class="metric-scores">
    <p><strong>ROC AUC:</strong> {auc_val:.3f}</p>
    <p><strong>Average Precision:</strong> {ap_val:.3f}</p>
</div>
""")
            # PR
            fig_single_pr = go.Figure()
            fig_single_pr.add_trace(go.Scatter(
                x=info['recall'], y=info['precision'], mode='lines',
                name=f"{display_name} (AP={ap_val:.3f})",
                hovertemplate='Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra></extra>'
            ))
            fig_single_pr.update_layout(title=f"{display_name} PR Curve", xaxis_title='Recall', yaxis_title='Precision',
                                        xaxis=dict(range=[0.7, 1.0]), yaxis=dict(range=[0.7, 1.0]))
            html.append(fig_single_pr.to_html(full_html=False, include_plotlyjs=False))
            html.append(f"""
<div class="metric-scores">
    <p><strong>ROC AUC:</strong> {auc_val:.3f}</p>
    <p><strong>Average Precision:</strong> {ap_val:.3f}</p>
</div>
""")
    html.append("</body></html>")
    report_path = os.path.join('models', 'model_report.html')
    with open(report_path, "w", encoding="utf-8") as f_html:
        f_html.write("\n".join(html))
    print(f"本地HTML报告已生成: {os.path.abspath(report_path)}")

    # === 生成更多报告 ===
    # Flatten for reporting if using window mode
    if args.mode == 'window':
        X_train_rep = X_train.reshape(len(X_train), -1)
        X_test_rep = X_test.reshape(len(X_test), -1)
    else:
        X_train_rep = X_train
        X_test_rep = X_test

    # 1. Yellowbrick 报告 (示例: 1D-CNN)
    # Wrap the trained Keras model in our sklearn‑compatible class
    # skl_cnn = SklearnKerasWrapper(cnn)
    # viz = YBClassificationReport(skl_cnn, support=True)
    # # Use original windowed data for Yellowbrick
    # viz.fit(X_train, np.argmax(y_train, axis=1))
    # viz.score(X_test, np.argmax(y_test, axis=1))
    # viz.show(outpath="models/yellowbrick_1dcnn.png")
    # plt.close('all')

    # 2. Imbalanced-learn 报告
    for name, (y_true_flat, y_pred) in model_reports:
        rpt = classification_report_imbalanced(np.argmax(y_test, axis=1), y_pred)
        with open(f"models/imbalanced_{name}.txt", "w") as f:
            f.write(rpt)

    # 3. MLflow 记录所有分类报告
    mlflow.start_run()
    mlflow.log_params({"window_size": args.window_size, "mode": args.mode})
    for name, (y_true_flat, y_pred) in model_reports:
        txt = f"models/mlflow_{name}_report.txt"
        with open(txt, "w") as f:
            f.write(classification_report(y_true_flat, y_pred))
        mlflow.log_artifact(txt)
    mlflow.end_run()

    # 4. PyCaret 快速对比
    if has_pycaret:
        try:
            df_pyc = pd.DataFrame(X_train_rep)
            df_pyc['label'] = np.argmax(y_train, axis=1)
            # Initialize PyCaret so that pull() can access the experiment
            pyc_exp = pyc_setup(df_pyc, target='label', session_id=42)
            # Optionally run model comparison
            best = pyc_compare_models()
            # Pull and save the comparison results
            df_comp = pyc_pull()
            with open("models/pycaret_comparison_report.html", "w", encoding="utf-8") as f:
                f.write(df_comp.to_html(border=0))
        except Exception as e:
            logger.warning(f"PyCaret report generation failed, skipping: {e}")
    else:
        logger.warning("Skipping PyCaret report generation because PyCaret is unavailable.")

        # 5. LIME 解释 (示例: CRNN)
        try:
            # Prepare flat training data for LIME
            flat_train = X_train.reshape(len(X_train), -1) if args.mode == 'window' else X_train
            feature_names = [f"f{i}" for i in range(flat_train.shape[1])]
            explainer_lime = LimeTabularExplainer(
                flat_train,
                feature_names=feature_names,
                class_names=[str(c) for c in le.classes_],
                discretize_continuous=True
            )
            flat_test0 = flat_train[0]
            exp_lime = explainer_lime.explain_instance(
                flat_test0,
                lambda x: crnn.predict(x.reshape(-1, X_train.shape[1], X_train.shape[2]))
                if args.mode == 'window' else crnn.predict(x),
                num_features=10
            )
            exp_lime.save_to_file("models/lime_crnn.html")
            logger.info("LIME report generated: models/lime_crnn.html")
        except Exception as e:
            logger.warning(f"LIME explanation skipped due to error: {e}")

        # 6. SHAP 解释 (示例: CRNN)
        try:
            # Use DeepExplainer for Keras models
            background = X_train[:100] if args.mode == 'window' else X_train
            explainer_shap = shap.DeepExplainer(crnn, background)
            test_data = X_test[:50]
            shap_vals_all = explainer_shap.shap_values(test_data)
            # If shap_values returns a list per class, select the positive class
            shap_vals = shap_vals_all[1] if isinstance(shap_vals_all, list) else shap_vals_all
            shap.summary_plot(shap_vals, test_data, show=False)
            plt.savefig("models/shap_crnn.png")
            plt.close('all')
            logger.info("SHAP report generated: models/shap_crnn.png")
        except Exception as e:
            logger.warning(f"SHAP explanation skipped due to error: {e}")

    print("✅ 已生成所有报告：Yellowbrick, Imbalanced, MLflow, PyCaret, LIME, SHAP")

    # 10. 生成统一 HTML 报告
    unified = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='UTF-8'><title>统一模型评估报告</title>",
        "<style>body{font-family:sans-serif;padding:2em;}h2{margin-top:1.5em;}</style>",
        "</head><body>",
        "<h1>统一模型评估报告</h1>",
        "<h2>交互式 训练 & 验证 曲线</h2>",
        interactive_summary,
        "<h2>分类指标对比 (Precision/Recall/F1)</h2>",
        interactive_metrics,
        "<h2>ROC 曲线</h2>",
        interactive_roc,
        # 插入每个模型的AUC分数
        "<div>",
    ]
    for name, (y_true_flat, y_pred) in model_reports:
        model_name_map = {
            "lstm_attention_v1": "LSTM+Att",
            "cnn_v1": "CNN",
            "lstm_v1": "LSTM",
            "crnn_v1": "CRNN",
            "resnet1d_v1": "ResNet1D",
            "tcn_v1": "TCN"
        }
        display_name = model_name_map.get(name, name)
        if display_name in auc_dict and display_name in ap_dict:
            unified.append(f"""
<h3>Model: {display_name}</h3>
<ul>
  <li>ROC AUC: {auc_dict[display_name]:.4f}</li>
  <li>Average Precision (AP): {ap_dict[display_name]:.4f}</li>
</ul>
""")
    unified.append("</div>")
    unified += [
        "<h2>Precision-Recall 曲线</h2>",
        interactive_pr,
        "<h2>不平衡分类文本报告</h2>"
    ]
    for name, _ in model_reports:
        txt_path = f"models/imbalanced_{name}.txt"
        unified.append(f"<h3>{name}</h3>")
        unified.append("<pre>")
        try:
            with open(txt_path, "r", encoding="utf-8") as f_txt:
                unified.append(f_txt.read())
        except:
            unified.append("报告文件未找到。")
        unified.append("</pre>")
    unified.append("</body></html>")
    unified_path = os.path.join("models", "unified_report.html")
    with open(unified_path, "w", encoding="utf-8") as f_uni:
        f_uni.write("\n".join(unified))
    print(f"已生成统一报告: {os.path.abspath(unified_path)}")


if __name__ == "__main__":
    main()
