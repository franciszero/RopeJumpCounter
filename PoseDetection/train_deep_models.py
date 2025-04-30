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
import itertools
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score


def load_frame_data(dataset_dir, test_size=0.2, val_size=0.1, random_state=42):
    import glob, pandas as pd
    from sklearn.model_selection import train_test_split
    files = glob.glob(os.path.join(dataset_dir, "*_labeled.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    feature_cols = [c for c in df.columns if c not in ('frame','timestamp','label')]
    X = df[feature_cols].values
    y = df['label'].values
    # standardize or normalize if desired here
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction,
        random_state=random_state, stratify=y_trainval)
    # reshape to (samples, window_size=1, D) for compatibility
    D = X.shape[1]
    X_train = X_train.reshape(-1, 1, D)
    X_val   = X_val.reshape(-1, 1, D)
    X_test  = X_test.reshape(-1, 1, D)
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
    p.add_argument("--dataset_dir", default="./dataset")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--window_size", type=int, default=64)
    p.add_argument('--mode', choices=['frame','window'], default='window',
                   help='选择训练模式：frame（单帧级）或 window（滑窗级）')
    args = p.parse_args()

    os.makedirs('models', exist_ok=True)

    # 1. Load pre-split data
    if args.mode == 'frame':
        X_train, y_train_raw, X_val, y_val_raw, X_test, y_test_raw = \
            load_frame_data(args.dataset_dir, test_size=0.2, val_size=0.1)
    else:
        X_train, y_train_raw, X_val, y_val_raw, X_test, y_test_raw = \
            load_window_data(args.dataset_dir, args.window_size, test_size=0.2, val_size=0.1)

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

    # 2. 训练并评估 1D-CNN
    cnn = build_cnn_model(input_shape, num_classes)
    print("\n=== Training 1D-CNN ===")
    history_cnn = cnn.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=cb,
        class_weight=class_weight_dict
    )
    cnn.evaluate(X_test, y_test, verbose=2)
    cnn.save("models/1d_cnn_jump_classifier.h5")
    print("Saved 1D-CNN model to models/1d_cnn_jump_classifier.h5")

    # 3. 训练并评估 LSTM
    lstm = build_lstm_model(input_shape, num_classes)
    print("\n=== Training LSTM ===")
    history_lstm = lstm.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=cb,
        class_weight=class_weight_dict
    )
    lstm.evaluate(X_test, y_test, verbose=2)
    lstm.save("models/lstm_jump_classifier.h5")
    print("Saved LSTM model to models/lstm_jump_classifier.h5")

    # 5. 训练并评估 CRNN
    crnn = build_crnn_model(input_shape, num_classes)
    print("\n=== Training CRNN (Conv + BiLSTM) ===")
    history_crnn = crnn.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=cb,
        class_weight=class_weight_dict
    )
    crnn.evaluate(X_test, y_test, verbose=2)
    crnn.save("models/crnn_jump_classifier.h5")
    print("Saved CRNN model to models/crnn_jump_classifier.h5")

    # 4. 打印测试集详细报告
    from sklearn.metrics import classification_report
    y_pred_cnn = np.argmax(cnn.predict(X_test), axis=1)
    y_pred_lstm = np.argmax(lstm.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 6. CRNN 测试集报告
    y_pred_crnn = np.argmax(crnn.predict(X_test), axis=1)

    # 7. 训练并评估 1D ResNet
    resnet = build_resnet1d_model(input_shape, num_classes)
    print("\n=== Training ResNet1D ===")
    history_resnet = resnet.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=cb,
        class_weight=class_weight_dict
    )
    resnet.evaluate(X_test, y_test, verbose=2)
    resnet.save("models/resnet1d_jump_classifier.h5")
    print("Saved ResNet1D model to models/resnet1d_jump_classifier.h5")

    # ResNet1D 分类报告
    y_pred_resnet = np.argmax(resnet.predict(X_test), axis=1)

    # 8. 训练并评估 TCN 模型
    tcn = build_tcn_model(input_shape, num_classes)
    print("\n=== Training TCN Model ===")
    history_tcn = tcn.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=cb,
        class_weight=class_weight_dict
    )
    tcn.evaluate(X_test, y_test, verbose=2)
    tcn.save("models/tcn_jump_classifier.h5")
    print("Saved TCN model to models/tcn_jump_classifier.h5")

    # TCN 分类报告
    y_pred_tcn = np.argmax(tcn.predict(X_test), axis=1)

    # 9. 汇总所有模型分类报告
    from sklearn.metrics import classification_report
    print("\n=== All Models Classification Reports ===")
    model_reports = [
        ("1D-CNN", y_pred_cnn),
        ("LSTM", y_pred_lstm),
        ("CRNN", y_pred_crnn),
        ("ResNet1D", y_pred_resnet),
        ("TCN", y_pred_tcn),
    ]
    for name, y_pred in model_reports:
        print(f"\n--- {name} classification report ---")
        print(classification_report(y_true, y_pred, target_names=[str(c) for c in le.classes_]))


    # 生成综合报告图
    os.makedirs('models', exist_ok=True)
    curves = [
        ('1d_cnn', history_cnn),
        ('lstm', history_lstm),
        ('crnn', history_crnn),
        ('resnet1d', history_resnet),
        ('tcn', history_tcn),
    ]
    fig = plt.figure(constrained_layout=True, figsize=(12, 12))
    gs = GridSpec(3, 1, figure=fig)

    # 1) 损失曲线
    ax0 = fig.add_subplot(gs[0, 0])
    for key, hist in curves:
        ax0.plot(hist.history['loss'], label=f"{key}-train")
        ax0.plot(hist.history['val_loss'], linestyle='--', label=f"{key}-val")
    ax0.set_title('Training & Validation Loss')
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('Loss')
    ax0.legend(loc='upper right')

    # 2) 准确率曲线
    ax1 = fig.add_subplot(gs[1, 0])
    for key, hist in curves:
        ax1.plot(hist.history['accuracy'], label=f"{key}-train")
        ax1.plot(hist.history['val_accuracy'], linestyle='--', label=f"{key}-val")
    ax1.set_title('Training & Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')

    # 3) 各模型测试集 F1 分数表
    # Map class labels (e.g., 0,1) as strings
    class_names = [str(c) for c in le.classes_]
    neg_label, pos_label = class_names[0], class_names[1]
    metrics = []
    for name, y_pred in model_reports:
        rep = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True
        )
        metrics.append([
            name,
            rep['accuracy'],
            rep[pos_label]['f1-score'],
            rep[neg_label]['f1-score'],
        ])
    df = pd.DataFrame(
        metrics,
        columns=['Model','Accuracy','Jump F1','NonJump F1']
    ).set_index('Model')

    ax2 = fig.add_subplot(gs[2, 0])
    ax2.axis('off')
    tbl = ax2.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)

    summary_path = os.path.join('models', 'summary_report.png')
    fig.savefig(summary_path)
    print(f"Saved comprehensive report to {summary_path}")


if __name__ == "__main__":
    main()
