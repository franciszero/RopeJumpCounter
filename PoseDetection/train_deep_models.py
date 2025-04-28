# train_deep_models.py

"""
使用 1D-CNN 和 LSTM 对跳绳序列做分类训练

用法：
  python train_deep_models.py \
    --dataset_dir ./dataset \
    --batch_size 32 \
    --epochs 30 \
    --window_size 64
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(dataset_dir, window_size):
    seqs = np.load(os.path.join(dataset_dir, "sequences.npy"))
    labels = np.load(os.path.join(dataset_dir, "labels.npy"))
    # 确保序列长度一致
    assert seqs.shape[1] == window_size, "window_size 与数据不匹配"
    return seqs, labels

def encode_labels(y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    # 转为 one-hot
    y_oh = tf.keras.utils.to_categorical(y_enc)
    return y_oh, le

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv1D(64, 3, activation="relu", input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(256, 3, activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_lstm_model(input_shape, num_classes):
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.LSTM(64),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir",  default="./dataset")
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--epochs",        type=int, default=30)
    p.add_argument("--window_size",   type=int, default=64)
    args = p.parse_args()

    # 1. 加载并准备数据
    X, y = load_data(args.dataset_dir, args.window_size)
    # X: (N, W, D); y: (N,)
    N, W, D = X.shape
    num_classes = len(np.unique(y))
    print(f"Loaded {N} samples, window={W}, dim={D}, classes={num_classes}")

    y_oh, le = encode_labels(y)

    # 分 train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_oh, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25,  # 0.25 * 0.8 = 0.2
        stratify=np.argmax(y_temp, axis=1), random_state=42
    )
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    input_shape = (W, D)
    # 回调：早停 + 保存最优
    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ]

    # 2. 训练并评估 1D-CNN
    cnn = build_cnn_model(input_shape, num_classes)
    print("\n=== Training 1D-CNN ===")
    cnn.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=cb
    )
    cnn.evaluate(X_test, y_test, verbose=2)
    cnn.save("models/1d_cnn_jump_classifier.h5")
    print("Saved 1D-CNN model to models/1d_cnn_jump_classifier.h5")

    # 3. 训练并评估 LSTM
    lstm = build_lstm_model(input_shape, num_classes)
    print("\n=== Training LSTM ===")
    lstm.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=cb
    )
    lstm.evaluate(X_test, y_test, verbose=2)
    lstm.save("models/lstm_jump_classifier.h5")
    print("Saved LSTM model to models/lstm_jump_classifier.h5")

    # 4. 打印测试集详细报告
    from sklearn.metrics import classification_report
    y_pred_cnn = np.argmax(cnn.predict(X_test), axis=1)
    y_pred_lstm = np.argmax(lstm.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n--- CNN classification report ---")
    print(classification_report(y_true, y_pred_cnn, target_names=le.classes_))
    print("\n--- LSTM classification report ---")
    print(classification_report(y_true, y_pred_lstm, target_names=le.classes_))

if __name__ == "__main__":
    main()