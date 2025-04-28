# train_model_selection.py

"""
python train_model_selection.py
"""

import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def load_data(dataset_dir="./dataset"):
    seqs = np.load(os.path.join(dataset_dir, "sequences.npy"))
    labels = np.load(os.path.join(dataset_dir, "labels.npy"))
    # 扁平化： (N, W, D) → (N, W*D)
    N, W, D = seqs.shape
    X = seqs.reshape(N, W * D)
    y = labels
    return X, y


def build_candidate_models():
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
        "RandomForest": RandomForestClassifier(n_estimators=200),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True))
        ]),
    }


def main():
    # 1. 加载数据
    X, y = load_data()
    print(f"Loaded {X.shape[0]} samples, feature dim={X.shape[1]}")

    # 2. 标签编码
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Classes:", list(le.classes_))

    # 3. 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc,
        test_size=0.2,
        stratify=y_enc,
        random_state=42
    )
    print(f"Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")

    # 4. 构建候选模型
    models = build_candidate_models()

    # 5. 交叉验证 & 验证集评估
    best_name, best_score = None, 0.0
    results = {}
    for name, model in models.items():
        print(f"\n>>> Evaluating {name}")
        # 5.1 5 折交叉验证 (训练集)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        print(f"  CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # 5.2 在训练集上训练，然后验证集评估
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"  Val accuracy: {acc:.4f}")
        print("  Classification report:")
        print(classification_report(y_val, y_pred, target_names=le.classes_))
        results[name] = acc

        # 更新最佳
        if acc > best_score:
            best_score, best_name = acc, name

    print(f"\n=== Best model: {best_name} with validation accuracy {best_score:.4f} ===")

    # 6. 保存最佳模型和 LabelEncoder
    os.makedirs("models", exist_ok=True)
    best_model = models[best_name]
    joblib.dump(best_model, f"models/{best_name}.joblib")
    joblib.dump(le, "models/label_encoder.joblib")
    print(f"Saved best model to models/{best_name}.joblib")
    print("Saved label encoder to models/label_encoder.joblib")


if __name__ == "__main__":
    main()
