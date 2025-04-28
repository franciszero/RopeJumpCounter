# 文件名：pose_sequence_dataset_builder.py

"""
pose_sequence_dataset_builder.py

功能：
- 从带标签的视频文件夹中读取视频
- 用 MediaPipe 提取每帧 N×2 的关键点序列
- 以滑动窗口切分成 (window_size, N*2) 的小段，加上标签
- 按 6:2:2 拆分为 train/val/test
- 保存为 .npz 数据包：X_train, y_train, X_val, y_val, X_test, y_test

用法示例：
python pose_sequence_dataset_builder.py \
    --input_dir ./raw_videos \
    --output_file ./dataset.npz \
    --window_size 64 --stride 8 \
    --test_size 0.2 --val_size 0.2
"""

import os
import glob
import argparse

import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True,
                   help="视频文件所在目录，文件名需含动作标签前缀")
    p.add_argument("--output_file", required=True,
                   help="输出 .npz 文件路径")
    p.add_argument("--window_size", type=int, default=64,
                   help="滑动窗口长度（帧）")
    p.add_argument("--stride", type=int, default=8,
                   help="滑动窗口步长（帧）")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="测试集比例")
    p.add_argument("--val_size", type=float, default=0.2,
                   help="验证集比例（相对于剩余部分）")
    return p.parse_args()


class PoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.num_landmarks = len(self.mp_pose.PoseLandmark)

    def extract(self, frame):
        """
        返回 一个长度为 num_landmarks*2 的 numpy 数组：
        [x0,y0, x1,y1, ..., x32,y32]
        若检测失败，返回 None
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        arr = np.zeros((self.num_landmarks * 2,), dtype=np.float32)
        for i, lm_pt in enumerate(lm):
            arr[2 * i] = lm_pt.x  # 归一化 x
            arr[2 * i + 1] = lm_pt.y  # 归一化 y
        return arr


def build_sequences_from_video(video_path, label, extractor, W, S):
    """
    读取单个视频，返回 (M, W, D) 和 (M,)：
      M = floor((T - W)/S) + 1  窗口数
      D = extractor.num_landmarks*2
    只取有完整 W 帧检测成功的窗口。
    """
    cap = cv2.VideoCapture(video_path)
    feats = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        vec = extractor.extract(frame)
        feats.append(vec)
    cap.release()

    feats = np.array(feats)  # (T, D), 包含 None
    T, D = feats.shape
    seqs, lbls = [], []
    for start in range(0, T - W + 1, S):
        window = feats[start:start + W]
        # 如果窗口内任一帧检测失败（None），跳过
        if np.any(window == None):
            continue
        seqs.append(window.astype(np.float32))
        lbls.append(label)
    return np.stack(seqs, 0), np.array(lbls)


def main():
    args = parse_args()
    extractor = PoseExtractor()

    all_X, all_y = [], []
    # 假设所有视频文件名形如 "jump_001.mp4","rest_abc.mp4"... 前缀即标签
    for path in glob.glob(os.path.join(args.input_dir, "*.mp4")):
        fname = os.path.basename(path)
        label = fname.split("_")[0]
        print(f"Processing {fname} → label={label}")
        X, y = build_sequences_from_video(
            path, label, extractor,
            W=args.window_size, S=args.stride
        )
        all_X.append(X);
        all_y.append(y)

    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    print(f"Total sequences: {all_X.shape[0]}, feature dim: {all_X.shape[1:]}")

    # 先拆一部分为 test，再从剩余里拆 val
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        all_X, all_y, test_size=args.test_size, stratify=all_y, random_state=42
    )
    val_ratio = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio, stratify=y_trainval, random_state=42
    )
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # 保存
    np.savez(
        args.output_file,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    print(f"Saved dataset to {args.output_file}")


if __name__ == "__main__":
    main()
