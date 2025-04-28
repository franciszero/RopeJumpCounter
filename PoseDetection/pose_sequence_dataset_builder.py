#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pose_sequence_dataset_builder.py

功能：
- 扫描指定目录下的视频文件（支持 *.mp4 和 *.avi 格式）
- 使用 MediaPipe 从每帧视频中提取 N×2 的关键点坐标序列
- 利用滑动窗口技术将视频关键点序列切分成形状为 (window_size, N*2) 的小段
- 为每个小段附加对应的标签（从文件名中提取）
- 按照指定的测试集和验证集比例，将数据划分为训练集、验证集和测试集
- 分别保存为 train.npz, val.npz, test.npz，每个文件包含特征 X 和标签 y

整体流程：
1. 使用 SequenceFileScanner 扫描视频文件路径
2. 使用 PoseSequenceBuilder 对每个视频提取关键点序列并切窗
3. 使用 DatasetSplitter 按比例划分数据集
4. 保存划分后的数据集至指定目录

python pose_sequence_dataset_builder.py \
  --input_dir ./raw_videos \
  --output_dir ./dataset \
  --window_size 64
"""

import os
import glob
import argparse

import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split


class SequenceFileScanner:
    """
    扫描指定目录下的视频文件，并按扩展名过滤。

    Attributes:
        input_dir (str): 视频文件所在目录。

    Methods:
        scan() -> List[str]:
            返回排序后的视频文件路径列表，匹配 .mp4 和 .avi。
            如果目录下没有匹配的视频文件，则抛出 FileNotFoundError。
    """

    def __init__(self, input_dir):
        self.input_dir = input_dir

    def scan(self):
        """
        扫描目录，返回所有符合扩展名的视频文件路径列表。

        Returns:
            List[str]: 视频文件完整路径列表，已排序。

        Raises:
            FileNotFoundError: 如果没有找到任何视频文件。
        """
        paths = []
        for ext in ("*.mp4", "*.avi"):
            paths.extend(glob.glob(os.path.join(self.input_dir, ext)))
        paths.sort()
        if not paths:
            raise FileNotFoundError(f"No video files found in {self.input_dir}")
        return paths


class PoseSequenceBuilder:
    """
    使用 MediaPipe 提取视频中的人体关键点序列，并基于滑动窗口切分成训练样本。

    Attributes:
        window_size (int): 滑动窗口的长度（帧数）。
        stride (int): 滑动窗口的步长（帧数）。
        extractor (PoseExtractor): 关键点提取器实例。
        scanner (SequenceFileScanner): 视频文件扫描器实例。

    Methods:
        build(input_dir: str) -> (np.ndarray, np.ndarray):
            扫描目录视频文件，提取所有视频的关键点序列并切窗，返回拼接后的数据和标签。
    """

    def __init__(self, window_size, stride):
        self.window_size = window_size
        self.stride = stride
        self.extractor = PoseExtractor()
        # 初始化扫描器，input_dir 由 build() 时传入
        self.scanner = SequenceFileScanner(input_dir=None)

    def _process_video(self, video_path):
        """
        从单个视频文件提取所有满足窗口条件的序列及对应标签。

        Args:
            video_path (str): 视频文件路径。

        Returns:
            tuple:
                - seqs (np.ndarray): 形状为 (M, window_size, D) 的关键点序列数组。
                - labels (np.ndarray): 形状为 (M,) 的标签数组。
        """
        fname = os.path.basename(video_path)
        # 从文件名中截取标签，假设格式为 label_XXX.ext
        label = fname.rsplit("_", 1)[0]
        print(f"Processing {fname} -> label={label}")

        # 调用外部函数进行序列切分
        X, y = build_sequences_from_video(
            video_path, label, self.extractor,
            W=self.window_size, S=self.stride
        )
        return X, y

    def build(self, input_dir):
        """
        扫描指定目录所有视频文件，提取并切分关键点序列，拼接所有视频数据。

        Args:
            input_dir (str): 包含视频文件的目录路径。

        Returns:
            tuple:
                - X_all (np.ndarray): 所有视频拼接后的序列数据，形状 (N, window_size, D)。
                - y_all (np.ndarray): 对应标签，形状 (N,)。

        Raises:
            FileNotFoundError: 如果目录下没有视频文件。
        """
        # 更新扫描器目录
        self.scanner.input_dir = input_dir
        video_paths = self.scanner.scan()

        all_seqs = []
        all_labels = []

        for path in video_paths:
            # 处理单个视频，获得序列和标签
            X, y = self._process_video(path)
            # 仅添加非空序列，避免空数据导致拼接错误
            if X.size > 0:
                all_seqs.append(X)
                all_labels.append(y)

        # 将所有视频的序列和标签沿第0维拼接
        X_all = np.concatenate(all_seqs, axis=0)
        y_all = np.concatenate(all_labels, axis=0)
        return X_all, y_all


class DatasetSplitter:
    """
    按指定比例将数据集划分为训练集、验证集和测试集。

    Attributes:
        test_size (float): 测试集比例（0-1）。
        val_size (float): 验证集比例（相对于剩余部分，0-1）。
        random_state (int): 随机种子，保证划分可复现。

    Methods:
        split(X: np.ndarray, y: np.ndarray) -> tuple:
            返回划分后的训练、验证、测试集特征与标签。
    """

    def __init__(self, test_size, val_size, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split(self, X, y):
        """
        按比例划分数据集，保证标签分布一致（分层抽样）。

        Args:
            X (np.ndarray): 特征数组，形状 (N, ...).
            y (np.ndarray): 标签数组，形状 (N,).

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # 先划分测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size,
            stratify=y, random_state=self.random_state
        )
        # 计算验证集占剩余数据的比例
        val_ratio = self.val_size / (1 - self.test_size)
        # 再从剩余数据中划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio,
            stratify=y_temp, random_state=self.random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test


def parse_args():
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 包含输入输出目录、窗口大小、步长、测试集和验证集比例等参数。
    """
    p = argparse.ArgumentParser(
        description="构建跳绳数据集（滑窗 + train/val/test 划分）")
    p.add_argument("--input_dir", required=True,
                   help="原始视频目录，文件名需含标签前缀，如 jump_001.mp4")
    p.add_argument("--output_dir", required=True,
                   help="输出目录，将保存 train.npz, val.npz, test.npz")
    p.add_argument("--window_size", type=int, default=64,
                   help="滑动窗口长度（帧）")
    p.add_argument("--stride", type=int, default=8,
                   help="滑动窗口步长（帧）")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="测试集比例（0-1）")
    p.add_argument("--val_size", type=float, default=0.2,
                   help="验证集比例，相对于剩余的比例（0-1）")
    return p.parse_args()


class PoseExtractor:
    """
    使用 MediaPipe Pose 模型提取单帧图像中的人体关键点坐标。

    Attributes:
        mp_pose: MediaPipe pose 模块。
        pose: MediaPipe pose 估计器实例。
        num_landmarks (int): 关键点数量。

    Methods:
        extract(frame: np.ndarray) -> np.ndarray or None:
            返回单帧关键点坐标数组，形状为 (num_landmarks*2,) 或 None（未检测到关键点）。
    """

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
        提取单帧图像的关键点坐标。

        Args:
            frame (np.ndarray): BGR 格式图像。

        Returns:
            np.ndarray or None: 形状为 (num_landmarks*2,) 的关键点坐标数组，
                                若未检测到关键点则返回 None。
        """
        # 转为 RGB 格式供 MediaPipe 处理
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            # 未检测到人体关键点
            return None
        lm = res.pose_landmarks.landmark
        arr = np.zeros((self.num_landmarks * 2,), dtype=np.float32)
        for i, lm_pt in enumerate(lm):
            # 提取 x, y 坐标
            arr[2 * i] = lm_pt.x
            arr[2 * i + 1] = lm_pt.y
        return arr


def build_sequences_from_video(video_path, label, extractor, W, S):
    """
    从单个视频中提取关键点序列，并利用滑动窗口切分成多个样本。

    Args:
        video_path (str): 视频文件路径。
        label (str): 该视频对应的标签。
        extractor (PoseExtractor): 关键点提取器实例。
        W (int): 窗口大小（帧数）。
        S (int): 窗口滑动步长（帧数）。

    Returns:
        tuple:
            - seqs (np.ndarray): 形状为 (M, W, D) 的序列样本数组。
            - lbls (np.ndarray): 形状为 (M,) 的标签数组。

    说明：
        - 如果窗口内含有任意一帧关键点缺失（None），则跳过该窗口。
        - D 为关键点维度（num_landmarks*2）。
    """
    cap = cv2.VideoCapture(video_path)
    feats = []
    while True:
        ret, frame = cap.read()
        if not ret:
            # 视频读取结束
            break
        vec = extractor.extract(frame)
        # 可能为 None，后续窗口过滤
        feats.append(vec)
    cap.release()

    T = len(feats)
    D = feats[0].shape[0] if feats and feats[0] is not None else 0

    seqs, lbls = [], []
    # 使用滑动窗口切分序列
    for start in range(0, T - W + 1, S):
        window = feats[start:start + W]
        # 跳过包含 None 的窗口，保证数据完整性
        if any(v is None for v in window):
            continue
        seqs.append(np.stack(window, axis=0))
        lbls.append(label)
    if seqs:
        return np.stack(seqs, axis=0), np.array(lbls)
    else:
        # 无有效窗口时返回空数组
        return np.zeros((0, W, D), dtype=np.float32), np.zeros((0,), dtype=str)


def main():
    """
    主函数，执行数据集构建流程：
    1. 解析命令行参数
    2. 构建关键点序列数据
    3. 按比例划分训练、验证、测试集
    4. 保存划分后的数据集文件
    """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 构建序列数据：扫描视频，提取关键点序列并切分滑窗
    builder = PoseSequenceBuilder(window_size=args.window_size,
                                  stride=args.stride)
    X_all, y_all = builder.build(args.input_dir)
    print(f"Total sequences: {X_all.shape[0]}, shape per seq: {X_all.shape[1:]}")

    # 2) 划分数据集：根据测试集和验证集比例，分层抽样划分数据
    splitter = DatasetSplitter(test_size=args.test_size,
                               val_size=args.val_size)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X_all, y_all)

    # 3) 保存数据集切分结果，分别保存 X 和 y
    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        out_path = os.path.join(args.output_dir, f"{split_name}.npz")
        np.savez(out_path, X=X_split, y=y_split)
        print(f"Saved {split_name} set to {out_path}")


if __name__ == "__main__":
    main()
