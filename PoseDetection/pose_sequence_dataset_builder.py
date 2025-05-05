#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_builder.py

用途：
    从指定目录的视频和对应标签文件生成帧级和窗口级带标签训练数据。
    - 自动扫描视频目录下的 *.avi 和 *.mp4 文件
    - 对每个视频：提取关键点、计算差分与手工特征
    - 合并标签文件 <base>_labels.csv，生成帧级 label 列
    - 如果指定 window_size>1，则生成窗口级样本并保存

依赖：
    pip install opencv-python pandas numpy

用法：
    python pose_sequence_dataset_builder.py \
        --videos_dir ./raw_videos \
        --labels_dir ./raw_videos \
        --output_dir ./dataset \
        --window_size 32 \
        --stride 1

参数说明：
    --videos_dir   输入视频目录，支持 *.avi, *.mp4
    --labels_dir   标签文件目录，标签文件需命名为 <视频名>_labels.csv
    --output_dir   输出目录，保存 *_labeled.csv 和 *_windows.csv
    --window_size  窗口大小；=1 时仅生成帧级
    --stride       窗口滑动步长

示例：
    # 仅帧级标签
    python dataset_builder.py --videos_dir raw_videos --labels_dir raw_videos --output_dir dataset

    # 窗口级标签
    python dataset_builder.py --videos_dir raw_videos --labels_dir raw_videos --output_dir dataset --window_size 32 --stride 1
"""

import os
import sys
import glob
import cv2
import pandas as pd
import numpy as np
import argparse

from features import PoseEstimator, PoseFrame, DistanceCalculator, AngleCalculator
from utils.VideoStabilizer import VideoStabilizer
from utils.Differentiator import get_differentiator

# 将项目根目录加入模块搜索路径，以便能够导入顶层的 utils 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def extract_features(video_path):
    """提取指定视频的帧级特征，返回 DataFrame 包含 frame, timestamp, 原始关键点、速度、加速度、距离、角度等列"""

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps
    estimator = PoseEstimator()
    stabilizer = VideoStabilizer()

    # 定义距离和角度计算器
    distance_pairs = [(24, 26), (26, 28), (11, 13), (13, 15)]  # 臀-膝，膝-踝，肩-肘，肘-腕
    angle_triplets = [(24, 26, 28), (11, 13, 15), (23, 11, 13)]  # 髋-膝-踝，肩-肘-腕，躯干倾斜
    diff = get_differentiator()
    dist_calc = DistanceCalculator(distance_pairs)
    ang_calc = AngleCalculator(angle_triplets)

    records = []
    frame_idx = 0
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx * dt

        # 1) 全局抖动补偿
        stable_frame = stabilizer.stabilize(raw_frame)
        # 2) 姿态估计，获取 landmarks (NormalizedLandmarkList)
        landmarks, _ = estimator.estimate(stable_frame)
        if landmarks is None:
            # 如果没有检测到人体，填充空值
            # 构造长度与 raw 对应的零列表
            raw = [0.0] * (33 * 4)
            raw_px = [0.0] * (33 * 2)
            vel = [0.0] * (33 * 4)
            acc = [0.0] * (33 * 4)
            dists = [0.0] * len(distance_pairs)
            angs = [0.0] * len(angle_triplets)
        else:
            # 3) 原始坐标封装
            h, w = stable_frame.shape[:2]
            pf = PoseFrame(frame_idx, timestamp, landmarks.landmark, frame_size=(h, w))
            raw = pf.raw
            raw_px = pf.raw_px
            # 4) 差分
            vel, acc = diff.compute(raw)
            # 5) 距离与角度
            dists = dist_calc.compute(landmarks.landmark)
            angs = ang_calc.compute(landmarks.landmark)

        # 6) 拼接到一条记录
        rec = {
            'frame': frame_idx,
            'timestamp': timestamp
        }
        # raw x,y,z,vis
        for i in range(33):
            rec[f'x_{i}'] = raw[4 * i]
            rec[f'y_{i}'] = raw[4 * i + 1]
            rec[f'z_{i}'] = raw[4 * i + 2]
            rec[f'vis_{i}'] = raw[4 * i + 3]
        # 像素坐标特征
        for i in range(33):
            rec[f'x_px_{i}'] = raw_px[2*i]
            rec[f'y_px_{i}'] = raw_px[2*i+1]
        # velocity features
        for i in range(33):
            rec[f'vx_{i}'] = vel[4 * i]
            rec[f'vy_{i}'] = vel[4 * i + 1]
            rec[f'vz_{i}'] = vel[4 * i + 2]
            rec[f'vvis_{i}'] = vel[4 * i + 3]
        # acceleration features
        for i in range(33):
            rec[f'ax_{i}'] = acc[4 * i]
            rec[f'ay_{i}'] = acc[4 * i + 1]
            rec[f'az_{i}'] = acc[4 * i + 2]
            rec[f'avis_{i}'] = acc[4 * i + 3]
        # distances
        for idx, (a, b) in enumerate(distance_pairs):
            rec[f'dist_{a}_{b}'] = dists[idx]
        # angles
        for idx, (a, b, c) in enumerate(angle_triplets):
            rec[f'angle_{a}_{b}_{c}'] = angs[idx]

        records.append(rec)
        frame_idx += 1

    cap.release()
    return pd.DataFrame(records)


def merge_labels(df_feat, labels_path):
    """读取标签区间文件，合并到特征 DataFrame，生成 label 列"""
    ranges = pd.read_csv(labels_path)

    def in_rise(f):
        for _, row in ranges.iterrows():
            if row.start_frame <= f <= row.end_frame:
                return 1
        return 0

    df_feat['label'] = df_feat['frame'].apply(in_rise)
    return df_feat


def build_windows(df_labeled, window_size, stride):
    """基于滑动窗口生成窗口级样本，返回 DataFrame 每行包含 window_start, window_end, flatten_features..., label"""
    feature_cols = [c for c in df_labeled.columns if c not in ('frame', 'timestamp', 'label')]
    windows = []
    num_frames = len(df_labeled)
    for start in range(0, num_frames - window_size + 1, stride):
        end = start + window_size
        window = df_labeled.iloc[start:end]
        feats = window[feature_cols].values.flatten()
        lbl = int(window['label'].any())
        windows.append({
            'window_start': window['frame'].iloc[0],
            'window_end': window['frame'].iloc[-1],
            **{f'feat_{i}': feats[i] for i in range(len(feats))},
            'label': lbl
        })
    return pd.DataFrame(windows)


def main():
    parser = argparse.ArgumentParser(description='生成帧级和窗口级带标签训练数据')
    parser.add_argument('--videos_dir', default='raw_videos', help='输入视频目录，支持 *.avi, *.mp4')
    parser.add_argument('--labels_dir', default='raw_videos', help='标签目录，包含 *_labels.csv')
    parser.add_argument('--output_dir', default='dataset', help='输出目录，保存数据集')
    parser.add_argument('--window_size', default=32, type=int, help='窗口大小，=1 时仅帧级')
    parser.add_argument('--stride', default=1, type=int, help='滑动步长')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_patterns = [os.path.join(args.videos_dir, ext) for ext in ('*.avi', '*.mp4')]

    for pattern in video_patterns:
        for video_path in glob.glob(pattern):
            base = os.path.splitext(os.path.basename(video_path))[0]
            print(f'Processing {base}...')
            labels_path = os.path.join(args.labels_dir, f'{base}_labels.csv')
            if not os.path.exists(labels_path):
                print(f"  Skipping {base}: label file not found ({labels_path})")
                continue

            # 步骤1：特征提取
            df_feat = extract_features(video_path)

            # 步骤2：合并标签，生成帧级带label
            df_labeled = merge_labels(df_feat, labels_path)
            out_frame = os.path.join(args.output_dir, f'{base}_labeled.csv')
            df_labeled.to_csv(out_frame, index=False)
            print(f'  Saved frame-level data: {out_frame}')

            # 步骤3：窗口级数据（如果需要）
            if args.window_size > 1:
                df_win = build_windows(df_labeled, args.window_size, args.stride)
                out_win = os.path.join(args.output_dir, f'{base}_windows.csv')
                df_win.to_csv(out_win, index=False)
                print(f'  Saved window-level data: {out_win}')


if __name__ == '__main__':
    main()

# In features.py (not shown here), update PoseFrame class as instructed:

# class PoseFrame:
#     def __init__(self, frame_idx: int, timestamp: float, landmarks, frame_size=None):
#         self.frame_idx = frame_idx
#         self.timestamp = timestamp
#         self.raw = []
#         for lm in landmarks:
#             self.raw.extend([lm.x, lm.y, lm.z, lm.visibility])
#         if frame_size is not None:
#             h, w = frame_size
#             self.raw_px = []
#             for lm in landmarks:
#                 self.raw_px.extend([lm.x * w, lm.y * h])
