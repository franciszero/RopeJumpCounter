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
    python builder.py \
        --videos_dir ./raw_videos_3 \
        --labels_dir ./raw_videos_3 \
        --output_dir ./dataset_3 \
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
import argparse
from tqdm import tqdm
import logging
import numpy as np

from features import FeaturePipeline
from utils.VideoStabilizer import VideoStabilizer

# 将项目根目录加入模块搜索路径，以便能够导入顶层的 utils 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_features(video_path, window_size, logger):
    """提取指定视频的帧级特征，返回 DataFrame 包含 frame, timestamp, 原始关键点、速度、加速度、距离、角度等列"""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return pd.DataFrame()

    # 试着拿到总帧数，用来给 tqdm 一个 total；如果拿不到就设为 None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    pipe = FeaturePipeline(cap, window_size)

    records = []
    frame_idx = 0

    # 用 tqdm 包装循环
    pbar = tqdm(total=total_frames, desc=f"Extracting [{video_path}]", unit="frame")
    while True:
        try:
            if not pipe.success_process_frame(frame_idx):
                break
            records.append(pipe.fs.rec)
        except Exception as e:
            logger.warning(f"Frame {frame_idx} processing error: {e}, skipping")
        frame_idx += 1
        pbar.update(1)
    pbar.close()

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
    parser.add_argument('--videos_dir', default='raw_videos_3', help='输入视频目录，支持 *.avi, *.mp4')
    parser.add_argument('--labels_dir', default='raw_videos_3', help='标签目录，包含 *_labels.csv')
    parser.add_argument('--output_dir', default='dataset_3', help='输出目录，保存数据集')
    parser.add_argument('--window_size', default=32, type=int, help='窗口大小，=1 时仅帧级')
    parser.add_argument('--stride', default=1, type=int, help='滑动步长')

    # New stabilizer params
    parser.add_argument('--stabilizer_max_corners', default=VideoStabilizer.max_corners, type=int,
                        help='VideoStabilizer max corners')
    parser.add_argument('--stabilizer_quality_level', default=VideoStabilizer.quality_level, type=float,
                        help='VideoStabilizer quality level')
    parser.add_argument('--stabilizer_min_distance', default=VideoStabilizer.min_distance, type=int,
                        help='VideoStabilizer min distance')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_patterns = [os.path.join(args.videos_dir, ext) for ext in ('*.avi', '*.mp4')]

    for pattern in video_patterns:
        for video_path in glob.glob(pattern):
            base = os.path.splitext(os.path.basename(video_path))[0]
            logger.info(f'Processing {base}...')
            labels_path = os.path.join(args.labels_dir, f'{base}_labels.csv')
            if not os.path.exists(labels_path):
                logger.warning(f"  Skipping {base}: label file not found ({labels_path})")
                continue

            # 步骤1：特征提取
            df_feat = extract_features(video_path, args.window_size, logger)

            # 步骤2：合并标签，生成帧级带label
            df_labeled = merge_labels(df_feat, labels_path)
            # Save frame-level numpy data and report shapes
            X_frame = df_labeled.drop(columns=['frame', 'timestamp', 'label']).values
            y_frame = df_labeled['label'].values
            npz_frame = os.path.join(args.output_dir, f"{base}_labeled.npz")
            np.savez_compressed(npz_frame, X=X_frame, y=y_frame)
            logger.info(f'  Saved frame-level npz: {npz_frame}')
            print(f"Frame-level data shape: X={X_frame.shape}, y={y_frame.shape}")

            # 步骤3：窗口级数据（如果需要）
            if args.window_size > 1:
                # Build and save window-level numpy data without flattening
                feature_cols = [c for c in df_labeled.columns if c not in ('frame', 'timestamp', 'label')]
                X_win, y_win = [], []
                num_frames = len(df_labeled)
                for start in range(0, num_frames - args.window_size + 1, args.stride):
                    window = df_labeled.iloc[start:start+args.window_size]
                    arr = window[feature_cols].values  # shape: (window_size, feature_dim)
                    X_win.append(arr)
                    y_win.append(int(window['label'].any()))
                X_win = np.stack(X_win)  # shape: (n_windows, window_size, feature_dim)
                y_win = np.array(y_win)
                npz_win = os.path.join(args.output_dir, f"{base}_windows.npz")
                np.savez_compressed(npz_win, X=X_win, y=y_win)
                logger.info(f'  Saved window-level npz: {npz_win}')
                print(f"Window-level data shape: X={X_win.shape}, y={y_win.shape}")


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
