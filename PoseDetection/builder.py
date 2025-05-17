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
import matplotlib.pyplot as plt
import seaborn as sns
# from collections import Counter  # Removed as per instructions
import json
import datetime
import yaml  # pip install pyyaml  (optional for --split_yaml)
import random
from pathlib import Path

from features import FeaturePipeline
from utils.VideoStabilizer import VideoStabilizer

import matplotlib

# 可用字体：'Heiti SC'  # 或 'STHeiti', 'Songti SC', 'Arial Unicode MS', 'Hiragino Sans GB'
matplotlib.rcParams['font.family'] = 'Hiragino Sans GB'
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示负号

# 将项目根目录加入模块搜索路径，以便能够导入顶层的 utils 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 分析每个窗口中正例帧数量分布，并保存直方图
def analyze_window_label_distribution(df_labeled, window_size, output_dir, base):
    """分析每个窗口中包含多少个正例帧（label=1）"""
    label_counts = []
    labels = df_labeled['label'].values
    for i in range(0, len(labels) - window_size + 1):
        window = labels[i:i + window_size]
        label_counts.append(int(np.sum(window)))

    # 打印统计摘要
    unique, counts = np.unique(label_counts, return_counts=True)
    logger.info(f"[{base}] Window内跳跃帧数量分布（label=1）:")
    for u, c in zip(unique, counts):
        logger.info(f"  {u} 个跳跃帧的窗口: {c} 个")

    # 绘图
    plt.figure(figsize=(8, 4))
    sns.histplot(label_counts, bins=range(0, window_size + 2), discrete=True)
    plt.title(f"每个窗口中含跳跃帧数量分布 [{base}]")
    plt.xlabel("跳跃帧数量")
    plt.ylabel("窗口个数")
    plt.grid(True)
    plot_path = os.path.join(output_dir, f"{base}_label_dist.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"  跳跃帧分布图已保存: {plot_path}")


# 检查数组中是否存在长度不少于 min_len 的连续1
def has_continuous_ones(arr, min_len=3):
    count = 0
    for v in arr:
        if v == 1:
            count += 1
            if count >= min_len:
                return True
        else:
            count = 0
    return False


# 统计跳跃之间连续0和连续1的段长分布
def analyze_jump_stretch_distributions(df_labeled, output_dir, base):
    """
    统计跳跃之间连续0的段长分布和跳跃阶段连续1的段长分布
    """
    labels = df_labeled['label'].values
    zero_stretches = []
    one_stretches = []
    count = 0
    current_val = labels[0]

    for val in labels:
        if val == current_val:
            count += 1
        else:
            if current_val == 0:
                zero_stretches.append(count)
            else:
                one_stretches.append(count)
            current_val = val
            count = 1
    # Add the last stretch
    if current_val == 0:
        zero_stretches.append(count)
    else:
        one_stretches.append(count)

    from collections import Counter
    logger.info(f"[{base}] 跳跃之间连续0的段长分布（帧）:")
    for val, cnt in Counter(zero_stretches).most_common(10):
        logger.info(f"  {val} 帧: {cnt} 次")
    logger.info(f"[{base}] 跳跃阶段连续1的段长分布（帧）:")
    for val, cnt in Counter(one_stretches).most_common(10):
        logger.info(f"  {val} 帧: {cnt} 次")

    # 绘图
    plt.figure(figsize=(10, 5))
    if zero_stretches:
        sns.histplot(zero_stretches, bins=range(0, max(zero_stretches) + 2), color='blue', label='连续0段长', kde=False)
    if one_stretches:
        sns.histplot(one_stretches, bins=range(0, max(one_stretches) + 2), color='orange', label='连续1段长', kde=False)
    plt.yscale("log")
    plt.xlabel("段长（帧）")
    plt.ylabel("出现频次（对数）")
    plt.title(f"连续0与连续1的段长分布 [{base}]")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plot_path = os.path.join(output_dir, f"{base}_jump_stretch_dist.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"  跳跃帧段长分布图已保存: {plot_path}")


def extract_features(video_path, window_size, mode, logger):
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

    # 用 tqdm 包装循环，按照真实帧数上限遍历
    pbar = tqdm(total=total_frames, desc=f"Extracting [{video_path}]", unit="frame")
    while frame_idx < total_frames:
        try:
            ret, frame = cap.read()  # Original BGR frame (ignore latency)
            if not ret:
                break
            pipe.process_frame(frame, frame_idx, mode=mode)  # raw xyz, diff()
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


def main():
    parser = argparse.ArgumentParser(description='生成帧级和窗口级带标签训练数据')
    parser.add_argument('--videos_dir', default='raw_videos_3', help='输入视频目录，支持 *.avi, *.mp4')
    parser.add_argument('--labels_dir', default='raw_videos_3', help='标签目录，包含 *_labels.csv')
    parser.add_argument('--output_dir', default='dataset_10100', help='输出目录，保存数据集')
    parser.add_argument('--window_size', default=8, type=int, help='窗口大小，=1 时仅帧级')
    parser.add_argument('--stride', default=1, type=int, help='滑动步长')
    parser.add_argument('--mode', default=0b10100, type=int, help='pipeline 种允许生成哪些维度')

    # New stabilizer params
    parser.add_argument('--stabilizer_max_corners', default=VideoStabilizer.max_corners, type=int,
                        help='VideoStabilizer max corners')
    parser.add_argument('--stabilizer_quality_level', default=VideoStabilizer.quality_level, type=float,
                        help='VideoStabilizer quality level')
    parser.add_argument('--stabilizer_min_distance', default=VideoStabilizer.min_distance, type=int,
                        help='VideoStabilizer min distance')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='测试集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--split_yaml', default=None,
                        help='预定义划分文件（yaml: train/val/test 列表），若提供则覆盖随机划分')
    parser.add_argument('--preview_split', default=True, action='store_true',
                        help='仅预览 train/val/test 划分与正例数量，然后退出（不做特征提取）')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -------- 统计每个候选视频的正例数量并做贪心平衡划分 ---------
    video_stats = []
    mp4s = glob.glob(os.path.join(args.videos_dir, '*.mp4'))
    avis = glob.glob(os.path.join(args.videos_dir, '*.avi'))
    for vp in sorted(mp4s + avis):
        base = Path(vp).stem
        label_csv = Path(args.labels_dir) / f"{base}_labels.csv"
        if not label_csv.exists():
            logger.warning(f"Skip (no label): {vp}")
            continue
        try:
            ranges = pd.read_csv(label_csv)
            pos_frames = int((ranges.end_frame - ranges.start_frame + 1).sum())
        except Exception as e:
            logger.error(f"Failed to read labels for {vp}: {e}")
            continue
        total_frames = int(cv2.VideoCapture(vp).get(cv2.CAP_PROP_FRAME_COUNT))
        video_stats.append({'path': vp,
                            'pos': pos_frames,
                            'total': total_frames})

    if len(video_stats) < 3:
        logger.warning("可用视频少于3个，将全部划入 train；val/test 为空。")
        train_vids = {v['path'] for v in video_stats}
        val_vids, test_vids = set(), set()
    else:
        rng = random.Random(args.seed)
        rng.shuffle(video_stats)  # 打散顺序再按正例降序
        video_stats.sort(key=lambda x: x['pos'], reverse=True)

        target_ratio = np.array([1 - args.val_ratio - args.test_ratio,
                                 args.val_ratio,
                                 args.test_ratio], dtype=float)
        target_ratio /= target_ratio.sum()  # 归一化
        tot_pos = sum(v['pos'] for v in video_stats)
        deficits = target_ratio * tot_pos      # 初始还需多少正例
        splits = [set(), set(), set()]         # train, val, test

        for v in video_stats:
            idx = int(np.argmax(deficits))
            splits[idx].add(v['path'])
            deficits[idx] -= v['pos']

        train_vids, val_vids, test_vids = splits

    # ---- 统计日志 ----
    def sum_pos(s):
        return sum(v['pos'] for v in video_stats if v['path'] in s)

    logger.info(f"Train videos: {len(train_vids)} (pos={sum_pos(train_vids)}) | "
                f"Val: {len(val_vids)} (pos={sum_pos(val_vids)}) | "
                f"Test: {len(test_vids)} (pos={sum_pos(test_vids)})")

    # optional detailed debug
    for tag, group in zip(['TRAIN', 'VAL', 'TEST'], [train_vids, val_vids, test_vids]):
        for p in group:
            st = next(v for v in video_stats if v['path'] == p)
            logger.debug(f"[{tag}] {Path(p).name}: pos={st['pos']}, total={st['total']}")

    # ---------- 如果只预览划分，则输出详细信息后退出 ----------
    if args.preview_split:
        def _detail(group):
            return [
                {
                    "video": Path(p).name,
                    "pos_frames": next(v['pos'] for v in video_stats if v['path'] == p),
                    "total_frames": next(v['total'] for v in video_stats if v['path'] == p)
                } for p in group
            ]

        preview = {
            "train": _detail(train_vids),
            "val":   _detail(val_vids),
            "test":  _detail(test_vids)
        }
        import pprint
        pprint.pprint(preview, sort_dicts=False)
        logger.info("预览完成 (--preview_split). 未进行特征提取/文件写入。")
        return

    video_patterns = [os.path.join(args.videos_dir, ext) for ext in ('*.avi', '*.mp4')]

    for pattern in video_patterns:
        for video_path in glob.glob(pattern):
            base = os.path.splitext(os.path.basename(video_path))[0]
            split = ('train' if video_path in train_vids else
                     'val' if video_path in val_vids else
                     'test')
            logger.info(f'Processing {base}...')
            labels_path = os.path.join(args.labels_dir, f'{base}_labels.csv')
            if not os.path.exists(labels_path):
                logger.warning(f"  Skipping {base}: label file not found ({labels_path})")
                continue

            # 步骤1：特征提取
            df_feat = extract_features(video_path, args.window_size, args.mode, logger)

            # 步骤2：合并标签，生成帧级带label
            df_labeled = merge_labels(df_feat, labels_path)

            # --- 数据完整性检查 ---
            assert len(df_labeled) == len(df_feat), "Label merge length mismatch"
            assert df_labeled['frame'].is_monotonic_increasing, "Frame index not monotonic"
            pos_ratio = df_labeled['label'].mean()
            if pos_ratio < 0.01 or pos_ratio > 0.99:
                logger.warning(f"Extreme class imbalance detected: positive ratio={pos_ratio:.4f}")

            # 跳跃帧间隔与跳跃段长分布分析
            analyze_jump_stretch_distributions(df_labeled, args.output_dir, base)
            # Save frame-level numpy data and report shapes
            X_frame = df_labeled.drop(columns=['frame', 'timestamp', 'label']).values
            y_frame = df_labeled['label'].values
            npz_frame = os.path.join(args.output_dir, f"{base}_labeled.npz")
            np.savez_compressed(npz_frame, X=X_frame, y=y_frame)
            logger.info(f'  Saved frame-level npz: {npz_frame}')
            print(f"Frame-level data shape: X={X_frame.shape}, y={y_frame.shape}")

            # 步骤3：窗口级数据（多窗口大小）
            window_sizes = [4, 6, 8, 12, 16, 24, 32]

            # --- 新增 is_jump_window 函数 ---
            def is_jump_window(window):
                # 统一：窗口内出现 >=3 连续正例则记为正
                return has_continuous_ones(window['label'].values, min_len=3)

            for win_size in window_sizes:
                # Build and save window-level numpy data without flattening
                feature_cols = [c for c in df_labeled.columns if c not in ('frame', 'timestamp', 'label')]
                X_win, y_win = [], []
                num_frames = len(df_labeled)
                for start in range(0, num_frames - win_size + 1, args.stride):
                    window = df_labeled.iloc[start:start + win_size]
                    arr = window[feature_cols].values  # shape: (win_size, feature_dim)
                    X_win.append(arr)
                    lbl = is_jump_window(window)
                    y_win.append(int(lbl))
                if X_win:
                    X_win = np.stack(X_win)
                    y_win = np.array(y_win)
                else:
                    X_win = np.empty((0, win_size, len(feature_cols)))
                    y_win = np.empty((0,))
                # ---------- 保存 .npz ----------
                size_dir = os.path.join(args.output_dir, f"size{win_size}", split)
                os.makedirs(size_dir, exist_ok=True)
                npz_win = os.path.join(size_dir, f"{base}.npz")
                np.savez_compressed(npz_win, X=X_win, y=y_win,
                                    pos_ratio=float(y_win.mean()))
                logger.info(f'  Saved window-level npz: {npz_win}')
                print(f"Window-level data shape (size={win_size}): X={X_win.shape}, y={y_win.shape}")

                # meta.json (仅首次创建)
                meta_path = os.path.join(size_dir, 'meta.json')
                if not os.path.exists(meta_path):
                    meta = {
                        "window_size": win_size,
                        "feature_dim": int(len(feature_cols)),
                        "generated_at": datetime.datetime.utcnow().isoformat(),
                        "creator": "dataset_builder.py"
                    }
                    with open(meta_path, 'w') as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)

                from collections import Counter
                cnt = Counter(y_win)
                logger.info(
                    f"[{base}] size={win_size} ({split}) 标签分布：负类={cnt[0]}，正类={cnt[1]}，正例比例={(cnt[1] / (cnt[0] + cnt[1]) * 100):.2f}%")
                analyze_window_label_distribution(df_labeled, win_size,
                                                  os.path.join(args.output_dir, f"size{win_size}"),
                                                  f"{base}_size{win_size}")


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
