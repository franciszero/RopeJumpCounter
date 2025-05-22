#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import json
import datetime
import random
from pathlib import Path

from PoseDetection.data_builder_utils.feature_mode import mode_to_str, get_feature_mode
from features import FeaturePipeline
from utils.FrameSample import SELECTED_LM
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
def analyze_window_label_distribution(labels, window_size, output_dir, base):
    """分析每个窗口中包含多少个正例帧（label=1）"""
    label_counts = []
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
def analyze_jump_stretch_distributions(labels, dest_path, video_name):
    """
    统计跳跃之间连续0的段长分布和跳跃阶段连续1的段长分布
    """
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
    logger.info(f"[{video_name}] 跳跃之间连续0的段长分布（帧）:")
    for val, cnt in Counter(zero_stretches).most_common(10):
        logger.info(f"  {val} 帧: {cnt} 次")
    logger.info(f"[{video_name}] 跳跃阶段连续1的段长分布（帧）:")
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
    plt.title(f"连续0与连续1的段长分布 [{video_name}]")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plot_path = os.path.join(dest_path, f"{video_name}_jump_stretch_dist.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"  跳跃帧段长分布图已保存: {plot_path}")


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

    # 用 tqdm 包装循环，按照真实帧数上限遍历
    pbar = tqdm(total=total_frames, desc=f"Extracting [{video_path}]", unit="frame")
    while frame_idx < total_frames:
        try:
            ret, frame = cap.read()  # Original BGR frame (ignore latency)
            if not ret:
                break
            pipe.process_frame(frame, frame_idx)  # raw xyz, diff()
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


def build_labeled_dataset(df_labeled, dest_file):
    pos_ratio = df_labeled['label'].mean()
    if pos_ratio < 0.01 or pos_ratio > 0.99:
        logger.warning(f"Extreme class imbalance detected: positive ratio={pos_ratio:.4f}")

    # Save frame-level numpy data and report shapes
    X = df_labeled.drop(columns=['frame', 'timestamp', 'label']).values
    y = df_labeled['label'].values
    np.savez_compressed(dest_file, X=X, y=y)
    logger.info(f'  Saved frame-level npz: {dest_file}')
    print(f"Frame-level data shape: X={X.shape}, y={y.shape}")
    return X, y


def gradient_split(args):
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

    rng = random.Random(args.seed)
    rng.shuffle(video_stats)  # 打散顺序再按正例降序
    video_stats.sort(key=lambda x: x['pos'], reverse=True)

    target_ratio = np.array([1 - args.val_ratio - args.test_ratio, args.val_ratio, args.test_ratio], dtype=float)
    target_ratio /= target_ratio.sum()  # 归一化
    tot_pos = sum(v['pos'] for v in video_stats)
    deficits = target_ratio * tot_pos  # 初始还需多少正例
    splits = {"train": list(),
              "val": list(),
              "test": list()
              }  # train, val, test

    for v in video_stats:
        idx = int(np.argmax(deficits))
        key = list(splits.keys())[idx]
        splits[key].append(v)
        deficits[idx] -= v['pos']

    # ---- 统计日志 ----
    def sum_pos(vs):
        return sum([v['pos'] for v in splits['train']])

    logger.info(f"Train videos: {len(splits['train'])} (pos={sum_pos(splits['train'])}) | "
                f"Val: {len(splits['val'])} (pos={sum_pos(splits['val'])}) | "
                f"Test: {len(splits['test'])} (pos={sum_pos(splits['test'])})")

    # ---------- 如果只预览划分，则输出详细信息后退出 ----------
    if args.preview_split:
        def _detail(vid_dicts):
            return [
                {
                    "video": Path(v['path']).name,
                    "pos_frames": v['pos'],
                    "total_frames": v['total']
                }
                for v in vid_dicts
            ]

        preview = {
            "train": _detail(splits['train']),
            "val": _detail(splits['val']),
            "test": _detail(splits['test']),
        }
        import pprint
        pprint.pprint(preview, sort_dicts=False)
        logger.info("预览完成 (--preview_split). 未进行特征提取/文件写入。")

    return splits


def building_win_dataset(X, y, win_size, stride):
    X_win, y_win = [], []
    num_frames = X.shape[0]
    for start in range(0, num_frames - win_size + 1, stride):
        X1 = X[start: start + win_size]
        y1 = y[start: start + win_size]
        X_win.append(X1)
        y_win.append(int(has_continuous_ones(y1)))
    X_win = np.stack(X_win)
    y_win = np.array(y_win)
    # X_win = np.empty((0, win_size, len(feature_cols)))
    # y_win = np.empty((0,))
    return X_win, y_win


def main():
    args, output_dir = get_command_line_params()

    # -------- 统计每个候选视频的正例数量并做贪心平衡划分 ---------
    splits = gradient_split(args)
    if args.preview_split:
        return

    for split_dest_set, videos in splits.items():
        for video in videos:
            video_file = video['path']
            video_name = os.path.splitext(os.path.basename(video_file))[0]

            labels_path = os.path.join(args.labels_dir, f'{video_name}_labels.csv')
            if not os.path.exists(labels_path):
                logger.warning(f"Skipping {video_name}: label file not found ({labels_path})")
                continue
            else:
                logger.info(f'Processing {video_name}...')

            # ------------- Build labeled dataset -------------
            dest_path = f"{output_dir}/size1/{split_dest_set}"
            os.makedirs(dest_path, exist_ok=True)
            dest_file = f"{dest_path}/{video_name}_labeled.npz"
            if not os.path.exists(dest_file):
                # 步骤1：特征提取
                df_feat = extract_features(video_file, args.window_size, logger)
                # 步骤2：合并标签，生成帧级带label
                df_labeled = merge_labels(df_feat, labels_path)
                # --- 数据完整性检查 ---
                assert len(df_labeled) == len(df_feat), "Label merge length mismatch"
                assert df_labeled['frame'].is_monotonic_increasing, "Frame index not monotonic"
                # build_labeled_dataset
                X, y = build_labeled_dataset(df_labeled, dest_file)
                # 跳跃帧间隔与跳跃段长分布分析
                analyze_jump_stretch_distributions(y, dest_path, video_name)
            else:
                npz_dic = np.load(dest_file)
                X, y = npz_dic["X"], npz_dic["y"]

            # 步骤3：窗口级数据（多窗口大小）
            window_sizes = [4, 5, 6]  # [4, 5, 6, 8, 12, 16, 24, 32]

            for win_size in window_sizes:
                X_win, y_win = building_win_dataset(X, y, win_size, args.stride)
                # ---------- 保存 .npz ----------
                dest_path = f"{output_dir}/size{win_size}/{split_dest_set}"
                os.makedirs(dest_path, exist_ok=True)
                dest_file = f"{dest_path}/{video_name}.npz"
                np.savez_compressed(dest_file, X=X_win, y=y_win, pos_ratio=float(y_win.mean()))
                logger.info(f'  Saved window-level npz: {dest_file}')
                print(f"Window-level data shape (size={win_size}): X={X_win.shape}, y={y_win.shape}")

                # meta.json (仅首次创建)
                meta_path = os.path.join(dest_path, 'meta.json')
                if not os.path.exists(meta_path):
                    meta = {
                        "window_size": win_size,
                        "feature_dim": int(len(y_win)),
                        "generated_at": datetime.datetime.utcnow().isoformat(),
                        "creator": "dataset_builder.py"
                    }
                    with open(meta_path, 'w') as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)

                from collections import Counter
                cnt = Counter(y_win)
                logger.info(f"[{video_name}] size={win_size} ({split_dest_set})" +
                            f" 标签分布：负类={cnt[0]}，正类={cnt[1]}，" +
                            f"正例比例={(cnt[1] / (cnt[0] + cnt[1]) * 100):.2f}%")
                analyze_window_label_distribution(y_win, win_size,
                                                  os.path.join(output_dir, f"size{win_size}"),
                                                  f"{video_name}_size{win_size}")


def get_command_line_params():
    parser = argparse.ArgumentParser(description='生成帧级和窗口级带标签训练数据')
    parser.add_argument('--videos_dir', default='../data/raw_videos_3', help='输入视频目录，支持 *.avi, *.mp4')
    parser.add_argument('--labels_dir', default='../data/raw_videos_3', help='标签目录，包含 *_labels.csv')
    parser.add_argument('--output_dir', default='../data/dataset', help='输出目录，保存数据集')
    parser.add_argument('--window_size', default=8, type=int, help='窗口大小，=1 时仅帧级')
    parser.add_argument('--stride', default=1, type=int, help='滑动步长')
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
    parser.add_argument('--preview_split', default=False, action='store_true',
                        help='仅预览 train/val/test 划分与正例数量，然后退出（不做特征提取）')
    args = parser.parse_args()

    suffix = mode_to_str(get_feature_mode())
    output_dir = f"{args.output_dir}_{len(SELECTED_LM)}_{suffix}"
    os.makedirs(output_dir, exist_ok=True)
    return args, output_dir


if __name__ == '__main__':
    main()
