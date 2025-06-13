"""
Builder data processing module

Handles data processing and feature extraction for jump rope analysis.
"""

# !/usr/bin/env python3
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

from .feature_mode import mode_to_str, get_feature_mode
from ..features.features import FeaturePipeline
from src.utils.common.FrameSample import SELECTED_LM
from src.utils.VideoStabilizer import VideoStabilizer

import matplotlib

# Available fonts: 'Heiti SC' or 'STHeiti', 'Songti SC', 'Arial Unicode MS', 'Hiragino Sans GB'
matplotlib.rcParams['font.family'] = 'Hiragino Sans GB'
matplotlib.rcParams['axes.unicode_minus'] = False  # Display minus sign

# Add project root directory to module search path, to import top-level utils package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Analyze positive example frame quantity distribution in each window and save histogram
def analyze_window_label_distribution(labels, window_size, output_dir, base):
    """Analyze Window Label Distribution

    Analyzes the distribution of positive labels within sliding windows
    and generates a histogram visualization.

    Args:
        labels: Array of binary labels (0/1)
        window_size: Size of sliding window
        output_dir: Directory to save the histogram plot
        base: Base name for the output file

    Returns:
        None (saves plot and logs statistics)
    """
    label_counts = []
    for i in range(0, len(labels) - window_size + 1):
        window = labels[i:i + window_size]
        label_counts.append(int(np.sum(window)))

    # Print statistical summary
    unique, counts = np.unique(label_counts, return_counts=True)
    logger.info(f"[{base}] Window jump frames quantity distribution (label=1):")
    for u, c in zip(unique, counts):
        logger.info(f"  {u} windows with jumps: {c} count")

    # Draw plot
    plt.figure(figsize=(8, 4))
    sns.histplot(label_counts, bins=range(0, window_size + 2), discrete=True)
    plt.title(f"Jump frames quantity distribution in each window [{base}]")
    plt.xlabel("Jump frames count")
    plt.ylabel("Window count")
    plt.grid(True)
    plot_path = os.path.join(output_dir, f"{base}_label_dist.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"  Jump frames distribution plot saved: {plot_path}")


# Check if array contains at least min_len consecutive 1s
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


# Calculate distribution of consecutive 0s and consecutive 1s segment lengths
def analyze_jump_stretch_distributions(labels, dest_path, video_name):
    """Analyze Jump Stretch Distributions

    Analyzes the distribution of consecutive 0s and 1s in jump labels.

    Returns:
        None (saves plot and logs statistics)
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
    logger.info(f"[{video_name}] Consecutive 0s segment length distribution (frames):")
    for val, cnt in Counter(zero_stretches).most_common(10):
        logger.info(f"  {val} frames: {cnt} times")
    logger.info(f"[{video_name}] Consecutive 1s segment length distribution (frames):")
    for val, cnt in Counter(one_stretches).most_common(10):
        logger.info(f"  {val} frames: {cnt} times")

    # Draw plot
    plt.figure(figsize=(10, 5))
    if zero_stretches:
        sns.histplot(zero_stretches, bins=range(0, max(zero_stretches) + 2), color='blue',
                     label='Consecutive 0s length', kde=False)
    if one_stretches:
        sns.histplot(one_stretches, bins=range(0, max(one_stretches) + 2), color='orange',
                     label='Consecutive 1s length', kde=False)
    plt.yscale("log")
    plt.xlabel("Segment length (frames)")
    plt.ylabel("Frequency (count)")
    plt.title(f"Consecutive 0s and 1s segment length distribution [{video_name}]")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plot_path = os.path.join(dest_path, f"{video_name}_jump_stretch_dist.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"  Jump frames segment length distribution plot saved: {plot_path}")


def extract_features(video_path, window_size, logger):
    """Extract Features

    Extracts features from video frames using the feature pipeline.

    Args:
        video_path: Path to the video file
        window_size: Window size for feature extraction
        logger: Logger instance for logging

    Returns:
        DataFrame containing extracted features for each frame
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return pd.DataFrame()

    # Try to get total frame count for tqdm progress bar; if not available set as None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    pipe = FeaturePipeline(cap, window_size)

    records = []
    frame_idx = 0

    # Use tqdm to wrap the loop, iterate according to actual frame count limit
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
    """Merge Labels

    Merges label information with feature data by adding a 'label' column
    based on frame ranges defined in the labels CSV file.

    Args:
        df_feat: DataFrame containing extracted features
        labels_path: Path to CSV file containing label ranges

    Returns:
        DataFrame with added 'label' column
    """
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
    # -------- Calculate statistics for each video and perform balanced data splitting ---------
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
        logger.warning("Less than 3 videos available, assigning all to train; val/test will be empty.")
        train_vids = {v['path'] for v in video_stats}
        val_vids, test_vids = set(), set()

    rng = random.Random(args.seed)
    rng.shuffle(video_stats)  # Shuffle order then sort by positive examples descending
    video_stats.sort(key=lambda x: x['pos'], reverse=True)

    target_ratio = np.array([1 - args.val_ratio - args.test_ratio, args.val_ratio, args.test_ratio], dtype=float)
    target_ratio /= target_ratio.sum()  # Normalize to sum to 1
    tot_pos = sum(v['pos'] for v in video_stats)
    deficits = target_ratio * tot_pos  # Initial deficit of positive examples needed
    splits = {"train": list(),
              "val": list(),
              "test": list()
              }  # train, val, test

    for v in video_stats:
        idx = int(np.argmax(deficits))
        key = list(splits.keys())[idx]
        splits[key].append(v)
        deficits[idx] -= v['pos']

    # ---- Calculate and log statistics ----
    def sum_pos(vs):
        return sum([v['pos'] for v in vs])

    logger.info(f"Train videos: {len(splits['train'])} (pos={sum_pos(splits['train'])}) | "
                f"Val: {len(splits['val'])} (pos={sum_pos(splits['val'])}) | "
                f"Test: {len(splits['test'])} (pos={sum_pos(splits['test'])})")

    # ---------- If preview mode, output detailed information and exit ----------
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
        logger.info("Preview complete (--preview_split). Not performing feature extraction/file writing.")

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

    # -------- Calculate statistics for each video and perform balanced data splitting ---------
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
                # Step 1: Feature extraction
                df_feat = extract_features(video_file, args.window_size, logger)
                # Step 2: Merge labels to generate frame-level labeled data
                df_labeled = merge_labels(df_feat, labels_path)
                # --- Data integrity checks ---
                assert len(df_labeled) == len(df_feat), "Label merge length mismatch"
                assert df_labeled['frame'].is_monotonic_increasing, "Frame index not monotonic"
                # build_labeled_dataset
                X, y = build_labeled_dataset(df_labeled, dest_file)
                # Analyze jump frame intervals and jump segment length distribution
                analyze_jump_stretch_distributions(y, dest_path, video_name)
            else:
                npz_dic = np.load(dest_file)
                X, y = npz_dic["X"], npz_dic["y"]

            # Step 3: Window-level data (multiple window sizes)
            window_sizes = [4, 5, 6, 8, 12, 16, 24, 32]

            for win_size in window_sizes:
                X_win, y_win = building_win_dataset(X, y, win_size, args.stride)
                # ---------- save .npz ----------
                dest_path = f"{output_dir}/size{win_size}/{split_dest_set}"
                os.makedirs(dest_path, exist_ok=True)
                dest_file = f"{dest_path}/{video_name}.npz"
                np.savez_compressed(dest_file, X=X_win, y=y_win, pos_ratio=float(y_win.mean()))
                logger.info(f'  Saved window-level npz: {dest_file}')
                print(f"Window-level data shape (size={win_size}): X={X_win.shape}, y={y_win.shape}")

                # meta.json (only create on first time)
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
                            f" label distribution: negative={cnt[0]}, positive={cnt[1]}, " +
                            f"positive ratio={(cnt[1] / (cnt[0] + cnt[1]) * 100):.2f}%")
                analyze_window_label_distribution(y_win, win_size,
                                                  os.path.join(output_dir, f"size{win_size}"),
                                                  f"{video_name}_size{win_size}")


def get_command_line_params():
    parser = argparse.ArgumentParser(description='Generate frame-level and window-level labeled training data')
    parser.add_argument('--videos_dir', default='data/raw_videos_3',
                        help='Input video directory, supports *.avi, *.mp4')
    parser.add_argument('--labels_dir', default='data/raw_videos_3', help='Labels directory, contains *_labels.csv')
    parser.add_argument('--output_dir', default='data/dataset', help='Output directory to save dataset')
    parser.add_argument('--window_size', default=8, type=int, help='Window size, =1 for frame-level only')
    parser.add_argument('--stride', default=1, type=int, help='Sliding window stride')
    # New stabilizer params
    parser.add_argument('--stabilizer_max_corners', default=VideoStabilizer.max_corners, type=int,
                        help='VideoStabilizer max corners')
    parser.add_argument('--stabilizer_quality_level', default=VideoStabilizer.quality_level, type=float,
                        help='VideoStabilizer quality level')
    parser.add_argument('--stabilizer_min_distance', default=VideoStabilizer.min_distance, type=int,
                        help='VideoStabilizer min distance')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--split_yaml', default=None,
                        help='Predefined split file (yaml: train/val/test lists), if provided overrides random split')
    parser.add_argument('--preview_split', default=True, action='store_true',
                        help='Only preview train/val/test split with positive example counts, then exit (no feature extraction)')
    args = parser.parse_args()

    suffix = mode_to_str(get_feature_mode())
    output_dir = f"{args.output_dir}_{len(SELECTED_LM)}_{suffix}"
    os.makedirs(output_dir, exist_ok=True)
    return args, output_dir


if __name__ == '__main__':
    main()
