#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_frames.py

用途：
    统计给定视频的总帧数，先通过元数据查询，再逐帧遍历以验证。

依赖：
    pip install opencv-python

用法：
    python count_frames.py --input jump_2025.05.08.09.30.00.avi
"""

import cv2
import argparse
import sys

def count_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频文件 {video_path}", file=sys.stderr)
        return

    # 方法一：从元数据直接获取（快速）
    total_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"从元数据读取的帧数（估计值）：{total_meta}")

    # 方法二：逐帧遍历（精确）
    count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1

    print(f"逐帧遍历统计的帧数：{count}")

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计视频总帧数")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入视频文件路径（如 jump_2025.05.08.09.30.00.avi）"
    )
    args = parser.parse_args()
    count_frames(args.input)