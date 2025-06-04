#!/usr/bin/env python3
"""
verify_labels.py

可视化验证标注效果：在视频回放时高亮所有“跳跃上升段”。
Usage:
    1.	批量验证目录下所有视频
    python verify_labels.py --dir raw_videos
	2.	单视频单标签验证
    python verify_labels.py --video ../raw_videos/jump_001.avi --labels ../raw_videos/jump_001_labels.csv
按 q 退出。
"""

import cv2
import csv
import argparse
import glob
import os


def load_labels(csv_path):
    segments = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = int(row['start_frame'])
            end = int(row['end_frame'])
            segments.append((start, end))
    return segments


def verify(video_path, labels_path):
    # 加载标注区间
    segments = load_labels(labels_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 判断当前帧是否在任一区间内
        in_rising = any(start <= idx <= end for start, end in segments)

        # 如果是上升段，高亮
        if in_rising:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                          (0, 0, 255), -1)
            alpha = 0.2  # 半透明强度
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.putText(frame, 'RISING', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        # 叠加帧号和时间戳
        ts = idx / fps
        cv2.putText(frame, f"Frame {idx}/{total - 1}  {ts:.2f}s",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Verify Labels', frame)
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):
            break

        idx += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', help='单个视频文件路径')
    group.add_argument('--dir', help='包含视频和对应_labels.csv文件的目录')
    p.add_argument('--labels', help='单个标签 CSV 文件，仅与 --video 一起使用')
    args = p.parse_args()

    # 如果指定目录，则批量验证
    if args.dir:
        video_files = glob.glob(os.path.join(args.dir, '*.avi')) + glob.glob(os.path.join(args.dir, '*.mp4'))
        for video_path in sorted(video_files):
            base = os.path.splitext(os.path.basename(video_path))[0]
            labels_path = os.path.join(args.dir, f'{base}_labels.csv')
            if not os.path.exists(labels_path):
                print(f"警告：未找到标签文件 {labels_path}, 跳过 {video_path}")
                continue
            print(f"Verifying {base} ...")
            verify(video_path, labels_path)
        return

    # 单个文件模式
    if args.video:
        if not args.labels:
            p.error('--labels 必须与 --video 一起使用')
        verify(args.video, args.labels)


if __name__ == '__main__':
    main()
