#!/usr/bin/env python3
"""
Verify_Labels data processing module

Handles data processing and feature extraction for jump rope analysis.
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
    # loadannotation intervals
    segments = load_labels(labels_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("unable to open video")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        # check if current frame is in any interval
        in_rising = any(start <= idx <= end for start, end in segments)

        # if rising phase, highlight
        if in_rising:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                          (0, 0, 255), -1)
            alpha = 0.2  # transparency strength
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.putText(frame, 'RISING', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        # overlayaddframe号andtimespace戳
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
    group.add_argument('--video', help='singleunitsvideo文item路径')
    group.add_argument('--dir', help='package containsvideoandpair应_labels.csv文itemof目录')
    p.add_argument('--labels', help='singleunitslabel CSV 文item，onlywith --video one起use')
    args = p.parse_args()

    # if resultpointdefinedirectory，rule批quantityvalidation
    if args.dir:
        video_files = glob.glob(os.path.join(args.dir, '*.avi')) + glob.glob(os.path.join(args.dir, '*.mp4'))
        for video_path in sorted(video_files):
            base = os.path.splitext(os.path.basename(video_path))[0]
            labels_path = os.path.join(args.dir, f'{base}_labels.csv')
            if not os.path.exists(labels_path):
                print(f"警告：not找tolabel文item {labels_path}, 跳pass {video_path}")
                continue
            print(f"Verifying {base} ...")
            verify(video_path, labels_path)
        return

    # singleunitsfilemodel式
    if args.video:
        if not args.labels:
            p.error('--labels mustneedwith --video one起use')
        verify(args.video, args.labels)


if __name__ == '__main__':
    main()
