"""
python pose_sequence_dataset_builder.py \
  --input_dir ./raw_videos \
  --output_dir ./dataset \
  --window_size 64
"""

import os
import time
import argparse
import glob

import cv2
import numpy as np
import mediapipe as mp


class SequenceCollector:
    def __init__(self, output_dir, window_size, fps,
                 warmup_seconds=2.0, regions=None):
        self.window_size = window_size
        self.output_dir = output_dir
        self.fps = fps
        # 预热帧数：启动收集后跳过的初始帧数
        self.warmup_frames = int(warmup_seconds * fps)
        self.regions = regions or []
        self.collecting = False
        self.buffer = []
        self.label = None

    def start(self, label):
        self.collecting = True
        self.label = label
        self.buffer = []
        # 预热计数
        self.frames_skipped = 0
        self.start_time = time.time()
        print(f">>> START collecting [{label}]")

    def stop(self):
        self.collecting = False
        if len(self.buffer) == self.window_size:
            arr = np.stack(self.buffer, axis=0)
            filename = f"{self.label}_{int(time.time())}.npy"
            filepath = os.path.join(self.output_dir, filename)
            np.save(filepath, arr)
            print(f"Saved sequence to {filepath}")
        else:
            print(f"Sequence too short, discarded (length={len(self.buffer)})")

    def run(self):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            vec = None
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                arr = np.zeros((len(mp_pose.PoseLandmark) * 2,), dtype=np.float32)
                for i, lm_pt in enumerate(lm):
                    arr[2 * i] = lm_pt.x
                    arr[2 * i + 1] = lm_pt.y
                vec = arr

            # 2) 收集逻辑：跳过预热帧，再开始存数据
            if self.collecting and vec is not None:
                if self.frames_skipped < self.warmup_frames:
                    self.frames_skipped += 1
                else:
                    self.buffer.append(vec)
                # 缓冲满后保存并停止
                if len(self.buffer) >= self.window_size:
                    self.stop()

            cv2.imshow("Live Pose Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                label = input("Enter label for this sequence: ")
                self.start(label)

        cap.release()
        cv2.destroyAllWindows()


def build_sequences_from_video(path, window_size):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(path)
    feats = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        vec = None
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            arr = np.zeros((len(mp_pose.PoseLandmark) * 2,), dtype=np.float32)
            for i, lm_pt in enumerate(lm):
                arr[2 * i] = lm_pt.x
                arr[2 * i + 1] = lm_pt.y
            vec = arr
        feats.append(vec)

    cap.release()

    # feats = np.array(feats)  # (T, D), 包含 None
    # T, D = feats.shape
    T = len(feats)
    D = feats[0].shape[0] if feats and feats[0] is not None else 0

    W = window_size
    seqs = []
    for start in range(T - W + 1):
        window = feats[start:start + W]
        # 如果窗口内任一帧检测失败（None），跳过
        # if np.any(window == None):
        #     continue
        if any(frame_vec is None for frame_vec in window):
            continue
        seqs.append(np.stack(window, axis=0))
    return seqs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./videos")
    parser.add_argument("--output_dir", default="./dataset")
    parser.add_argument("--window_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_paths = []
    for ext in ("*.mp4", "*.avi"):
        video_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))

    seqs = []
    lbls = []
    for path in video_paths:
        fname = os.path.basename(path)
        label = fname.rsplit("_", 1)[0]
        print(f"Processing {path} with label {label}")
        sequences = build_sequences_from_video(path, args.window_size)
        seqs.extend(sequences)
        lbls.extend([label] * len(sequences))

    seqs = np.stack(seqs, axis=0)
    lbls = np.array(lbls)

    print(f"Built dataset with {len(seqs)} sequences")

    # Save sequences and labels
    np.save(os.path.join(args.output_dir, "sequences.npy"), seqs)
    np.save(os.path.join(args.output_dir, "labels.npy"), lbls)
