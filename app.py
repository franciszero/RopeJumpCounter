"""
app.py

主控脚本：基于姿态估计和深度模型实时检测并计数跳绳动作。

功能:
  1. 打开摄像头，实时读取视频帧。
  2. 使用 MediaPipe Pose 提取人体关键点，并进行背景抖动补偿。
  3. 构建特征（关键点高度、趋势滤波等），送入深度学习模型进行跳跃检测。
  4. 实时统计上升沿跳跃次数，并写入 CSV 文件。
  5. 可通过命令行参数定制输出目录、倒计时、模型路径等。

使用方法:
    python app.py --out <输出目录> [--countdown N] [--model 模型文件路径]
    python app.py --out output_dir --countdown 3
    python app.py --out output_dir --countdown 3 --model models/lstm_jump_classifier.h5

参数说明:
  --out       摄像头数据及结果保存路径（默认: record_output）
  --countdown 录制前倒计时秒数（默认: 3）
  --model     跳跃检测模型文件，支持 .keras 或 .h5 格式（默认: models/best_crnn.keras）
"""

import cv2
import os
import csv
import time
import argparse
import sys
from utils.vision import PoseEstimator
from utils.flow import BackgroundTracker
from utils.filter import TrendFilter

import tensorflow as tf
import numpy as np
from collections import deque

from PoseDetection.features import PoseFrame, Differentiator, DistanceCalculator, AngleCalculator


def record_session(output_dir, regions=None, countdown=3, model_path='PoseDetection/models/best_crnn.keras'):
    # 创建输出目录
    regions = regions or ["head", "torso"]
    os.makedirs(output_dir, exist_ok=True)

    # 初始化视频捕获与模块
    cap = cv2.VideoCapture(0)
    pose = PoseEstimator()
    bg = BackgroundTracker()
    filters = {r: TrendFilter() for r in regions}
    prev_heights = {r: None for r in regions}

    # 加载跳绳动作识别模型，并决定使用窗口模式还是单帧模式
    if not model_path.endswith(('.keras', '.h5')):
        model_path += '.keras'
    model = tf.keras.models.load_model(model_path)

    # 根据 model.input_shape 决定使用窗口模式还是单帧模式
    input_shape = model.input_shape  # e.g. (None, W, F) or (None, F)
    if len(input_shape) == 3 and input_shape[1] is not None:
        window_size = int(input_shape[1])
        use_window = window_size > 1
    else:
        window_size = 1
        use_window = False

    # === 特征提取初始化 ===
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps
    diff = Differentiator(dt)
    distance_pairs = [(24, 26), (26, 28), (11, 13), (13, 15)]
    angle_triplets = [(24, 26, 28), (11, 13, 15), (23, 11, 13)]
    dist_calc = DistanceCalculator(distance_pairs)
    ang_calc = AngleCalculator(angle_triplets)
    feature_buffer = deque(maxlen=window_size)
    prev_pred = 0

    # 倒计时，提示用户准备
    for i in range(countdown, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(
            frame,
            f"Starting in {i}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3,
        )
        cv2.imshow("Recorder", frame)
        cv2.waitKey(1000)

    # 构建CSV文件及写入表头
    csv_path = os.path.join(output_dir, "data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["frame", "timestamp"]
        header += [f"{r}_height" for r in regions]
        header += [f"{r}_fluct" for r in regions]
        header += ["jump_count"]
        writer.writerow(header)

        frame_idx = 0
        prev_timestamp = None
        jump_count = 0  # total jumps so far
        # 主循环：读取视频帧，估计姿态，补偿背景抖动，构建特征，模型推理，跳跃计数，写入CSV，显示窗口
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            timestamp = time.time()

            # Compute FPS
            if prev_timestamp is None:
                fps_display = 0.0
            else:
                fps_display = 1.0 / (timestamp - prev_timestamp)
            prev_timestamp = timestamp

            # 姿态估计，获取关键点高度
            lm, heights = pose.estimate(frame)
            if not heights:
                heights = {r: prev_heights[r] or 0.0 for r in regions}

            # 背景抖动补偿，计算背景位移
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bg_dy = bg.compensate(gray)

            # 构建数据行：帧号、时间戳
            row = [frame_idx, timestamp]

            # === 全量 469 维特征提取 ===
            lm, heights = pose.estimate(frame)
            # 获取帧尺寸
            height, width = frame.shape[:2]
            # 如果未检测到人体，填充零向量
            if lm is None:
                raw = [0.0] * (33 * 4)
                raw_px = [0.0] * (33 * 2)
                vel = [0.0] * (33 * 4)
                acc = [0.0] * (33 * 4)
                dists = [0.0] * len(distance_pairs)
                angs = [0.0] * len(angle_triplets)
            else:
                pf = PoseFrame(frame_idx, timestamp, lm.landmark, frame_size=(height, width))
                raw = pf.raw            # 132 dims
                raw_px = pf.raw_px      # 66 dims
                vel, acc = diff.compute(raw)
                dists = dist_calc.compute(lm.landmark)
                angs = ang_calc.compute(lm.landmark)

            # 拼接成特征向量，匹配模型期望
            feat = raw + raw_px + vel + acc + dists + angs  # 132+66+132+132+4+3 = 469 dims

            # 模型推理与跳跃计数
            if use_window:
                feature_buffer.append(feat)
                if len(feature_buffer) == window_size:
                    inp = np.stack(feature_buffer, axis=0)[np.newaxis, ...]
                    pred = model.predict(inp, verbose=0)[0, 0]
                    label = 1 if pred > 0.5 else 0
                else:
                    pred = 0.0
                    label = 0
            else:
                inp = np.array(feat, dtype=np.float32)[np.newaxis, np.newaxis, :]
                pred = model.predict(inp, verbose=0)[0, 0]
                label = 1 if pred > 0.5 else 0

            # 检测上升沿，累加跳绳计数
            jump_flag = (prev_pred == 0 and label == 1)
            row.append(1 if jump_flag else 0)
            if jump_flag:
                jump_count += 1
            prev_pred = label

            # 写入CSV文件
            writer.writerow(row)

            # Overlay debug info
            debug_texts = [
                f"FPS: {fps_display:.1f}",
                f"P(jump): {pred:.2f}",
                f"Jump Count: {jump_count}",
            ]
            y0, dy = 30, 30
            for i, txt in enumerate(debug_texts):
                y = y0 + i * dy
                cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            # 显示摄像头画面
            cv2.imshow("Recorder", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # 清理资源，关闭摄像头和窗口
    cap.release()
    cv2.destroyAllWindows()
    print(f"Recording complete. Data saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose-based data recorder")
    parser.add_argument("--out", default="record_output", help="Output directory for data")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown seconds before recording starts")
    parser.add_argument(
        '--model',
        default='PoseDetection/models/best_crnn.keras',
        help='Path to .keras or .h5 model file (default: PoseDetection/models/best_crnn.keras)'
    )
    args = parser.parse_args()
    record_session(args.out, countdown=args.countdown, model_path=args.model)
