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


def record_session(output_dir, regions=None, countdown=3, model_path='jump_detection_model.keras'):
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
    model = tf.keras.models.load_model(model_path)  # adjust path if necessary
    use_window = hasattr(model.input, 'shape') and model.input.shape[1] > 1
    window_size = model.input.shape[1]
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
        # 主循环：读取视频帧，估计姿态，补偿背景抖动，构建特征，模型推理，跳跃计数，写入CSV，显示窗口
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            timestamp = time.time()

            # 姿态估计，获取关键点高度
            lm, heights = pose.estimate(frame)
            if not heights:
                heights = {r: prev_heights[r] or 0.0 for r in regions}

            # 背景抖动补偿，计算背景位移
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bg_dy = bg.compensate(gray)

            # 构建数据行：帧号、时间戳、各部位高度、趋势滤波后的波动值
            row = [frame_idx, timestamp]
            fluctuations = {}
            for r in regions:
                prev = prev_heights[r]
                curr = heights.get(r, prev or 0.0)
                body_dy = 0.0 if prev is None else (curr - prev)
                prev_heights[r] = curr

                # 趋势滤波器更新，滤除噪声
                f_val = filters[r].update(body_dy - bg_dy, frame_idx)
                fluctuations[r] = f_val
                row.append(curr)
            for r in regions:
                row.append(fluctuations[r])

            # 构建输入特征，送入模型进行跳跃检测
            feat = np.array(row[2:], dtype=np.float32)  # shape (F,)
            if use_window:
                feature_buffer.append(feat)
                if len(feature_buffer) == window_size:
                    inp = np.stack(feature_buffer, axis=0)[np.newaxis, ...]  # shape (1, W, F)
                    pred = model.predict(inp, verbose=0)[0, 0]  # probability of jump
                    label = 1 if pred > 0.5 else 0
                else:
                    label = 0
            else:
                inp = feat[np.newaxis, np.newaxis, :]  # shape (1,1,F)
                pred = model.predict(inp, verbose=0)[0, 0]
                label = 1 if pred > 0.5 else 0

            # 跳跃计数：检测预测标签的上升沿
            jump_count = prev_pred == 0 and label == 1
            if jump_count:
                row.append(1)
            else:
                row.append(0)
            prev_pred = label

            # 写入CSV文件
            writer.writerow(row)

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
        default='models/best_crnn.keras',
        help='Path to .keras or .h5 model file (default: models/best_crnn.keras)'
    )
    args = parser.parse_args()
    record_session(args.out, countdown=args.countdown, model_path=args.model)
