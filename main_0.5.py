# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import time
from collections import deque
import cv2
from datetime import datetime
from PoseDetection.data_builder_utils.feature_mode import mode_to_str, get_feature_mode
from capture.pyav_capture import PyAVCapture
import numpy as np
import pandas as pd
from PoseDetection.features import FeaturePipeline
from utils.FrameSample import SELECTED_LM
from utils.Perf import PerfStats
from PoseDetection.models.ModelParams.ThresholdHolder import ThresholdHolder
from PoseDetection.models.ModelParams.TCNBlock import TCNBlock

# 强制使用 MPS/GPU
import tensorflow as tf
from tensorflow.keras import mixed_precision, models

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus, 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# interpreter = tf.lite.Interpreter(
#     model_path="model.tflite",
#     experimental_delegates=[tf.lite.experimental.load_delegate('libtensorflowlite_gpu_delegate.dylib')]
# )
# interpreter.allocate_tensors()
# print("=== physical devices ===")
# for d in tf.config.list_physical_devices():
#     print(d)
# print("=== GPUs ===", tf.config.list_physical_devices('GPU'))
# tf.debugging.set_log_device_placement(True)
# # print("Built with MPS support:", tf.test.is_built_with_mps())
# print("MPS GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoPredictor:
    """封装模型 + 滑动窗口推理逻辑"""

    def __init__(self, model_path: str):
        self.model = models.load_model(model_path, compile=False)
        # (batch, timestamps, feature_dim)
        _, self.window_size, feat_dim = self.model.input_shape
        print("window_size =", self.window_size)  # 4
        print("feature_dim =", feat_dim)  # 403 之类
        self.threshold = 0.5  # float(self.model.get_layer("f1_threshold").t.numpy())

        # 用 deque 维护最近 window_size 帧特征
        self.buffer = deque(maxlen=self.window_size)
        # 在首次喂满窗口前，无推理结果
        self._warmup = self.window_size

    def predict(self, feature_dim: np.ndarray) -> float:
        """
        传入 BGR frame → 更新窗口 → 若已满返回正例概率，否则 None
        """
        self.buffer.append(feature_dim)

        if len(self.buffer) < self.window_size:
            return 0.0  # still warming‑up

        window = np.stack(self.buffer, axis=0)  # (win, feat_dim)
        self.model.run_eagerly = True
        prob = float(self.model(np.expand_dims(window, axis=0), training=False)[0])
        return prob


class PlayerGUI:
    """
    简易播放器：空格暂停/继续；← → 单帧步进；Esc 退出
    """

    def __init__(self, predictor: VideoPredictor, width, height, fps, save_path: str | None = None):
        self.cap = PyAVCapture(device_index=0, width=width, height=height, fps=fps)
        self.zoom_height = 920  # 原始 cv2 图像，高度变成 zoom_height，放大一点

        self.stats = PerfStats(window_size=10)

        self.predictor = predictor
        self.fps = fps

        # ---- simple FPS meter ----
        self.proc_times = deque(maxlen=30)  # ms of recent frames

        if save_path:
            time_str = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
            dest_file = f"{save_path}/jump_{time_str}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(dest_file, fourcc, fps, (int(width), int(height)))
            if not self.writer.isOpened():
                logger.error(f"VideoWriter open failed: {dest_file}, fourcc=XVID")
            else:
                logger.info(f"VideoWriter OK → {dest_file}")
            print(f"[DEBUG] 保存视频目标地址：{dest_file}")
        else:
            self.writer = None

    def _overlay(self, frame: np.ndarray, jump_cnt: int, prob: float, is_on_rising: bool, t0) -> np.ndarray:
        """在 frame 上绘制概率/标签"""
        if jump_cnt is not None:
            cv2.putText(frame, f"JUMPS: {jump_cnt}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 255), 2,
                        cv2.LINE_AA)
        if prob is not None and is_on_rising:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), thickness=-1)
            alpha = 0.15
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.putText(frame, "RISING", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 255), 2,
                        cv2.LINE_AA)
        if prob is not None:
            cv2.putText(frame, f"p={prob:.2f}", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2,
                        cv2.LINE_AA)
        if self.stats.proc_fps is not None and self.stats.last_latency_ms is not None:
            txt = f"{self.stats.proc_fps:4.1f} FPS | {self.stats.last_latency_ms:3.0f} ms"
            cv2.putText(frame, txt,
                        (frame.shape[1] - 260, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
                        cv2.LINE_AA)
        return frame

    def run(self):
        pipe = FeaturePipeline(self.cap, self.predictor.window_size)
        frame_idx = 0
        jump_cnt = 0
        jump_cnt_binary_mark = 0  # start with 000 然后

        while True:
            arr_ts = list()

            arr_ts.append(time.time())
            ret, frame, _ = self.cap.read()  # Original BGR frame (ignore latency)
            if not ret:
                continue

            arr_ts.append(time.time())
            # 1) 拉帧 + 特征抽取
            pipe.process_frame(frame, frame_idx)
            frame_idx += 1

            arr_ts.append(time.time())
            # 2) 模型推理
            feat_vec = pd.DataFrame([pipe.fs.rec]).iloc[0][2:].values.astype(np.float32)
            prob = self.predictor.predict(feat_vec)

            arr_ts.append(time.time())
            # 3) 叠加性能统计 & 跳绳计数/高亮等
            jump_cnt_binary_mark, is_on_rising, jump_cnt = self.jump_event_detect(jump_cnt, jump_cnt_binary_mark, prob)
            frame_vis = self._overlay(pipe.fs.raw_frame.copy(), jump_cnt, prob, is_on_rising, arr_ts[0])
            # frame_vis = imutils.resize(frame_vis, height=self.zoom_height)

            # 4) 显示 & 可选录制
            cv2.imshow("JumpRope RealTime", frame_vis)
            if self.writer:
                self.writer.write(frame_vis)
                print("[DEBUG] Write frame")

            arr_ts.append(time.time())
            # 5) 更新性能统计
            self.stats.update("[Main Process]: ", arr_ts, 0)

            # 6) 唯一键：按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        if self.writer is not None:
            self.writer.release()

    def jump_event_detect(self, jump_cnt, jump_cnt_binary_mark, prob):
        y_pred = int((prob > self.predictor.threshold))
        mark1 = (jump_cnt_binary_mark << 1) & 0b111  # 保留最后3位
        jump_cnt_binary_mark = (mark1 | y_pred) & 0b111
        if jump_cnt_binary_mark in [3, 7]:  # 3:011 -> 7:111
            is_on_rising = True
            if jump_cnt_binary_mark == 3:  # 只有事件 3 检测为起跳事件，进行跳绳计数
                jump_cnt += 1  # 判断为一次起跳，由 0 变为 1 表明模型判断起跳，2个以上连续 1 表明模型认为目标一直在上升
        else:  # 0:000, 1:001, 2:010, 4:100, 5:101, 不是稳定的检测结果, 6:110 表明跳绳刚结束
            is_on_rising = False
        print(
            f"[DEBUG][{jump_cnt:04d}][{prob * 100:.2f}%][{self.predictor.threshold * 100:.2f}%] jump mask: {mark1:03b}+{y_pred:03b}={jump_cnt_binary_mark:03b}")
        return jump_cnt_binary_mark, is_on_rising, jump_cnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_cnn8_ws4_withT.keras")  # 17ms 25.6FPS
    # parser.add_argument("--model", default="best_cnn_hybrid_ws4_withT.keras")  # 117ms
    # parser.add_argument("--model", default="best_crnn_ws12_withT.keras")  # 68ms 14.2FPS
    # parser.add_argument("--model", default="best_efficientnet1d_ws4_withT.keras")  # 39ms 25.6FPS
    # parser.add_argument("--model", default="best_inception_ws4_withT.keras")  # 50ms 19.7FPS
    # parser.add_argument("--model", default="best_lstm_attention_ws16_withT.keras")  # 124ms 8FPS
    # parser.add_argument("--model", default="best_resnet1d_ws16_withT.keras")  # 44ms 22.7FPS
    # parser.add_argument("--model", default="best_resnet1d_tcn_ws16_withT.keras")  # 58ms 17FPS
    # parser.add_argument("--model", default="best_seresnet1d_ws16_withT.keras")  # 49ms 19.5FPS
    # parser.add_argument("--model", default="best_tcn_ws24_withT.keras")  # 40ms 24FPS
    # parser.add_argument("--model", default="best_tcn_se_ws24_withT.keras")  # 60ms 16FPS
    # parser.add_argument("--model", default="best_tftlite_ws16_withT.keras")  # 127ms 8FPS
    # parser.add_argument("--model", default="best_transformerlite_ws16_withT.keras")  # 45ms 22.3FPS
    # parser.add_argument("--model", default="best_wavenet_ws8_withT.keras")  # 57ms 17.7FPS

    parser.add_argument("--width", type=int, default=640, help="Video frame width")
    parser.add_argument("--height", type=int, default=480, help="Video frame height")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save_video", default="data/raw_videos_3", help="Video save path")
    args = parser.parse_args()

    predictor = VideoPredictor(f"model_files/models_{len(SELECTED_LM)}_{mode_to_str(get_feature_mode())}/" + args.model)
    gui = PlayerGUI(predictor, args.width, args.height, args.fps, save_path=args.save_video)
    gui.run()


if __name__ == "__main__":
    main()
