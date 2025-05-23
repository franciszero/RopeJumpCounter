# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelVisualize.py

小工具：载入训练好的 *.keras 模型，对 *未参与训练* 的本地视频做逐帧推理，
在播放窗口实时叠加 “rising” 标签，并用浅红色蒙版高亮。

依赖：
    pip install opencv-python PySimpleGUIQt tensorflow

⚠️ 注意：
    1. 下面示例的 `_extract_feature_vec()` 仅作演示，返回空向量。
       你应当按自己的 FeaturePipeline / mediapipe 逻辑改写此函数，确保
         `feat.shape == (feature_dim,)`
    2. 如果模型是 Conv1D / TCN 等时序网络，需要指定正确的 `window_size`
       与特征维 `feature_dim`。

Usage
------
python ModelVisualize.py \
    --model model_files/best_cnn_ws4.keras \
    --video ../raw_videos/new_jump.mp4 \
    --window_size 4 \
    --threshold 0.5
"""
import argparse
import collections
import pathlib
import time

import cv2

import imutils
import numpy as np
import pandas as pd
import tensorflow as tf
import PySimpleGUIQt as sg

from PoseDetection.data_builder_utils.feature_mode import get_feature_mode_all, get_feature_mode, mode_to_str
from PoseDetection.features import FeaturePipeline
from PoseDetection.models.ModelParams.TCNBlock import TCNBlock

import logging

from PoseDetection.models.ModelParams.ThresholdHolder import ThresholdHolder
from utils.FrameSample import SELECTED_LM
from utils.Perf import PerfStats

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Qt6 compatibility patch: allow fromRawData(buf) ---
try:
    from PySide6 import QtCore  # PySimpleGUIQt uses PySide6 on Qt6

    _orig_from_raw = QtCore.QByteArray.fromRawData


    @staticmethod
    def _from_raw_compat(buf, length=None):
        """
        Qt6’s QByteArray.fromRawData keeps a *view* of the Python buffer.
        When the Python `bytes` object is GC‑ed the image data becomes invalid,
        leading to “wrong (missing signature)” PNG errors a few frames later.

        PySimpleGUIQt (Qt6) may pass either:
          • raw bytes
          • a tuple ``(bytes, len)`` we supplied in Element.update
        Therefore we accept both.
        """
        # --- handle tuple wrapper (buf, len) ---
        if isinstance(buf, tuple):
            buf, length = buf  # unpack
        # fall back to automatic len if not provided
        if length is None:
            length = len(buf)
        # Make an owned copy → safe after Python buffer is freed
        # buf should be bytes or bytearray, slice returns bytes
        return QtCore.QByteArray(buf[:length])


    QtCore.QByteArray.fromRawData = _from_raw_compat  # monkey‑patch
except Exception:
    # If PySide6 unavailable / already Qt5, ignore
    pass


class VideoPredictor:
    """封装模型 + 滑动窗口推理逻辑"""

    def __init__(self,
                 model_path: str,
                 threshold: float = 0.5):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        # (batch, timesteps, feature_dim)
        _, self.window_size, feat_dim = self.model.input_shape
        print("window_size =", self.window_size)  # 4
        print("feature_dim =", feat_dim)  # 403 之类
        self.threshold = float(self.model.get_layer("f1_threshold").t.numpy())

        # 用 deque 维护最近 window_size 帧特征
        self.buffer = collections.deque(maxlen=self.window_size)
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
        prob = float(self.model(np.expand_dims(window, axis=0), training=False)[0])
        return prob


class PlayerGUI:
    """
    简易播放器：空格暂停/继续；← → 单帧步进；Esc 退出
    """

    def __init__(self, video_path: str, predictor: VideoPredictor):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")

        self.predictor = predictor
        self.playing = True
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.zoom_height = 920  # 原始 cv2 图像，高度变成 zoom_height，放大一点

        self.stats = PerfStats(window_size=10)

        sg.theme("DarkBlue3")
        layout = [[sg.Image(filename="", key="-IMAGE-")],
                  [sg.Text("Space:Play/Pause  ←/→:Step  Esc:Quit")]]
        self.window = sg.Window(f"Visualize – {pathlib.Path(video_path).name}",
                                layout,
                                return_keyboard_events=True,
                                finalize=True)

    def _overlay(self, frame: np.ndarray, jump_cnt: int, prob: float, is_on_rising: bool, t0) -> np.ndarray:
        """calculate FPS"""
        # proc_ms = (time.time() - t0) * 1000.0
        # self.proc_times.append(proc_ms)
        # fps_disp = 1000.0 / (sum(self.proc_times) / len(self.proc_times))

        """在 frame 上绘制概率/标签"""
        if jump_cnt is not None:
            cv2.putText(frame, f"JUMPS: {jump_cnt}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 255), 2,
                        cv2.LINE_AA)

        """在 frame 上绘制概率/标签"""
        if prob is not None and is_on_rising:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                          (0, 0, 255), thickness=-1)
            alpha = 0.15
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.putText(frame, "RISING", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 20, 255), 2,
                        cv2.LINE_AA)
        if prob is not None:
            cv2.putText(frame, f"p={prob:.2f}", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2,
                        cv2.LINE_AA)

        # draw runtime metrics (bottom‑right)
        info = self.stats.info_text(self.fps)
        (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (frame.shape[1] - tw - 20, frame.shape[0] - th - 30),
                      (frame.shape[1] - 10, frame.shape[0] - 10), (0, 0, 0), thickness=-1)
        cv2.putText(frame, info,
                    (frame.shape[1] - tw - 15, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        return frame

    # ---------- stats ----------
    # def _update_stats(self, elapsed: float):
    #     "Keep a sliding‑window FPS / latency estimate"
    #     self.proc_times.append(elapsed)
    #     if self.proc_times:
    #         self.last_latency_ms = elapsed * 1000.0
    #         self.proc_fps = sum(self.proc_times) / len(self.proc_times)

    def run(self, mode):
        """
        Main event‑loop.
        * Space – play / pause
        * ← / → – single‑step back / forward (while paused)
        * Esc / window‑close – quit
        """
        pipe = FeaturePipeline(self.cap, self.predictor.window_size)
        frame_idx = 0
        prev_time = time.time()
        jump_cnt = 0
        jump_cnt_binary_mark = 0  # start with 000 然后

        # we do _one_ Window.read() per iteration to keep Qt alive
        while True:
            timeout = 0 if self.playing else 100  # ms
            event, _ = self.window.read(timeout=timeout)

            # ---------- handle UI events ----------
            if event in (sg.WIN_CLOSED, "Escape:27"):
                break
            if event in ("space:32",):
                self.playing = not self.playing
            if (event in ("Left:37", "Right:39")) and not self.playing:
                step = -1 if "Left" in event else 1
                new_pos = max(0, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) + step)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                frame_idx = new_pos
                self.predictor.buffer.clear()  # window reset
                continue  # wait for next loop

            # ---------- decode & infer next frame ----------
            if not self.playing:
                continue

            arr_ts = list()

            arr_ts.append(time.time())
            ret, frame = self.cap.read()  # Original BGR frame (ignore latency)
            if not ret:
                break

            arr_ts.append(time.time())
            pipe.process_frame(frame, frame_idx, mode=mode)
            frame_idx += 1

            arr_ts.append(time.time())
            # numeric feature vector (length = feature_dim)
            feat_vec = pd.DataFrame([pipe.fs.rec]).iloc[0][2:].values.astype(np.float32)
            prob = self.predictor.predict(feat_vec)
            y_pred = int((prob > self.predictor.threshold))

            arr_ts.append(time.time())
            jump_cnt_binary_mark = ((jump_cnt_binary_mark << 1) | y_pred) & 0b111  # 保留最后3位
            mark1 = (jump_cnt_binary_mark << 1) & 0b111
            jump_cnt_binary_mark = (mark1 | y_pred) & 0b111
            # print(f"[DEBUG] jump mask: {mark1:03b}+{y_pred:03b}={jump_cnt_binary_mark:03b}")
            if jump_cnt_binary_mark in [3, 7]:  # 3:011 -> 7:111
                is_on_rising = True
                if jump_cnt_binary_mark == 3:  # 只有事件 3 检测为起跳事件，进行跳绳计数
                    jump_cnt += 1  # 判断为一次起跳，由 0 变为 1 表明模型判断起跳，2个以上连续 1 表明模型认为目标一直在上升
            else:  # 0:000, 1:001, 2:010, 4:100, 5:101, 不是稳定的检测结果, 6:110 表明跳绳刚结束
                is_on_rising = False

            frame_vis = self._overlay(pipe.fs.raw_frame.copy(), jump_cnt, prob, is_on_rising, arr_ts[0])
            # resize to fill the window height, maintain aspect ratio
            frame_vis = imutils.resize(frame_vis, height=self.zoom_height)

            png_bytes = cv2.imencode(".png", frame_vis)[1].tobytes()
            # Qt6: QByteArray.fromRawData now needs both buffer & length → pass a tuple
            self.window["-IMAGE-"].update(data=(png_bytes, len(png_bytes)))

            # update stats
            arr_ts.append(time.time())
            self.stats.update("[Main Process]: ", arr_ts)
            # ---------- pacing ----------
            elapsed = time.time() - prev_time
            wait = max(1.0 / self.fps - elapsed, 0)
            time.sleep(wait)
            prev_time = time.time()

        self.cap.release()
        self.window.close()


def main():
    parser = argparse.ArgumentParser()
    # ========= models ==========
    # parser.add_argument("--model", default="best_cnn8_ws4_withT.keras")
    parser.add_argument("--model", default="best_tcn_ws24_withT.keras")

    # ========= videos ==========
    # parser.add_argument("--video", default="raw_videos_3/jump_2025.05.14.08.34.44.avi")
    parser.add_argument("--video", default="../data/raw_videos_3/jump_2025.05.22.08.33.08__100.avi")
    # parser.add_argument("--video", default="raw_videos_3/jump_2025.05.15.08.37.31.avi")

    # ===========================
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model_path = f"../model_files/models_{len(SELECTED_LM)}_{mode_to_str(get_feature_mode())}/{args.model}"
    predictor = VideoPredictor(model_path, args.threshold)
    gui = PlayerGUI(args.video, predictor)

    gui.run(get_feature_mode())


if __name__ == "__main__":
    main()
