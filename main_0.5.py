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
import sys
import time

import cv2
import base64
import numpy as np
import pandas as pd
import tensorflow as tf
import PySimpleGUIQt as sg
from PoseDetection.models.ModelParams.TCNBlock import TCNBlock

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
        We therefore **copy** the bytes so that QByteArray owns its own memory.
        """
        if length is None:
            length = len(buf)
        # Make an owned copy → safe after Python buffer is freed
        return QtCore.QByteArray(bytes(buf[:length]))


    QtCore.QByteArray.fromRawData = _from_raw_compat  # monkey‑patch
except Exception:
    # If PySide6 unavailable / already Qt5, ignore
    pass

from tqdm import tqdm

from PoseDetection.features import FeaturePipeline

import logging

from PoseDetection.models.ModelParams.ThresholdHolder import ThresholdHolder

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoPredictor:
    """封装模型 + 滑动窗口推理逻辑"""

    def __init__(self, model_path: str):
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

    def __init__(self, predictor: VideoPredictor, width, height, fps):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.predictor = predictor
        self.playing = True
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        sg.theme("DarkBlue3")
        layout = [[sg.Image(filename="", key="-IMAGE-")],
                  [sg.Text("Space:Play/Pause  ←/→:Step  Esc:Quit")]]
        self.window = sg.Window(f"Visualize – camera",
                                layout,
                                return_keyboard_events=True,
                                finalize=True)

    def _overlay(self, frame: np.ndarray, prob: float) -> np.ndarray:
        """在 frame 上绘制概率/标签"""
        if prob is not None and prob >= self.predictor.threshold:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                          (0, 0, 255), thickness=-1)
            alpha = 0.15
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.putText(frame, "RISING", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 255), 2,
                        cv2.LINE_AA)
        if prob is not None:
            cv2.putText(frame, f"p={prob:.2f}", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2,
                        cv2.LINE_AA)
        return frame

    def run(self):
        """
        Main event‑loop.
        * Space – play / pause
        * ← / → – single‑step back / forward (while paused)
        * Esc / window‑close – quit
        """
        pipe = FeaturePipeline(self.cap, self.predictor.window_size)
        frame_idx = 0
        prev_time = time.time()

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

            ok = pipe.success_process_frame(frame_idx)
            if not ok:
                break  # EOF
            frame_idx += 1

            # numeric feature vector (length = feature_dim)
            feat_vec = pd.DataFrame([pipe.fs.rec]).iloc[0][2:].values.astype(np.float32)
            prob = self.predictor.predict(feat_vec)

            frame_vis = self._overlay(pipe.fs.raw_frame, prob)

            png_bytes = cv2.imencode(".png", frame_vis)[1].tobytes()
            # Qt6: QByteArray.fromRawData now needs both buffer & length → pass a tuple
            self.window["-IMAGE-"].update(data=png_bytes)

            # ---------- pacing ----------
            elapsed = time.time() - prev_time
            wait = max(1.0 / self.fps - elapsed, 0)
            time.sleep(wait)
            prev_time = time.time()

        self.cap.release()
        self.window.close()


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", default="PoseDetection/model_files/tcn_ws24.keras", help="path to *.keras model")
    # parser.add_argument("--model", default="PoseDetection/model_files/seresnet1d_ws16.keras", help="path to *.keras model")
    # parser.add_argument("--model", default="PoseDetection/model_files/resnet1d_ws16.keras", help="path to *.keras model")
    parser.add_argument("--model", default="PoseDetection/model_files/efficientnet1d_ws4.keras", help="path to *.keras model")
    # parser.add_argument("--model", default="PoseDetection/model_files/inception_ws4.keras", help="path to *.keras model")
    # parser.add_argument("--model", default="PoseDetection/model_files/cnn_ws4.keras", help="path to *.keras model")
    parser.add_argument("--width", type=int, default=640, help="Video frame width")
    parser.add_argument("--height", type=int, default=480, help="Video frame height")
    parser.add_argument("--fps", type=int, default=30, help="Capture frames per second")
    args = parser.parse_args()

    predictor = VideoPredictor(args.model)
    gui = PlayerGUI(predictor, args.width, args.height, args.fps)
    gui.run()


if __name__ == "__main__":
    main()
