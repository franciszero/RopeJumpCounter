# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Visualization Tool

Utility tool to load trained *.keras models and process local videos not used in training
with frame-by-frame inference, overlaying rising labels in real-time with light red mask highlight.

Dependencies:
    pip install opencv-python PySimpleGUIQt tensorflow

⚠️ Notes:
    1. The example below is for demonstration only and returns empty vectors.
       You should modify this function according to your own FeaturePipeline/MediaPipe logic,
       ensuring that `feat.shape == (feature_dim,)`
    2. If the model is Conv1D/TCN or other time sequence networks, you need to correctly
       define the `window_size` and feature dimension `feature_dim`.

Usage:
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

from src.ml.data.builders.feature_mode import get_feature_mode_all, get_feature_mode, mode_to_str
from src.ml.data.features.features import FeaturePipeline
from src.ml.models.ModelParams.TCNBlock import TCNBlock

import logging
import mediapipe as mp

from src.ml.models.ModelParams.ThresholdHolder import ThresholdHolder
from src.utils.FrameSample import SELECTED_LM
from src.utils.Perf import PerfStats

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
    """Video prediction with sliding window inference

    Encapsulates model loading and sliding window inference logic for
    real-time jump detection from video frames.
    """

    def __init__(self, model_path: str, threshold: float = 0.5):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        # (batch, timesteps, feature_dim)
        _, self.window_size, feat_dim = self.model.input_shape
        print("window_size =", self.window_size)  # 4
        print("feature_dim =", feat_dim)  # 403 etc
        self.threshold = float(self.model.get_layer("f1_threshold").t.numpy())

        # Use deque to maintain recent window_size frame features
        self.buffer = collections.deque(maxlen=self.window_size)
        # Before first window is full, no inference result
        self._warmup = self.window_size

    def predict(self, feature_vector: np.ndarray) -> float:
        """Predict jump probability from feature vector

        Args:
            feature_vector: Feature vector extracted from current frame

        Returns:
            float: Jump probability (0.0 if still warming up, otherwise model prediction)
        """
        self.buffer.append(feature_vector)

        if len(self.buffer) < self.window_size:
            return 0.0  # Still warming up

        window = np.stack(self.buffer, axis=0)  # (win, feat_dim)
        prob = float(self.model(np.expand_dims(window, axis=0), training=False)[0])
        return prob


class PlayerGUI:
    """Simple video player with jump detection visualization

    Controls:
    - Space: Play/Pause
    - ← →: Single frame step (when paused)
    - Esc: Exit
    """

    def __init__(self, video_path: str, predictor: VideoPredictor, show_stick_figure: bool = True):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")

        self.predictor = predictor
        self.playing = True
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.zoom_height = 920  # Original cv2 image height scaled to zoom_height for better visibility
        self.show_stick_figure = show_stick_figure

        self.stats = PerfStats(window_size=10)

        sg.theme("DarkBlue3")
        layout = [[sg.Image(filename="", key="-IMAGE-")],
                  [sg.Text("Space:Play/Pause  ←/→:Step  Esc:Quit")]]
        self.window = sg.Window(f"Visualize – {pathlib.Path(video_path).name}",
                                layout,
                                return_keyboard_events=True,
                                finalize=True)

    def _draw_stick_figure(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Draw stick figure pose overlay on frame

        Args:
            frame: Input video frame
            landmarks: MediaPipe pose landmarks

        Returns:
            Frame with stick figure overlay drawn
        """
        if not landmarks:
            return frame

        h, w = frame.shape[:2]

        # MediaPipe pose connection definitions
        pose_connections = [
            # Head and torso
            (mp.solutions.pose.PoseLandmark.LEFT_EYE, mp.solutions.pose.PoseLandmark.RIGHT_EYE),
            (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER),
            (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP),
            (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
            (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_HIP),

            # Left arm
            (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
            (mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST),

            # Right arm
            (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
            (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST),

            # Left leg
            (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE),
            (mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.LEFT_HEEL),
            (mp.solutions.pose.PoseLandmark.LEFT_HEEL, mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX),

            # Right leg
            (mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
            (mp.solutions.pose.PoseLandmark.RIGHT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_HEEL),
            (mp.solutions.pose.PoseLandmark.RIGHT_HEEL, mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX),
        ]

        # Draw connection lines
        for connection in pose_connections:
            start_idx, end_idx = connection
            start_landmark = landmarks.landmark[start_idx.value]
            end_landmark = landmarks.landmark[end_idx.value]

            # Check landmark visibility
            if (start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5):
                start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
                end_point = (int(end_landmark.x * w), int(end_landmark.y * h))

                # Select color and thickness based on connection type
                if 'EYE' in start_idx.name or 'EYE' in end_idx.name:
                    color, thickness = (255, 255, 0), 2  # Yellow - head
                elif 'SHOULDER' in start_idx.name or 'SHOULDER' in end_idx.name:
                    color, thickness = (255, 0, 0), 3    # Red - torso
                elif 'HIP' in start_idx.name or 'HIP' in end_idx.name:
                    color, thickness = (255, 0, 0), 3    # Red - torso
                elif 'ARM' in start_idx.name or 'ELBOW' in start_idx.name or 'WRIST' in start_idx.name:
                    color, thickness = (0, 255, 255), 2  # Cyan - arms
                elif 'ARM' in end_idx.name or 'ELBOW' in end_idx.name or 'WRIST' in end_idx.name:
                    color, thickness = (0, 255, 255), 2  # Cyan - arms
                else:
                    color, thickness = (0, 255, 0), 2    # Green - legs

                # Draw connection line
                cv2.line(frame, start_point, end_point, color, thickness)

        # Draw landmarks
        for landmark_idx in SELECTED_LM:
            landmark = landmarks.landmark[landmark_idx.value]
            if landmark.visibility > 0.5:
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                # Select color and size based on landmark type
                if 'EYE' in landmark_idx.name:
                    color, radius = (255, 255, 0), 3  # Yellow - eyes
                elif 'SHOULDER' in landmark_idx.name or 'HIP' in landmark_idx.name:
                    color, radius = (255, 0, 0), 5    # Red - major joints
                elif 'ELBOW' in landmark_idx.name or 'KNEE' in landmark_idx.name:
                    color, radius = (0, 255, 255), 4  # Cyan - middle joints
                elif 'WRIST' in landmark_idx.name or 'HEEL' in landmark_idx.name or 'FOOT' in landmark_idx.name:
                    color, radius = (255, 0, 255), 4  # Magenta - end joints
                else:
                    color, radius = (0, 255, 0), 3    # Green - others

                # Draw landmarks with border effect
                cv2.circle(frame, (x, y), radius + 1, (0, 0, 0), -1)  # Black border
                cv2.circle(frame, (x, y), radius, color, -1)           # Colored fill

        return frame

    def _overlay(self, frame: np.ndarray, jump_cnt: int, prob: float, is_on_rising: bool, t0, landmarks=None) -> np.ndarray:
        """Draw overlay information on frame

        Args:
            frame: Input video frame
            jump_cnt: Current jump count
            prob: Jump probability from model
            is_on_rising: Whether currently in rising phase
            t0: Timestamp for performance calculation
            landmarks: MediaPipe pose landmarks

        Returns:
            Frame with overlay information drawn
        """

        # Draw stick figure pose
        if self.show_stick_figure and landmarks is not None:
            frame = self._draw_stick_figure(frame, landmarks)

        # Draw jump count on frame
        if jump_cnt is not None:
            cv2.putText(frame, f"JUMPS: {jump_cnt}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 255), 2,
                        cv2.LINE_AA)

        # Draw rising indicator with red overlay
        if prob is not None and is_on_rising:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                          (0, 0, 255), thickness=-1)
            alpha = 0.15
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.putText(frame, "RISING", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 20, 255), 2,
                        cv2.LINE_AA)

        # Draw probability value
        if prob is not None:
            cv2.putText(frame, f"p={prob:.2f}", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2,
                        cv2.LINE_AA)

        # Draw runtime metrics (bottom-right)
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
        jump_cnt_binary_mark = 0  # Start with 000 binary pattern

        # We do _one_ Window.read() per iteration to keep Qt alive
        while True:
            timeout = 0 if self.playing else 100  # ms
            event, _ = self.window.read(timeout=timeout)

            # ---------- Handle UI events ----------
            if event in (sg.WIN_CLOSED, "Escape:27"):
                break
            if event in ("space:32",):
                self.playing = not self.playing
            if (event in ("Left:37", "Right:39")) and not self.playing:
                step = -1 if "Left" in event else 1
                new_pos = max(0, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) + step)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                frame_idx = new_pos
                self.predictor.buffer.clear()  # Window reset
                continue  # Wait for next loop

            # ---------- Decode & infer next frame ----------
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
            # Numeric feature vector (length = feature_dim)
            feat_vec = pd.DataFrame([pipe.fs.rec]).iloc[0][2:].values.astype(np.float32)
            prob = self.predictor.predict(feat_vec)
            y_pred = int((prob > self.predictor.threshold))

            arr_ts.append(time.time())
            jump_cnt_binary_mark = ((jump_cnt_binary_mark << 1) | y_pred) & 0b111  # Keep last 3 bits
            mark1 = (jump_cnt_binary_mark << 1) & 0b111
            jump_cnt_binary_mark = (mark1 | y_pred) & 0b111
            # print(f"[DEBUG] jump mask: {mark1:03b}+{y_pred:03b}={jump_cnt_binary_mark:03b}")
            if jump_cnt_binary_mark in [3, 7]:  # 3:011 -> 7:111
                is_on_rising = True
                if jump_cnt_binary_mark == 3:  # Only event 3 detected as jump event, increment jump count
                    jump_cnt += 1  # Detected as one jump: 0->1 indicates model detected jump start, 2+ consecutive 1s indicate model considers target still rising
            else:  # 0:000, 1:001, 2:010, 4:100, 5:101, not stable detection result, 6:110 indicates jump rope just ended
                is_on_rising = False

            frame_vis = self._overlay(pipe.fs.raw_frame.copy(), jump_cnt, prob, is_on_rising, arr_ts[0], pipe.landmarks)
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
    parser.add_argument("--model", default="best_cnn_ws4_withT.keras")

    # ========= videos ==========
    # parser.add_argument("--video", default="raw_videos_3/jump_2025.05.14.08.34.44.avi")
    parser.add_argument("--video", default="data/raw_videos_3/jump_2025.05.22.08.33.08__100.avi")
    # parser.add_argument("--video", default="raw_videos_3/jump_2025.05.15.08.37.31.avi")

    # ===========================
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--stick-figure", action="store_true", default=True, help="Show stick figure pose overlay")
    parser.add_argument("--no-stick-figure", action="store_false", dest="stick_figure", help="Hide stick figure pose overlay")
    args = parser.parse_args()

    model_path = f"model_files/models_{len(SELECTED_LM)}_{mode_to_str(get_feature_mode())}/{args.model}"
    predictor = VideoPredictor(model_path, args.threshold)
    gui = PlayerGUI(args.video, predictor, show_stick_figure=args.stick_figure)

    gui.run(get_feature_mode())


if __name__ == "__main__":
    main()
