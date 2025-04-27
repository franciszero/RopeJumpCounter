# main_oop.py
"""
功能：多人跳绳计数程序骨架（单人模式演示）
版本：0.4.0
更新日志：
  0.4.0 - 首次引入 Detector/Tracker/Participant 分层架构骨架，单人模式演示
"""

import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque

# —— Detector 模块 —— #
class Detector:
    """负责检测目标（框），当前仅返回整帧中心区作为单人示例"""
    def __init__(self, method="mediapipe"):
        self.method = method
        if method == "mediapipe":
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(min_detection_confidence=0.5)
        else:
            raise NotImplementedError

    def detect(self, frame):
        """
        返回列表 of (x1,y1,x2,y2) 检测框。
        单人模式：返回全画面。
        """
        h, w = frame.shape[:2]
        return [(0, 0, w, h)]


# —— Tracker 模块 —— #
class Tracker:
    """负责给每个框分配稳定 ID，单人模式总是 ID=0"""
    def __init__(self):
        self.next_id = 0

    def update(self, detections):
        """
        detections: list of (x1,y1,x2,y2)
        返回 list of (x1,y1,x2,y2,track_id)
        """
        return [(*bbox, 0) for bbox in detections]


# —— Participant 模块 —— #
class JumpPipeline:
    """信号处理：趋势分离 + 零交叉计数"""
    def __init__(self, buf_len=320, alpha=0.2, trend_win=64,
                 baseline=150, min_interval=0.3, abs_thresh=0.02):
        self.raw    = deque(maxlen=buf_len)
        self.smooth = deque(maxlen=buf_len)
        self.trend  = deque(maxlen=buf_len)
        self.fluct  = deque(maxlen=buf_len)
        self.noise_std     = None
        self.prev_sign     = -1
        self.last_jump_t   = 0.0
        self.jump_count    = 0
        # params
        self.ALPHA         = alpha
        self.TREND_WIN     = trend_win
        self.BASELINE      = baseline
        self.MIN_INTERVAL  = min_interval
        self.ABS_THRESH    = abs_thresh

    def update(self, y):
        """向管线输入一个标量 y，更新并可能累加一次跳跃"""
        if len(self.raw) < self.BASELINE:
            self.raw.append(y)
            self.smooth.append(y)
            t0 = np.mean(self.raw)
            self.trend.append(t0)
            self.fluct.append(y - t0)
            if len(self.raw) == self.BASELINE:
                self.noise_std = np.std(self.fluct)
            return
        # 指数平滑
        last_s = self.smooth[-1]
        s = self.ALPHA * y + (1 - self.ALPHA) * last_s
        # 计算趋势
        t = np.mean(list(self.smooth)[-self.TREND_WIN:])
        # 去趋势
        f = s - t
        # 存储
        self.raw.append(y)
        self.smooth.append(s)
        self.trend.append(t)
        self.fluct.append(f)
        # 零交叉计数
        sign = 1 if f > 0 else -1
        now = time.time()
        if sign > 0 and self.prev_sign < 0 and now - self.last_jump_t > self.MIN_INTERVAL:
            seg = list(self.fluct)[-self.TREND_WIN:]
            p2p = max(seg) - min(seg)
            thresh = max(1.5*(self.noise_std or 0), self.ABS_THRESH)
            if p2p > thresh:
                self.jump_count += 1
                self.last_jump_t = now
        self.prev_sign = sign


class Participant:
    """每个被跟踪的“人”实例，负责姿势+跳绳管线"""
    def __init__(self, track_id):
        self.id = track_id
        self.pipeline = JumpPipeline()
        # MediaPipe Pose（复用 Detector 中的模型也可）
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5)

    def update(self, frame, bbox):
        """给定一帧与对应 bbox，做姿势分析并更新管线"""
        x1,y1,x2,y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_world_landmarks:
            return
        lm = res.pose_world_landmarks.landmark
        # 用躯干中点高度作为信号
        sy = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y + \
             lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        hy = lm[self.mp_pose.PoseLandmark.LEFT_HIP].y + \
             lm[self.mp_pose.PoseLandmark.RIGHT_HIP].y
        y_val = (sy + hy) / 4.0
        self.pipeline.update(y_val)

    def visualize(self, frame, bbox):
        """在 frame 上绘制 bbox、计数和尾迹波形（可选）"""
        x1,y1,x2,y2 = map(int, bbox)
        color = (0,255,0)
        # 画框和计数
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        cv2.putText(frame, f"ID{self.id} J:{self.pipeline.jump_count}",
                    (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2)


# —— App/Manager 主控 —— #
class RopeJumpApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # 单人模式 demo，后续可扩展为 Detector("yolo") + Tracker()
        self.detector = Detector()
        self.tracker  = Tracker()
        # track_id -> Participant
        self.participants = {}

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 1. 检测
            dets = self.detector.detect(frame)  # list of bboxes

            # 2. 跟踪
            tracks = self.tracker.update(dets)  # list of (bbox + id)

            # 3. 更新/可视化每个 Participant
            for x1,y1,x2,y2,tid in tracks:
                # 新 ID 时创建实例
                if tid not in self.participants:
                    self.participants[tid] = Participant(tid)
                p = self.participants[tid]
                bbox = (x1,y1,x2,y2)
                p.update(frame, bbox)
                p.visualize(frame, bbox)

            # 显示
            cv2.imshow("RopeJump OOP Demo", frame)
            if cv2.waitKey(1)&0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    RopeJumpApp().run()