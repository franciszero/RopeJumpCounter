# main_oop.py
"""
功能：多人跳绳计数程序骨架（单人模式演示）
版本：0.4.1
更新日志：
  0.4.0 - 首次引入 Detector/Tracker/Participant 分层架构骨架，单人模式演示
  0.4.1 - 在 Participant 中标记头、躯干、髋部、膝盖等关键节点，并在画面上叠加实时 y 值
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
        """返回列表 of (x1,y1,x2,y2) 检测框。单人模式：全画面。"""
        h, w = frame.shape[:2]
        return [(0, 0, w, h)]


# —— Tracker 模块 —— #
class Tracker:
    """负责给每个框分配稳定 ID，单人模式总是 ID=0"""
    def __init__(self):
        pass

    def update(self, detections):
        """detections: list of (x1,y1,x2,y2)；返回 list of (x1,y1,x2,y2,track_id)"""
        return [(*bbox, 0) for bbox in detections]


# —— Participant 模块 —— #
class JumpPipeline:
    """信号处理：趋势分离 + 零交叉计数（同 0.2 版本）"""
    def __init__(self, buf_len=320, alpha=0.2, trend_win=64,
                 baseline=150, min_interval=0.3, abs_thresh=0.02):
        self.raw    = deque(maxlen=buf_len)
        self.smooth = deque(maxlen=buf_len)
        self.trend  = deque(maxlen=buf_len)
        self.fluct  = deque(maxlen=buf_len)
        self.noise_std   = None
        self.prev_sign   = -1
        self.last_jump_t = 0.0
        self.jump_count  = 0
        # 参数
        self.ALPHA        = alpha
        self.TREND_WIN    = trend_win
        self.BASELINE     = baseline
        self.MIN_INTERVAL = min_interval
        self.ABS_THRESH   = abs_thresh

    def update(self, y):
        # 跟 0.2 逻辑完全一致
        if len(self.raw) < self.BASELINE:
            self.raw.append(y)
            self.smooth.append(y)
            t0 = np.mean(self.raw)
            self.trend.append(t0)
            self.fluct.append(y - t0)
            if len(self.raw) == self.BASELINE:
                self.noise_std = np.std(list(self.fluct))
            return
        last_s = self.smooth[-1]
        s = self.ALPHA * y + (1 - self.ALPHA) * last_s
        t = np.mean(list(self.smooth)[-self.TREND_WIN:])
        f = s - t
        self.raw.append(y);    self.smooth.append(s)
        self.trend.append(t);  self.fluct.append(f)
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
    """每个被跟踪的“人”，负责姿势+跳绳管线+可视化"""
    def __init__(self, track_id):
        self.id        = track_id
        self.pipeline  = JumpPipeline()
        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose    = self.mp_pose.Pose(min_detection_confidence=0.5)
        # 本次迭代新增：存储 4 个关键点的坐标与归一化 y 值
        self.joint_points = {}
        self.joint_values = {}

    def update(self, frame, bbox):
        x1, y1, x2, y2, = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_world_landmarks:
            return

        lm = res.pose_world_landmarks.landmark
        # 计算躯干中点高度信号
        sy = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y + \
             lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        hy = lm[self.mp_pose.PoseLandmark.LEFT_HIP].y + \
             lm[self.mp_pose.PoseLandmark.RIGHT_HIP].y
        y_val = (sy + hy) / 4.0

        # —— 本次迭代：计算四关键点的像素坐标与归一化 y 值 —— #
        # 头部 (Nose)
        nose = lm[self.mp_pose.PoseLandmark.NOSE]
        hx = x1 + int(nose.x * (x2 - x1)); hy_px = y1 + int(nose.y * (y2 - y1))
        # 躯干 (肩膀中点)
        ls = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        tx = x1 + int(((ls.x+rs.x)/2)*(x2-x1)); ty = y1 + int(((ls.y+rs.y)/2)*(y2-y1))
        # 髋部 (髋骨中点)
        lh = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
        rh = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]
        hipx = x1 + int(((lh.x+rh.x)/2)*(x2-x1)); hipy = y1 + int(((lh.y+rh.y)/2)*(y2-y1))
        # 腿部 (膝盖中点)
        lk = lm[self.mp_pose.PoseLandmark.LEFT_KNEE]
        rk = lm[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        legx = x1 + int(((lk.x+rk.x)/2)*(x2-x1)); legy = y1 + int(((lk.y+rk.y)/2)*(y2-y1))

        self.joint_points = {
            'head':  (hx,   hy_px),
            'trunk': (tx,   ty),
            'hip':   (hipx, hipy),
            'leg':   (legx, legy)
        }
        self.joint_values = {
            'head':  nose.y,
            'trunk': (ls.y+rs.y)/2,
            'hip':   (lh.y+rh.y)/2,
            'leg':   (lk.y+rk.y)/2
        }

        # 更新跳绳管线
        self.pipeline.update(y_val)

    def visualize(self, frame, bbox):
        x1, y1, x2, y2, = map(int, bbox)
        # 画框 & 计数
        color = (0,255,0)
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        cv2.putText(frame,
                    f"ID{self.id} J:{self.pipeline.jump_count}",
                    (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # —— 本次迭代：绘制关键点与 y 值 —— #
        for label, (cx, cy) in self.joint_points.items():
            # 小圆点
            cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)
            # 文本：label:归一化 y
            val = self.joint_values.get(label, 0)
            cv2.putText(frame,
                        f"{label}:{val:.2f}",
                        (cx+5, cy-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255,255,255), 1)


# —— App/Manager 主控 —— #
class RopeJumpApp:
    def __init__(self):
        self.cap        = cv2.VideoCapture(0)
        self.detector   = Detector()
        self.tracker    = Tracker()
        self.participants = {}

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            dets   = self.detector.detect(frame)
            tracks = self.tracker.update(dets)
            for x1,y1,x2,y2,tid in tracks:
                if tid not in self.participants:
                    self.participants[tid] = Participant(tid)
                p = self.participants[tid]
                p.update(frame, (x1,y1,x2,y2))
                p.visualize(frame, (x1,y1,x2,y2))
            cv2.imshow("RopeJump OOP 0.4.1", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    RopeJumpApp().run()