# main_oop.py
"""
功能：多目标跳绳计数程序骨架（单人示例）
版本：0.4.5
更新日志：
  0.4.0 - 首次引入 Detector/Tracker/Participant 分层架构骨架，单人演示
  0.4.1 - 在 Participant 中标记头/躯干/髋部/膝盖关键点并叠加实时 y 值
  0.4.2 - 改进 Detector：用 MediaPipe 关键点自动算出紧凑包围盒，实际可见检测框
  0.4.3 - 修复 pt 函数返回值导致的 unpack 错误，将其改为返回(point, y_val)二元组
  0.4.4 - 修复坐标投影：使用 pose_landmarks 做图像点投影，使用 pose_world_landmarks 做跳跃信号
  0.4.5 - 添加目标远近与平移方向检测：基于 bbox 中心和尺寸变化，判定左右/上下/远近移动
"""

import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque

# —— Detector 模块 —— #
class Detector:
    """
    负责检测目标。当前版本用 MediaPipe Pose 关键点
    自动生成人物包围盒，未来可换成 YOLO。
    """
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5)

    def detect(self, frame):
        """
        返回 list of bbox，每个 bbox=(x1,y1,x2,y2)
        用关键点 min/max 计算紧凑包围盒，外扩 20px 边距。
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return []
        # 采集所有 landmark 的归一化坐标
        xs = [lm.x for lm in res.pose_landmarks.landmark]
        ys = [lm.y for lm in res.pose_landmarks.landmark]
        # 转为像素并加边距
        x1 = max(int(min(xs) * w) - 20, 0)
        y1 = max(int(min(ys) * h) - 20, 0)
        x2 = min(int(max(xs) * w) + 20, w)
        y2 = min(int(max(ys) * h) + 20, h)
        return [(x1, y1, x2, y2)]


# —— Tracker 模块 —— #
class Tracker:
    """单人演示恒定 ID=0；多人时替换成 SORT/DeepSORT 即可。"""
    def update(self, detections):
        return [(*bbox, 0) for bbox in detections]


# —— Participant 模块 —— #
class JumpPipeline:
    """趋势分离 + 零交叉计数（同 0.2 版本）。"""
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
        self.ALPHA        = alpha
        self.TREND_WIN    = trend_win
        self.BASELINE     = baseline
        self.MIN_INTERVAL = min_interval
        self.ABS_THRESH   = abs_thresh

    def update(self, y):
        if len(self.raw) < self.BASELINE:
            self.raw.append(y); self.smooth.append(y)
            t0 = np.mean(self.raw)
            self.trend.append(t0); self.fluct.append(y - t0)
            if len(self.raw) == self.BASELINE:
                self.noise_std = np.std(self.fluct)
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
            thresh = max(1.5 * (self.noise_std or 0), self.ABS_THRESH)
            if p2p > thresh:
                self.jump_count += 1
                self.last_jump_t = now
        self.prev_sign = sign


class Participant:
    """每个跟踪到的人：姿势→跳绳管线→可视化"""
    def __init__(self, track_id):
        self.id        = track_id
        self.pipeline  = JumpPipeline()
        # MediaPipe Pose（用于关键点可视化）
        self.mp_pose = mp.solutions.pose
        self.pose    = self.mp_pose.Pose(min_detection_confidence=0.5)
        self.joint_points = {}
        self.joint_values = {}
        # 记录上一次中心点和面积，用于运动方向检测
        self.last_center = None  # (cx, cy)
        self.last_area   = None
        self.direction   = None  # 最近一帧的运动方向标签

    def update(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        # 计算运动方向
        if self.last_center is not None and self.last_area is not None:
            dx = cx - self.last_center[0]
            dy = cy - self.last_center[1]
            da = area - self.last_area
            dir_list = []
            # 平移左右阈值 5 像素
            if dx > 5:
                dir_list.append('Right')
            elif dx < -5:
                dir_list.append('Left')
            # 平移上下阈值 5 像素
            if dy > 5:
                dir_list.append('Down')
            elif dy < -5:
                dir_list.append('Up')
            # 尺寸变化表示远近，阈值为面积相对变化 0.05
            if da > self.last_area * 0.05:
                dir_list.append('Closer')
            elif da < -self.last_area * 0.05:
                dir_list.append('Away')
            self.direction = '+'.join(dir_list) if dir_list else 'Static'
        # 更新历史
        self.last_center = (cx, cy)
        self.last_area   = area

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        # 需要同时有图像和世界关键点
        if not res.pose_landmarks or not res.pose_world_landmarks:
            return
        img_lm   = res.pose_landmarks.landmark
        world_lm = res.pose_world_landmarks.landmark

        # 用世界坐标计算跳跃信号
        sy = world_lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y + \
             world_lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        hy = world_lm[self.mp_pose.PoseLandmark.LEFT_HIP].y + \
             world_lm[self.mp_pose.PoseLandmark.RIGHT_HIP].y
        y_val = (sy + hy) / 4.0
        self.pipeline.update(y_val)

        # 改用图像坐标投影关键点
        def pt(idx):
            p = img_lm[idx]
            px = x1 + int(p.x * (x2 - x1))
            py = y1 + int(p.y * (y2 - y1))
            return (px, py), p.y

        self.joint_points['head'],  self.joint_values['head']  = pt(self.mp_pose.PoseLandmark.NOSE)
        # 计算肩部中点
        sx = (img_lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x + img_lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
        sy_norm = (img_lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y + img_lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
        trunk_point = (x1 + int(sx * (x2 - x1)), y1 + int(sy_norm * (y2 - y1)))
        self.joint_points['trunk'], self.joint_values['trunk'] = trunk_point, sy_norm
        self.joint_points['hip'],   self.joint_values['hip']   = pt(self.mp_pose.PoseLandmark.LEFT_HIP)
        self.joint_points['leg'],   self.joint_values['leg']   = pt(self.mp_pose.PoseLandmark.LEFT_KNEE)

    def visualize(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        # 框 & 跳绳计数
        cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 3)
        cv2.putText(frame,
                    f"ID{self.id} J:{self.pipeline.jump_count}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # 显示运动方向
        if self.direction:
            cv2.putText(frame,
                        f"Dir:{self.direction}",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,255), 2)

        # 四个关键点 & 值
        for label, (cx, cy) in self.joint_points.items():
            val = self.joint_values.get(label, 0)
            cv2.circle(frame, (cx, cy), 6, (0,0,255), -1)
            cv2.putText(frame,
                        f"{label}:{val:.2f}",
                        (cx+8, cy-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,255,255), 2)


# —— App/Manager 主控 —— #
class RopeJumpApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector    = Detector()
        self.tracker     = Tracker()
        self.participants = {}

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            dets   = self.detector.detect(frame)
            tracks = self.tracker.update(dets)

            for x1, y1, x2, y2, tid in tracks:
                if tid not in self.participants:
                    self.participants[tid] = Participant(tid)
                p = self.participants[tid]
                p.update(frame, (x1,y1,x2,y2))
                p.visualize(frame, (x1,y1,x2,y2))

            cv2.imshow("RopeJump OOP 0.4.2", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    RopeJumpApp().run()