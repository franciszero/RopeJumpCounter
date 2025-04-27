# main_oop.py
"""
功能：多目标跳绳计数程序骨架（单人示例）
版本：0.4.7
更新日志：
  0.4.0 - 首次引入 Detector/Tracker/Participant 分层架构骨架，单人演示
  0.4.1 - 在 Participant 中标记头/躯干/髋部/膝盖关键点并叠加实时 y 值
  0.4.2 - 改进 Detector：用 MediaPipe 关键点自动算出紧凑包围盒，实际可见检测框
  0.4.3 - 修复 pt 函数返回值导致的 unpack 错误，将其改为返回(point, y_val)二元组
  0.4.4 - 修复坐标投影：使用 pose_landmarks 做图像点投影，使用 pose_world_landmarks 做跳跃信号
  0.4.5 - 添加目标远近与平移方向检测：基于 bbox 中心和尺寸变化，判定左右/上下/远近移动
  0.4.6 - 移除 bbox 检测，启用全帧姿势检测，仅显示关键节点，无计数
  0.4.7 - 集成 MediaPipe 绘制函数，显示完整骨架（火柴人）
"""

import cv2
import time
import numpy as np
import mediapipe as mp


# —— Participant 模块 —— #
class Participant:
    """每个跟踪到的人：姿势→跳绳管线→可视化"""
    def __init__(self, track_id):
        self.id        = track_id
        # MediaPipe Pose（用于关键点可视化）
        self.mp_pose = mp.solutions.pose
        self.pose    = self.mp_pose.Pose(min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.joint_points = {}
        self.joint_values = {}
        # 记录上一次中心点和面积，用于运动方向检测
        self.last_center = None  # (cx, cy)
        self.last_area   = None
        self.direction   = None  # 最近一帧的运动方向标签
        self.latest_landmarks = None

    def update(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_img = self.pose.process(rgb)
        # 保存最新的图像关键点用于可视化骨架
        self.latest_landmarks = res_img.pose_landmarks
        if not res_img.pose_landmarks:
            return
        img_lm = res_img.pose_landmarks.landmark
        h, w, _ = frame.shape

        def pt(idx):
            p = img_lm[idx]
            return (int(p.x * w), int(p.y * h)), p.y

        self.joint_points['head'],  self.joint_values['head']  = pt(self.mp_pose.PoseLandmark.NOSE)
        # 躯干中点:
        sx = (img_lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x + img_lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x)/2
        sy = (img_lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y + img_lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y)/2
        trunk_pt = (int(sx * w), int(sy * h))
        self.joint_points['trunk'], self.joint_values['trunk'] = trunk_pt, sy
        self.joint_points['hip'],   self.joint_values['hip']   = pt(self.mp_pose.PoseLandmark.LEFT_HIP)
        self.joint_points['leg'],   self.joint_values['leg']   = pt(self.mp_pose.PoseLandmark.LEFT_KNEE)

    def visualize(self, frame):
        # 绘制完整骨架火柴人
        if self.latest_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                self.latest_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        # 仅绘制关键点
        for label, (cx, cy) in self.joint_points.items():
            val = self.joint_values[label]
            cv2.circle(frame, (cx,cy), 6, (0,0,255), -1)
            cv2.putText(frame, f"{label}:{val:.2f}", (cx+5,cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)


# —— App/Manager 主控 —— #
class RopeJumpApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # 单人全帧姿势检测，只维护一个 Participant
        self.participant = Participant(0)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 全帧姿势检测
            p = self.participant
            p.update(frame)
            p.visualize(frame)

            cv2.imshow("RopeJump OOP 0.4.6", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    RopeJumpApp().run()