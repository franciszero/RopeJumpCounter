# main_oop.py
"""
功能：多目标跳绳计数程序骨架（单人示例）
版本：0.4.12
更新日志：
  0.4.0 - 首次引入 Detector/Tracker/Participant 分层架构骨架，单人演示
  0.4.1 - 在 Participant 中标记头/躯干/髋部/膝盖关键点并叠加实时 y 值
  0.4.2 - 改进 Detector：用 MediaPipe 关键点自动算出紧凑包围盒，实际可见检测框
  0.4.3 - 修复 pt 函数返回值导致的 unpack 错误，将其改为返回(point, y_val)二元组
  0.4.4 - 修复坐标投影：使用 pose_landmarks 做图像点投影，使用 pose_world_landmarks 做跳跃信号
  0.4.5 - 添加目标远近与平移方向检测：基于 bbox 中心和尺寸变化，判定左右/上下/远近移动
  0.4.6 - 移除 bbox 检测，启用全帧姿势检测，仅显示关键节点，无计数
  0.4.7 - 集成 MediaPipe 绘制函数，显示完整骨架（火柴人）
  0.4.8 - 添加运动方向检测：基于关键点包围盒中心与面积变化，判定左/右/靠近/远离，并在屏幕及标准输出显示
  0.4.9 - 四标签 左近远右 动态字体大小显示移动速度，标准输出打印
  0.4.10 - 将方向标签替换为 ASCII 字符 L/N/F/R，并增大间距避免重叠
  0.4.11 - 实现第二步: 头部时间序列波形绘制及分割标记
  0.4.12 - 修复画面拼接错误：draw_panel 返回新图像并在 run 中接收
"""

import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque
# 右侧面板宽度（也等于时间序列缓冲长度）
PANEL_WIDTH = 320


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
        self.latest_landmarks = None
        # 用于四方向速度显示
        self.dx = 0
        self.da = 0

    def update(self, frame):
        # reset movement deltas
        self.dx = 0
        self.da = 0
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

        # 运动方向检测：基于归一化关键点包围盒中心与面积变化
        # 计算归一化 bbox min/max
        xs = [p.x for p in img_lm]
        ys = [p.y for p in img_lm]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        area = (max_x - min_x) * (max_y - min_y)
        # 四方向检测与速度
        if self.last_center is not None:
            dx = center_x - self.last_center[0]
            da = area - self.last_area
            # store for visualization
            self.dx = dx
            self.da = da
            # magnitudes for left, near, far, right
            mags = [max(0, -dx), max(0, da), max(0, -da), max(0, dx)]
            max_mag = max(mags)
            chars = ['L', 'N', 'F', 'R']  # Left, Near, Far, Right (ASCII avoids乱码)
            # print to stdout the strongest direction
            if max_mag > 0:
                idx = mags.index(max_mag)
                print(f"ID{self.id} Dir:{chars[idx]} speed:{max_mag:.4f}")
        # update history
        self.last_center = (center_x, center_y)
        self.last_area   = area

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
        # four-direction display with dynamic font size
        # compute magnitudes
        mag_left  = max(0, -self.dx)
        mag_near  = max(0, self.da)
        mag_far   = max(0, -self.da)
        mag_right = max(0, self.dx)
        mags = [mag_left, mag_near, mag_far, mag_right]
        max_mag = max(mags)
        chars = ['L', 'N', 'F', 'R']  # Left, Near, Far, Right (ASCII avoids乱码)
        x0, y0 = 10, 30
        gap = 60  # 增大间距
        for i, ch in enumerate(chars):
            mag = mags[i]
            if max_mag > 1e-6:
                scale = 0.8 + (mag / max_mag) * 0.7
            else:
                scale = 0.8
            cv2.putText(frame, ch, (x0 + gap * i, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (0, 255, 255), 2)


# —— App/Manager 主控 —— #
class RopeJumpApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # 单人全帧姿势检测，只维护一个 Participant
        self.participant = Participant(0)
        # head 节点时间序列缓存
        self.head_buffer = deque(maxlen=PANEL_WIDTH)

    def draw_panel(self, frame):
        """在画面右侧绘制 head 时间序列及分割标记"""
        h, w, _ = frame.shape
        # 新建画布
        panel = np.zeros((h, PANEL_WIDTH, 3), dtype=np.uint8)
        vals = list(self.head_buffer)
        if len(vals) > 1:
            arr = np.array(vals)
            mn, mx = arr.min(), arr.max()
            norm = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)
            ys = (h - 1 - (norm * (h - 1))).astype(int)
            xs = np.linspace(0, PANEL_WIDTH - 1, len(ys)).astype(int)
            # 绘制波形
            for i in range(1, len(xs)):
                cv2.line(panel,
                         (xs[i-1], ys[i-1]),
                         (xs[i],   ys[i]),
                         (0, 255, 255), 1)
            # 分割标记：零交叉点
            mean = arr.mean()
            for i in range(1, len(arr)):
                if (arr[i] - mean) > 0 and (arr[i-1] - mean) <= 0:
                    x = int(i * PANEL_WIDTH / len(arr))
                    cv2.line(panel, (x, 0), (x, h - 1), (0, 0, 255), 1)
        # 拼接
        # 返回拼接后的全画面
        return np.hstack((frame, panel))

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 全帧姿势检测
            p = self.participant
            p.update(frame)

            # 记录 head y 值
            head_val = p.joint_values.get('head')
            if head_val is not None:
                self.head_buffer.append(head_val)

            p.visualize(frame)

            # 绘制并接收拼接面板后的新画面
            frame = self.draw_panel(frame)
            cv2.imshow("RopeJump OOP 0.4.12", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    RopeJumpApp().run()