# main-0.3.py
"""
功能：基于 MediaPipe Pose 的单人跳绳计数程序，检测四个关节，支持趋势分解与零交叉计数。
版本：0.3.9
# 0.3.9 - 恢复单人姿势关节检测，屏蔽多人跟踪和平滑，仅保留趋势分解功能

更新日志：
  0.2.0 - 单人跳绳计数初始版本，实现趋势分离和零交叉过零点计数，带调试波形显示
  0.2.1 - 增加双人 YOLO 检测支持，支持最多 2 人计数，保留调试波形显示
  0.2.2 - 修复断绳时无人姿势检测导致的误计数：仅在检测到人体姿势时更新跳跃管线
  0.2.3 - 将时间序列曲线移至摄像头图像右侧
  0.3.0 - 重构为多目标计数：集成 SORT，任意人数跟踪，按 ID 并行维护跳绳管线
  0.3.3 - 注释 sort/sort.py 中的 matplotlib.use('TkAgg') 避免 tkinter 依赖
  0.3.6 - 限制最多前五个跟踪对象绘制时间序列，并分行显示避免重叠
  0.3.7 - 使用原始数据绘制波形并移除平滑；在摄像头画面上为每个对象绘制尾迹线
  0.3.8 - 调整 SORT 参数：降低 iou_threshold，增加 max_age 和 min_hits，提高跟踪稳定性
  0.3.9 - 恢复单人姿势关节检测，屏蔽多人跟踪和平滑，仅保留趋势分解功能

依赖：
  - mediapipe
  - opencv-python
  - numpy
  - ultralytics
  - sort    （pip install sort）
用法：
  1. pip install mediapipe opencv-python numpy ultralytics sort
  2. python main-0.3.py
"""

import cv2
import numpy as np
import time
import mediapipe as mp
from collections import deque

# 配色列表，用于不同关节
COLORS = {
    'head':  (0,255,0),
    'trunk': (0,0,255),
    'hip':   (255,0,0),
    'leg':   (255,255,0),
}
# 跟踪尾迹长度
TRACE_LEN = 20

# 参数配置
BUFFER_LEN     = 320
#ALPHA          = 0.2
TREND_WINDOW   = 64
BASELINE_BUF   = 150
MIN_INTERVAL   = 0.3

# 初始化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

class JumpPipeline:
    def __init__(self):
        self.raw = deque(maxlen=BUFFER_LEN)
        self.smooth = deque(maxlen=BUFFER_LEN)
        self.trend = deque(maxlen=BUFFER_LEN)
        self.fluct = deque(maxlen=BUFFER_LEN)
        self.noise_std = None
        self.prev_sign = -1
        self.last_jump_time = 0
        self.jump_count = 0
        self.trace = deque(maxlen=TRACE_LEN)

    def update(self, y):
        if len(self.raw) < BASELINE_BUF:
            self.raw.append(y)
            self.smooth.append(y)
            t0 = np.mean(self.raw)
            self.trend.append(t0)
            self.fluct.append(y - t0)
            if len(self.raw)==BASELINE_BUF:
                self.noise_std = np.std(np.array(self.fluct))
            return
        #last_s = self.smooth[-1]
        #s = ALPHA*y + (1-ALPHA)*last_s  # 关闭平滑，直接用原始y
        s = y
        t = np.mean(list(self.smooth)[-TREND_WINDOW:])
        f = s - t
        self.raw.append(y)
        self.smooth.append(s)
        self.trend.append(t)
        self.fluct.append(f)
        sign = 1 if f>0 else -1
        now = time.time()
        if sign>0 and self.prev_sign<0 and now-self.last_jump_time>MIN_INTERVAL:
            seg = list(self.fluct)[-TREND_WINDOW:]
            if self.noise_std and (max(seg)-min(seg)>self.noise_std):
                self.jump_count +=1
                self.last_jump_time=now
        self.prev_sign = sign

# 主程序
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

# 单人模式：四个关节各自一条跳跃管线
joint_pipelines = {
    'head':  JumpPipeline(),
    'trunk': JumpPipeline(),
    'hip':   JumpPipeline(),
    'leg':   JumpPipeline(),
}

while True:
    ret, frame = cap.read()
    if not ret: break
    h,w,_ = frame.shape

    # 单人姿势检测
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if not res.pose_world_landmarks:
        cv2.imshow("RopeJump 0.3.9 单人调试", frame)
        if cv2.waitKey(1)&0xFF==27: break
        continue
    lm = res.pose_world_landmarks.landmark
    # 计算四个关节点的 y 值
    head_y  = lm[mp_pose.PoseLandmark.NOSE].y
    trunk_y = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y + lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
    hip_y   = (lm[mp_pose.PoseLandmark.LEFT_HIP].y + lm[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
    leg_y   = (lm[mp_pose.PoseLandmark.LEFT_KNEE].y + lm[mp_pose.PoseLandmark.RIGHT_KNEE].y) / 2

    # 更新每个关节的管线，记录尾迹并绘制
    for joint in joint_pipelines.keys():
        joint_y = locals()[f"{joint}_y"]
        p = joint_pipelines[joint]
        # 记录关节尾迹点 (x,y)
        # 获取关节点在原图像的像素坐标
        if joint == 'head':
            idx = mp_pose.PoseLandmark.NOSE
        elif joint == 'trunk':
            idx1 = mp_pose.PoseLandmark.LEFT_SHOULDER
            idx2 = mp_pose.PoseLandmark.RIGHT_SHOULDER
            x = int((lm[idx1].x + lm[idx2].x)/2 * w)
            y = int((lm[idx1].y + lm[idx2].y)/2 * h)
        elif joint == 'hip':
            idx1 = mp_pose.PoseLandmark.LEFT_HIP
            idx2 = mp_pose.PoseLandmark.RIGHT_HIP
            x = int((lm[idx1].x + lm[idx2].x)/2 * w)
            y = int((lm[idx1].y + lm[idx2].y)/2 * h)
        elif joint == 'leg':
            idx1 = mp_pose.PoseLandmark.LEFT_KNEE
            idx2 = mp_pose.PoseLandmark.RIGHT_KNEE
            x = int((lm[idx1].x + lm[idx2].x)/2 * w)
            y = int((lm[idx1].y + lm[idx2].y)/2 * h)
        if joint == 'head':
            x = int(lm[mp_pose.PoseLandmark.NOSE].x * w)
            y = int(lm[mp_pose.PoseLandmark.NOSE].y * h)
        p.trace.append((x, y))
        # 更新管线
        p.update(joint_y)
        # 绘制尾迹
        color = COLORS[joint]
        for i in range(1, len(p.trace)):
            pt1 = p.trace[i-1]
            pt2 = p.trace[i]
            cv2.line(frame, pt1, pt2, color, 2)
        # 绘制关节计数
        cv2.putText(frame, f"{joint}:{p.jump_count}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 显示画面
    cv2.imshow("RopeJump 0.3.9 单人调试", frame)
    if cv2.waitKey(1)&0xFF==27: break

cap.release()
cv2.destroyAllWindows()
