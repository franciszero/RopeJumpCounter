"""
功能：多人体实时跳绳计数，支持背景运动补偿
---------------------------------------------------------------
需求：
  1. 实时统计一个或多个人的跳绳次数；
  2. 基于姿态估计获取标准化躯干高度（图像坐标）；
  3. 使用光流估计背景运动，实现相对速度补偿；
  4. 趋势分离：指数平滑 + 移动平均趋势；
  5. 去趋势波动零交叉检测触发计数；
  6. 调试可视化：显示原始、平滑、趋势和波动曲线；
  7. 可配置缓冲长度、平滑参数和光流跟踪设置。

更新日志：
  - 2025-04-24 v0.2：增加背景光流补偿相对运动功能
    * 使用 Lucas-Kanade 光流跟踪背景特征点
    * 计算人体相对垂直速度（体动-背景）
    * 将相对速度信号输入原有趋势分离与零交叉算法
"""

import cv2
import numpy as np
import time
import mediapipe as mp
from collections import deque

# ===== Configuration =====
VERSION = "v0.2"
BUFFER_LEN = 320  # Number of samples to keep for each curve
ALPHA = 0.2  # Exponential smoothing factor
TREND_WINDOW = 64  # Window size for moving average trend
BASELINE_BUF = 150  # Initial frames to collect baseline
MIN_INTERVAL = 0.3  # Minimum seconds between counts
MAX_BG_PTS = 200  # Max background points to track
# =========================

# ===== State Buffers =====
raw_buf = deque(maxlen=BUFFER_LEN)
smooth_buf = deque(maxlen=BUFFER_LEN)
trend_buf = deque(maxlen=BUFFER_LEN)
fluct_buf = deque(maxlen=BUFFER_LEN)

jump_count = 0
prev_sign = -1
last_jump_time = 0
frame_idx = 0

# For background flow
prev_gray = None
bg_pts = None

# For body delta
torso_y_prev = None


# =========================

def main():
    global frame_idx, jump_count, prev_sign, last_jump_time
    global prev_gray, bg_pts, torso_y_prev

    # Initialize MediaPipe Pose (normalized image coords)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Initialize background points on first frame ---
        if frame_idx == 1:
            # detect up to MAX_BG_PTS good features (corners) in background
            bg_pts = cv2.goodFeaturesToTrack(gray,
                                             maxCorners=MAX_BG_PTS,
                                             qualityLevel=0.01,
                                             minDistance=10)
            prev_gray = gray.copy()

        # Pose estimation (normalized y)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            sy = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y + lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            hy = lm[mp_pose.PoseLandmark.LEFT_HIP].y + lm[mp_pose.PoseLandmark.RIGHT_HIP].y
            torso_y = (sy + hy) / 4.0
        else:
            torso_y = torso_y_prev if torso_y_prev is not None else 0.5

        # On frame_idx 1 set previous torso_y
        if torso_y_prev is None:
            torso_y_prev = torso_y

        # initialize background delta in case tracking hasn't set it
        bg_dy_norm = 0.0

        # --- Compute background vertical movement (normalized) ---
        if frame_idx > 1 and bg_pts is not None and len(bg_pts) > 0:
            new_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, bg_pts, None, **lk_params)
            # keep only successfully tracked points
            mask = (st.flatten() == 1)
            if bg_pts is not None and mask.any():
                # reshape to N x 2 for indexing
                good0 = bg_pts[mask].reshape(-1, 2)
                good1 = new_pts[mask].reshape(-1, 2)
                # vertical pixel shifts
                dy_vals = good1[:, 1] - good0[:, 1]
                bg_dy_norm = np.median(dy_vals) / h
                # update for next iteration
                bg_pts = good1.reshape(-1, 1, 2)
                prev_gray = gray.copy()
            else:
                bg_dy_norm = 0.0

        # --- Body vertical movement (normalized) ---
        body_dy_norm = torso_y - torso_y_prev
        torso_y_prev = torso_y

        # --- Relative speed signal ---
        rel_speed = body_dy_norm - bg_dy_norm

        # --- Baseline collection or trend processing ---
        if frame_idx <= BASELINE_BUF:
            raw_buf.append(0.0)
            smooth_buf.append(0.0)
            trend_buf.append(0.0)
            fluct_buf.append(0.0)
        else:
            raw_buf.append(rel_speed)
            # exponential smoothing
            last_s = smooth_buf[-1]
            s = ALPHA * rel_speed + (1 - ALPHA) * last_s
            # moving-average trend
            window = list(smooth_buf)[-TREND_WINDOW:]
            t = np.mean(window)
            # detrended fluctuation
            f = s - t

            smooth_buf.append(s)
            trend_buf.append(t)
            fluct_buf.append(f)

            # zero-cross count
            sign = 1 if f > 0 else -1
            now = time.time()
            if sign > 0 and prev_sign < 0 and (now - last_jump_time) > MIN_INTERVAL:
                jump_count += 1
                last_jump_time = now
            prev_sign = sign

        # --- Visualization ---
        cv2.putText(frame, f"Jumps: {jump_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)

        # build debug canvas at right
        canvas_h, canvas_w = h, BUFFER_LEN
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        row_h = canvas_h // 4
        series = [
            (raw_buf, "Raw RelSpd"),
            (smooth_buf, "Smoothed"),
            (trend_buf, "Trend"),
            (fluct_buf, "Fluctuation")
        ]
        for idx, (buf, label) in enumerate(series):
            arr = np.array(buf)
            y0 = idx * row_h
            y1 = y0 + row_h
            if len(arr) >= 2:
                mn, mx = arr.min(), arr.max()
                norm = (arr - mn) / (mx - mn) if mx > mn else np.full_like(arr, 0.5)
                for i in range(1, len(norm)):
                    x1, x2 = i - 1, i
                    yy1 = int(y1 - norm[i - 1] * row_h)
                    yy2 = int(y1 - norm[i] * row_h)
                    cv2.line(canvas, (x1, yy1), (x2, yy2), (200, 200, 200), 1)
            cv2.putText(canvas, label, (5, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        full_img = cv2.hconcat([frame, canvas])
        cv2.imshow("JumpRope Debug v0.2", full_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
