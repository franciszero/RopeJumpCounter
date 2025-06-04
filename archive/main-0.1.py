# main.py

import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
import mediapipe as mp

# ————— 常量配置 ————— #
DETECT_INTERVAL = 5  # 每隔多少帧做一次 YOLO 检测
BUFFER_LEN = 320  # 曲线缓存长度（像素宽度）
ALPHA = 0.2  # 指数平滑系数
TREND_WINDOW = 64  # 移动平均趋势窗口（帧数）
BASELINE_BUF = 150  # 启动时采集的基线帧数
# ————— 常量配置 ————— #
IMG_WIDTH = 640
IMG_HEIGHT = 480
MAX_MISSES = DETECT_INTERVAL * 2  # 超过此帧数未更新即移除无效 tracker
# ———————————————————— #

# ————— 全局状态 ————— #
model = YOLO('yolov10n.pt')  # YOLOv10n 模型
current_boxes = []
pose = mp.solutions.pose.Pose(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 每个目标的状态字典，key=target_id, value=dict(…buffers…, jump_count, prev_sign, frame_idx, baseline_done)
trackers = {}

frame_idx = 0


# ———————————————— #

def init_target():
    """初始化一个新目标的状态字典"""
    return {
        'raw_buf': deque(maxlen=BUFFER_LEN),
        'smooth_buf': deque(maxlen=BUFFER_LEN),
        'trend_buf': deque(maxlen=BUFFER_LEN),
        'fluct_buf': deque(maxlen=BUFFER_LEN),
        'jump_count': 0,
        'prev_sign': -1,
        'frame_idx': 0,
        'baseline_done': False,
        'baseline_vals': [],
        'last_seen': 0,
    }


def process_signal(data, val):
    """对单个目标数据进行更新与计数，返回当前 fluct"""
    data['frame_idx'] += 1
    idx = data['frame_idx']
    # 启动阶段采集基线
    if idx <= BASELINE_BUF:
        data['baseline_vals'].append(val)
        data['raw_buf'].append(val)
        data['smooth_buf'].append(val)
        data['trend_buf'].append(np.mean(data['baseline_vals']))
        data['fluct_buf'].append(0.0)
        if idx == BASELINE_BUF:
            data['baseline_done'] = True
        return 0.0
    # 指数平滑
    last_s = data['smooth_buf'][-1]
    s = ALPHA * val + (1 - ALPHA) * last_s
    # 移动平均趋势
    window = list(data['smooth_buf'])[-TREND_WINDOW:]
    t = np.mean(window)
    # 去趋势波动
    f = s - t
    # 更新 buffers
    data['raw_buf'].append(val)
    data['smooth_buf'].append(s)
    data['trend_buf'].append(t)
    data['fluct_buf'].append(f)
    # 零交叉检测
    sign = 1 if f > 0 else -1
    if sign > 0 and data['prev_sign'] < 0:
        data['jump_count'] += 1
    data['prev_sign'] = sign
    return f


def draw_pose_and_count(frame, lm_list, x1, y1, x2, y2, count):
    """在原图上绘制骨架和计数"""
    h_roi = y2 - y1;
    w_roi = x2 - x1
    for i, lm in enumerate(lm_list):
        px = int(lm.x * w_roi) + x1
        py = int(lm.y * h_roi) + y1
        cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
    # 连线（简易骨架：肩-肘-腕、髋-膝-踝）
    pairs = [
        (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
        (mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST),
        (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
        (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
        (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE),
        (mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
        (mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
        (mp.solutions.pose.PoseLandmark.RIGHT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE)
    ]
    for a, b in pairs:
        p1 = lm_list[a.value];
        p2 = lm_list[b.value]
        pt1 = (int(p1.x * w_roi) + x1, int(p1.y * h_roi) + y1)
        pt2 = (int(p2.x * w_roi) + x1, int(p2.y * h_roi) + y1)
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    # 计数显示
    cv2.putText(frame, f"ID {count['id']} Jumps: {count['jump_count']}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def main():
    global frame_idx
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

    global current_boxes
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        h, w, _ = frame.shape

        # 1. 每隔 DETECT_INTERVAL 帧，用 YOLOv10 更新检测框
        if frame_idx % DETECT_INTERVAL == 0:
            results = model.predict(frame, imgsz=320, conf=0.5)
            current_boxes = []
            for r in results:
                for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                    if int(cls) == 0:
                        x1, y1, x2, y2 = map(int, box.cpu().numpy())
                        current_boxes.append((x1, y1, x2, y2))

        # 2. 使用持久化的 current_boxes 进行遍历和处理
        for idx, (x1, y1, x2, y2) in enumerate(current_boxes):
            # 裁剪 ROI 并做 Pose 估计
            roi = frame[y1:y2, x1:x2]
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            # 过滤掉未检测到姿态的框
            if not res.pose_landmarks:
                continue
            # 有效 pose，初始化或获取 tracker
            if idx not in trackers:
                trackers[idx] = init_target()
                trackers[idx]['id'] = idx
            data = trackers[idx]
            # 更新最后出现帧
            data['last_seen'] = frame_idx

            lm_list = res.pose_landmarks.landmark
            # 用像素归一化身体躯干高度：肩-髋的平均 y
            sy = lm_list[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y
            sy += lm_list[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y
            hy = lm_list[mp.solutions.pose.PoseLandmark.LEFT_HIP].y
            hy += lm_list[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y
            torso_norm = ((sy + hy) / 4.0)

            # 更新信号并计数
            fluct = process_signal(data, torso_norm)

            # 绘制骨架与跳绳计数
            draw_pose_and_count(frame, lm_list, x1, y1, x2, y2, data)

        # 清除长时间未更新的 trackers
        for tid, data in list(trackers.items()):
            if frame_idx - data['last_seen'] > MAX_MISSES:
                del trackers[tid]

        # 3. 生成多目标 Debug 画布，放在画面右侧
        num = len(trackers)
        if num > 0:
            canvas = np.zeros((h, BUFFER_LEN, 3), dtype=np.uint8)
            row_h = h // (4 * num)
            for tidx, data in trackers.items():
                for sidx, (buf, name) in enumerate([
                    (data['raw_buf'], 'Raw'),
                    (data['smooth_buf'], 'Smooth'),
                    (data['trend_buf'], 'Trend'),
                    (data['fluct_buf'], 'Fluct')
                ]):
                    y0 = (tidx * 4 + sidx) * row_h
                    y1 = y0 + row_h
                    arr = np.array(buf)
                    if len(arr) >= 2:
                        mn, mx = arr.min(), arr.max()
                        norm = (arr - mn) / (mx - mn) if mx > mn else np.full_like(arr, 0.5)
                        for i in range(1, len(norm)):
                            x_a, x_b = i - 1, i
                            y_a = int(y1 - norm[i - 1] * row_h)
                            y_b = int(y1 - norm[i] * row_h)
                            cv2.line(canvas, (x_a, y_a), (x_b, y_b), (200, 200, 200), 1)
                    cv2.putText(canvas, f"ID{tidx}-{name}",
                                (5, y0 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            full = cv2.hconcat([frame, canvas])
        else:
            full = frame

        cv2.imshow("Multi-Person JumpRope Debug", full)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
