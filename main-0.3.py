import cv2
import time
import numpy as np
from collections import deque
import mediapipe as mp
import tensorflow as tf


class PoseEstimator:
    def __init__(self,
                 model_complexity=0,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # 定义各个“区域”对应的关键点索引
        self.REGION_LANDMARKS = {
            "head": [
                self.mp_pose.PoseLandmark.NOSE,
                self.mp_pose.PoseLandmark.LEFT_EYE,
                self.mp_pose.PoseLandmark.RIGHT_EYE,
                self.mp_pose.PoseLandmark.LEFT_EAR,
                self.mp_pose.PoseLandmark.RIGHT_EAR,
            ],
            "torso": [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
            ],
            "legs": [
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            ],
        }

    def estimate(self, frame):
        """
        输入 BGR 图像，输出 (pose_landmarks, dict of region→height)
        region heights are normalized y in [0,1]
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return None, {}
        lm = res.pose_landmarks.landmark

        # 计算每个区域的平均归一化高度
        heights = {}
        for region, idxs in self.REGION_LANDMARKS.items():
            ys = [lm[i].y for i in idxs]
            heights[region] = sum(ys) / len(ys)

        return res.pose_landmarks, heights


# =========================
# 2. BackgroundTracker：LK 光流背景补偿
# =========================
class BackgroundTracker:
    def __init__(self, max_pts=200):
        self.max_pts = max_pts
        self.prev_gray = None
        self.bg_pts = None

    def compensate(self, gray):
        """
        返回当前帧背景垂直归一化速度 bg_dy_norm
        """
        h, _ = gray.shape
        if self.prev_gray is None:
            self.bg_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_pts,
                qualityLevel=0.01, minDistance=10
            )
            self.prev_gray = gray.copy()
            return 0.0

        new_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray,
            self.bg_pts, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        mask = (st.flatten() == 1)
        if mask.any():
            p0 = self.bg_pts[mask].reshape(-1, 2)
            p1 = new_pts[mask].reshape(-1, 2)
            dy = np.median(p1[:, 1] - p0[:, 1]) / h
            self.bg_pts = p1.reshape(-1, 1, 2)
            self.prev_gray = gray.copy()
            return dy
        else:
            self.prev_gray = gray.copy()
            return 0.0


# =========================
# 3. TrendFilter：指数平滑 + 移动平均趋势分离
# =========================
class TrendFilter:
    def __init__(self, buffer_len=320, alpha=0.2, trend_win=64, baseline=150):
        self.alpha = alpha
        self.trend_win = trend_win
        self.baseline = baseline
        self.raw_buf = deque(maxlen=buffer_len)
        self.smooth_buf = deque(maxlen=buffer_len)
        self.trend_buf = deque(maxlen=buffer_len)
        self.fluct_buf = deque(maxlen=buffer_len)

    def update(self, rel_speed, idx):
        """
        输入去背景后的相对速度 rel_speed 与帧号 idx
        返回高频波动 f
        """
        if idx <= self.baseline:
            for buf in (self.raw_buf,
                        self.smooth_buf,
                        self.trend_buf,
                        self.fluct_buf):
                buf.append(0.0)
            return 0.0

        # 原始速度
        self.raw_buf.append(rel_speed)
        # 指数平滑
        last_s = self.smooth_buf[-1]
        s = self.alpha * rel_speed + (1 - self.alpha) * last_s
        # 移动平均趋势
        t = np.mean(list(self.smooth_buf)[-self.trend_win:])
        # 高频分量
        f = s - t

        # 更新缓存
        self.smooth_buf.append(s)
        self.trend_buf.append(t)
        self.fluct_buf.append(f)
        return f


# =========================
# 4. MultiRegionJumpDetector：多区域同相位跳跃检测
# =========================
class MultiRegionJumpDetector:
    def __init__(self, regions, min_interval=0.3):
        """
        regions: list of region names, e.g. ["head","torso","legs"]
        """
        self.regions = regions
        self.min_interval = min_interval
        self.prev_signs = {r: -1 for r in regions}
        self.last_jump_time = 0.0
        self.count = 0

    def detect(self, f_dict):
        """
        f_dict: {region: f_value}
        仅当所有 region 同时从负过零到正 且间隔足够时计数
        """
        now = time.time()
        signs = {r: (1 if f_dict[r] > 0 else -1) for r in self.regions}

        # 判断所有区域是否都负→正
        if all(signs[r] > 0 and self.prev_signs[r] < 0 for r in self.regions):
            if (now - self.last_jump_time) > self.min_interval:
                self.count += 1
                self.last_jump_time = now

        self.prev_signs = signs
        return self.count


# =========================
# 5. DebugRenderer：画三条波动曲线 + 跳数
# =========================
class DebugRenderer:
    def __init__(self, frame_h, buffer_len, regions):
        self.frame_h = frame_h
        self.buffer_len = buffer_len
        self.regions = regions

    def render(self, frame, filters, jump_count):
        """
        filters: dict region→TrendFilter
        """
        h, w = frame.shape[:2]
        canvas = np.zeros((h, self.buffer_len, 3), np.uint8)
        row_h = h // len(self.regions)

        for i, r in enumerate(self.regions):
            buf = filters[r].fluct_buf
            arr = np.array(buf)
            y0, y1 = i * row_h, (i + 1) * row_h

            if len(arr) >= 2:
                mn, mx = arr.min(), arr.max()
                norm = (arr - mn) / (mx - mn) if mx > mn else np.full_like(arr, 0.5)
                pts = [(x, int(y1 - norm[x] * row_h)) for x in range(len(norm))]
                for p0, p1 in zip(pts, pts[1:]):
                    cv2.line(canvas, p0, p1, (200, 200, 200), 1)
            cv2.putText(canvas, r, (5, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 跳数
        cv2.putText(frame, f"Jumps: {jump_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 4)
        return cv2.hconcat([frame, canvas])


# =========================
# 6. MainApp：串联所有组件
# =========================
class MainApp:
    def __init__(self, regions=None):
        if regions is None:
            regions = ["head", "torso"]  # , "legs"]
        self.cap = cv2.VideoCapture(0)
        _, tmp = self.cap.read()
        h, _ = tmp.shape[:2]

        # 组件
        self.pose = PoseEstimator()
        # drawing utils and pose connections from mp alias
        self.drawing_utils = mp.solutions.drawing_utils
        self.pose_connections = mp.solutions.pose.POSE_CONNECTIONS
        self.bg = BackgroundTracker()
        # 为每个 region 各自创建一个趋势滤波器
        self.filters = {r: TrendFilter() for r in regions}

        # Load the trained LSTM model (.h5) and prepare sequence buffer
        self.lstm_model = tf.keras.models.load_model("./PoseDetection/models/crnn_jump_classifier.h5")
        # Determine input window size and feature dim from model
        _, W, D = self.lstm_model.input_shape
        from collections import deque
        self.seq_buffer = deque(maxlen=W)  # will hold W frames of raw landmarks

        self.detector = MultiRegionJumpDetector(regions)
        self.renderer = DebugRenderer(frame_h=h,
                                      buffer_len=self.filters[regions[0]].raw_buf.maxlen,
                                      regions=regions)

        # 用于计算相对速度
        self.prev_heights = {r: None for r in regions}

    def run(self):
        frame_idx = 0
        regions = list(self.filters.keys())
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame_idx += 1

            jump_prob = 0.0

            # 1) 姿势估计 → 各区域高度字典
            lm, heights = self.pose.estimate(frame)
            # 实时在画面上叠加火柴人骨架
            if lm:
                self.drawing_utils.draw_landmarks(
                    frame, lm, self.pose_connections)
            if not heights:
                heights = {r: (self.prev_heights[r] or 0.5) for r in self.prev_heights}

            # 2) 背景补偿
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bg_dy_norm = self.bg.compensate(gray)

            # 3) Extract full landmark vector for deep model
            # If pose landmarks detected, flatten all (x,y) coords
            if lm:
                lm_pts = lm.landmark
                vec = np.array([coord for pt in lm_pts for coord in (pt.x, pt.y)], dtype=np.float32)
                self.seq_buffer.append(vec)
            else:
                # skip this frame if no landmarks
                continue

            # 4) Deep LSTM model inference on raw landmark sequences
            count = self.detector.count
            if len(self.seq_buffer) == self.seq_buffer.maxlen:
                seq = np.stack(self.seq_buffer, axis=0)[None, ...]  # shape (1, W, D)
                probs = self.lstm_model.predict(seq, verbose=0)[0]
                # assume probs = [non_jump_prob, jump_prob]
                jump_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                # count once when window fills
                now = time.time()
                if jump_prob > 0.5 and (now - self.detector.last_jump_time) > self.detector.min_interval:
                    count += 1
                    self.detector.last_jump_time = now
            self.detector.count = count

            # 在右上角大字显示当前跳绳状态
            status = "JUMPING" if (len(self.seq_buffer) == self.seq_buffer.maxlen and jump_prob > 0.5) else "NOT JUMPING"
            # 选择大字号和高对比颜色（黄色）
            cv2.putText(
                frame, status,
                (frame.shape[1] - 350, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 255, 255), 4, cv2.LINE_AA)

            # 5) 渲染并展示
            output = self.renderer.render(frame, self.filters, count)
            cv2.imshow("Multi-Region JumpRope Debug", output)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    MainApp().run()
