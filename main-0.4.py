"""
跳绳计数器主程序 (面向对象版)
版本：0.3.11

功能：
- 面向对象重构：PoseEstimator, BackgroundTracker, TrendFilter, MultiRegionJumpDetector, DebugRenderer, MainApp
- 区域高度计算：支持 head, torso, legs 区域高度提取
- 背景补偿：LK 光流消除摄像头抖动
- 趋势分离：指数平滑 + 移动平均分离高频波动
- 多区域同相位检测：同时监测多条波动的负→正过零
- 可配置区域列表：MainApp 可传入不同区域组合
- 可视化调试：在摄像头画面下方绘制每个区域高频波动时间序列；左上角高亮跳数
- 支持动态调整跳数字体大小与颜色

更新日志：
0.3.0  - 初始 OOP 重构版本，实现 0.2 核心跳绳管线 (背景补偿+趋势分解+零交叉+调试 UI)
0.3.1  - 增加多区域支持 (head, torso, legs) 及对应滤波器
0.3.2  - 集成 MultiRegionJumpDetector，实现多区域同相位跳跃检测
0.3.3  - 支持通过构造函数配置区域列表
0.3.4  - 优化调试 UI：增大跳数文本字体、修改文本颜色为黄色
0.3.5  - 修复相对速度计算逻辑，使用 prev_torso_y 替换错误引用
0.3.6  - 代码清理及注释增强
0.3.11 - 最终迭代，完善文档与版本标记
"""

import cv2
import time
import numpy as np
from collections import deque
import mediapipe as mp
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(created)f | %(levelname)1.1s | %(message)s"
)
logger = logging.getLogger("JumpDebug")


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

    """
    返回当前帧背景垂直归一化速度 bg_dy_norm
    """

    def compensate(self, gray):
        h, _ = gray.shape
        if self.prev_gray is None:
            self.bg_pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_pts, qualityLevel=0.01, minDistance=10)
            self.prev_gray = gray.copy()
            return 0.0

        new_pts, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray,
                                                  self.bg_pts, np.zeros(self.bg_pts.shape),
                                                  winSize=(15, 15), maxLevel=2,
                                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
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

    """
    输入去背景后的相对速度 rel_speed 与帧号 idx
    返回高频波动 f
    """

    def update(self, rel_speed, idx):
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
    """
    regions: list of region names, e.g. ["head","torso","legs"]
    """

    def __init__(self, regions, min_interval=0.1):
        self.regions = regions
        self.min_interval = min_interval
        self.prev_signs = {r: -1 for r in regions}
        self.last_jump_time = 0.0
        self.count = 0

    """
    f_dict: {region: f_value}
    仅当所有 region 同时从负过零到正 且间隔足够时计数
    """

    def detect(self, f_dict, frame_idx):
        now = time.time()
        signs = {r: (1 if f_dict[r] > 0 else -1) for r in self.regions}
        logger.debug(f"[DETECT][Frame {frame_idx}] signs={signs} prev_signs={self.prev_signs} "
                     f"last_jump={self.last_jump_time:.3f} count={self.count}")

        # 判断负→正跨零
        crossed = [signs[r] > 0 > self.prev_signs[r] for r in self.regions]
        if all(crossed):
            interval = now - self.last_jump_time
            logger.debug(f"[DETECT] all regions crossed_up={crossed}, interval={interval:.3f}s")
            if interval > self.min_interval:
                self.count += 1
                self.last_jump_time = now
                logger.info(f"[JUMP!] ++count -> {self.count} (interval ok)")
            else:
                logger.debug(f"[SKIP] interval {interval:.3f}s < min_interval {self.min_interval}s")
        else:
            # 哪些区域没跨？
            failed = [r for r, ok in zip(self.regions, crossed) if not ok]
            logger.debug(f"[SKIP] not all crossed, failed regions={failed}")

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

        # 绘制跳绳计数，自动调整位置以确保完整显示
        text = f"Jumps: {jump_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4
        thickness = 15
        # 获取文本尺寸，避免超出画面
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = 10
        y = text_height + 10  # 将文本基线设置在高度 text_height + 10 处，确保完整显示
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 255), thickness)
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
        self.bg = BackgroundTracker()
        # 为每个 region 各自创建一个趋势滤波器
        self.filters = {r: TrendFilter() for r in regions}
        self.detector = MultiRegionJumpDetector(regions)
        self.renderer = DebugRenderer(frame_h=h,
                                      buffer_len=self.filters[regions[0]].raw_buf.maxlen,
                                      regions=regions)

        # 用于计算相对速度
        self.prev_heights = {r: None for r in regions}

    def run(self):
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame_idx += 1

            # 1) 姿势估计 → 各区域高度字典
            lm, heights = self.pose.estimate(frame)
            if not heights:
                logger.debug(
                    f"[Frame {frame_idx}] PoseEstimator MISS. Fallback heights from prev: {self.prev_heights}")
                heights = {r: (self.prev_heights[r] or 0.5) for r in self.prev_heights}
            else:
                logger.debug(f"[Frame {frame_idx}] PoseEstimator OK. Heights: {heights}")
            if not heights:
                heights = {r: (self.prev_heights[r] or 0.5) for r in self.prev_heights}

            # 2) 背景补偿
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bg_dy_norm = self.bg.compensate(gray)
            logger.debug(f"[Frame {frame_idx}] Background dy_norm: {bg_dy_norm:.4f}")

            # 3) 计算去背景后的相对速度 f for each region
            f_vals = {}
            for r, filt in self.filters.items():
                prev_h = self.prev_heights[r]
                body_dy = 0.0 if prev_h is None else (heights[r] - prev_h)
                self.prev_heights[r] = heights[r]

                rel_speed = body_dy - bg_dy_norm
                f = filt.update(rel_speed, frame_idx)
                f_vals[r] = f
                logger.debug(f"[Frame {frame_idx}] Region '{r}': prev_h={prev_h} curr_h={heights[r]:.4f} "
                             f"body_dy={body_dy:.4f} rel_speed={rel_speed:.4f} f={f:.4f}")

                # curr_h = heights[r]
                # if prev_h is None:
                #     body_dy = 0.0
                # else:
                #     body_dy = curr_h - prev_h
                # self.prev_heights[r] = curr_h
                #
                # rel_speed = body_dy - bg_dy_norm
                # f_vals[r] = filt.update(rel_speed, frame_idx)

            # 4) 多区域跳跃检测
            count = self.detector.detect(f_vals, frame_idx)

            # 5) 渲染并展示
            output = self.renderer.render(frame, self.filters, count)
            cv2.imshow("Multi-Region JumpRope Debug", output)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    MainApp().run()
