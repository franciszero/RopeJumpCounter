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
    """
    趋势分离滤波器：将相对速度信号分解为“长期趋势”（低频）和“高频波动”两部分。

    采用：
      1. 指数加权移动平均（EWMA）提取平滑信号 s，
      2. 对 s 再做固定窗口的移动平均得到趋势 t，
      3. 高频分量 f = s - t。

    参数：
      buffer_len (int): 各历史缓存最大长度（以帧数计），决定了能回溯的最长历史。
      alpha (float): EWMA 的平滑系数 ∈ (0,1]，α 越大，s 对最新测量越敏感。
      trend_win (int): 计算趋势 t 时，s 使用的窗口大小（帧数）。
      baseline (int): 前 baseline 帧内强制返回 0，主要用于初始化各缓存。

    属性：
      raw_buf (deque): 原始相对速度缓冲区。
      smooth_buf (deque): EWMA 平滑后的速度缓冲区。
      trend_buf (deque): 移动平均趋势缓冲区。
      fluct_buf (deque): 高频波动 f = s - t 的缓冲区，用于后续绘图或检测。

    用法：
      每帧调用 update(rel_speed, frame_idx)，返回当前的高频成分 f。
    """

    def __init__(self, buffer_len=600, alpha=0.2, trend_win=64, baseline=0):
        self.alpha = alpha
        self.trend_win = trend_win
        self.baseline = baseline

        # 各缓冲区用于存放历史数据，长度上限为 buffer_len
        self.raw_buf = deque(maxlen=buffer_len)
        self.smooth_buf = deque(maxlen=buffer_len)
        self.trend_buf = deque(maxlen=buffer_len)
        self.fluct_buf = deque(maxlen=buffer_len)

    def update(self, rel_speed, idx):
        """
        更新滤波器并返回高频波动 f。

        Args:
          rel_speed (float): 当前帧的“去背景”后相对速度值（body_dy - bg_dy_norm）。
          idx (int): 当前帧序号，用于跳过前 baseline 帧的初始化。

        Returns:
          float: 高频分量 f = s - t。
            - s: 对 rel_speed 做指数平滑后的值；
            - t: 对最近 trend_win 帧的 s 做简单平均得到的趋势；
            - f: s 与 t 之差，表示“短期波动”。

        逻辑：
          - 当 idx <= baseline 时，向所有缓存添加 0 并直接返回 0，保证缓冲区被填满；
          - 否则：
              1) 将 rel_speed 加入 raw_buf；
              2) 计算 s = α * rel_speed + (1 - α) * last_s；
              3) 从 smooth_buf 取最近 trend_win 个 s 计算 t = mean(...)；
              4) f = s - t；
              5) 更新各缓存并返回 f。
        """
        # 初始化阶段：填充零值，直到 baseline
        if idx <= self.baseline:
            for buf in (self.raw_buf, self.smooth_buf, self.trend_buf, self.fluct_buf):
                buf.append(0.0)
            return 0.0

        import numpy as np

        # 1. 原始速度入队
        self.raw_buf.append(rel_speed)

        # 2. 指数平滑：s = α·当前速度 + (1−α)·上一次平滑值
        last_s = self.smooth_buf[-1] if self.smooth_buf else 0.0
        s = self.alpha * rel_speed + (1 - self.alpha) * last_s

        # 3. 移动平均趋势：t = 最近 trend_win 帧的 s 的平均
        recent_s = list(self.smooth_buf)[-self.trend_win:]
        t = np.mean(recent_s) if recent_s else 0.0

        # 4. 高频分量 = 平滑值 − 趋势
        f = s - t

        # 5. 更新缓存
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
    def __init__(self, frame_h, buffer_len, regions, zoom=5.0, bar_ratio=0.2, time_zoom=2.0):
        """
        frame_h: 原视频帧高度
        buffer_len: 时间序列长度（像素宽度）
        regions: ["head","torso",...]
        zoom: 波形放大系数
        bar_ratio: 底部柱状图区占整个画布高度的比例
        """
        self.frame_h = frame_h
        self.buffer_len = buffer_len
        self.regions = regions
        self.zoom = zoom
        # 新增：横向每帧占用像素数
        self.time_zoom = time_zoom

        # 持久化 jump history
        self.jump_buf = deque(maxlen=buffer_len)
        self.prev_cnt = 0

        # 计算每个区域和柱状区的高度
        total_h = frame_h
        # 保留 (1-bar_ratio) 给波形区域，均分给每条曲线
        self.region_h = int((1 - bar_ratio) * total_h / len(regions))
        # 底部柱状区高度
        self.bar_h = int(bar_ratio * total_h)

    def render(self, frame, filters, jump_count):
        # —— 在视频画面左上角，大字显示跳数 ——
        cv2.putText(
            frame,
            f"Jumps: {jump_count}",
            (20, 60),  # 距离左边 20px，距离顶边 60px
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,  # 字体放大 2.5 倍
            (0, 255, 255),  # 黄色
            5  # 粗一点
        )

        # —— 1. 更新跳绳事件历史 ——
        if jump_count > self.prev_cnt:
            # 本帧检测到新跳跃
            self.jump_buf.append(jump_count)
        else:
            self.jump_buf.append(0)
        self.prev_cnt = jump_count

        # —— 2. 新建画布 ——
        H = self.region_h * len(self.regions) + self.bar_h
        # 横向总宽 = buffer_len * time_zoom
        W = int(self.buffer_len * self.time_zoom)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # —— 3. 绘制每条波形曲线 ——
        for i, r in enumerate(self.regions):
            buf = np.array(filters[r].fluct_buf)
            if buf.size < 2:
                continue
            # zoom in
            buf_z = buf * self.zoom
            mn, mx = buf_z.min(), buf_z.max()
            norm = (buf_z - mn) / (mx - mn) if mx > mn else np.full_like(buf_z, 0.5)

            y0 = i * self.region_h
            y1 = y0 + self.region_h
            # 横向拉伸，每帧占 time_zoom 像素
            pts = [
                (int(x * self.time_zoom), int(y1 - norm[x] * self.region_h))
                for x in range(len(norm))
            ]
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                cv2.line(canvas, (x0, y0), (x1, y1), (200, 200, 200), 1)

            cv2.putText(
                canvas, r,
                (5, y0 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )

        # —— 4. 底部柱状图 ——
        base_y = self.region_h * len(self.regions)
        for idx, val in enumerate(self.jump_buf):
            x_pix = int(idx * self.time_zoom)
            if val:
                cv2.line(
                    canvas,
                    (x_pix, base_y),
                    (x_pix, base_y + self.bar_h),
                    (200, 200, 0),
                    1
                )
                cv2.putText(
                    canvas,
                    str(val),
                    (x_pix, base_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 255, 255),
                    1
                )

        # —— 5. 拼到原帧右侧 ——
        out = cv2.hconcat([frame, canvas])
        return out


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

        buf_len = 100
        # 组件
        self.pose = PoseEstimator()
        self.bg = BackgroundTracker()
        # 为每个 region 各自创建一个趋势滤波器
        # 对应把 alpha 从 0.2 提升到 0.5，把 trend_win 从 64 缩短到 32
        self.filters = {
            r: TrendFilter(buffer_len=buf_len, alpha=0.16, trend_win=64, baseline=50)
            for r in regions
        }
        self.detector = MultiRegionJumpDetector(regions)
        self.renderer = DebugRenderer(
            frame_h=h,
            buffer_len=buf_len,
            regions=regions,
            zoom=2.0,  # 波形放大倍率，可调
            bar_ratio=0.2,  # 底部柱状图占画布高度比例
            time_zoom=10.0  # 每帧 3px
        )

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
