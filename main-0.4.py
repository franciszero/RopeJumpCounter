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

from PoseDetection.features import FeaturePipeline
from utils.Differentiator import get_differentiator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(created)f | %(levelname)1.1s | %(message)s"
)
logger = logging.getLogger("JumpDebug")


class MultiRegionJumpDetector:
    """
    regions: list of region names, e.g. ["head","torso","legs"]
    """

    def __init__(self, min_interval=0.1):
        self.min_interval = min_interval
        self.prev_signs = {}  # 会根据第一次调用动态生成
        self.last_jump_time = 0.0
        self.count = 0
        self.differentiators = {}  # landmark_idx -> Differentiator

    """
    f_dict: {region: f_value}
    仅当所有 region 同时从负过零到正 且间隔足够时计数
    """

    def detect(self, speed_dict, frame_idx):
        now = time.time()
        # 动态为每个新的 idx 创建 Differentiator
        for idx in speed_dict:
            if idx not in self.differentiators:
                self.differentiators[idx] = get_differentiator()

        # 一阶差分已经给出 speed_dict，这里只示意如何复用符号跨零
        signs = {}
        for idx, v in speed_dict.items():
            signs[idx] = 1 if v > 0 else -1
        # 第一次调用 prev_signs 还没这个 idx，就初始化为 -1
        for idx in signs:
            self.prev_signs.setdefault(idx, -1)
        logger.debug(f"[DETECT][Frame {frame_idx}] signs={signs} prev_signs={self.prev_signs} "
                     f"last_jump={self.last_jump_time:.3f} count={self.count}")

        # 检测“所有可见 idx”是否都经历了 -1→+1
        crossed = [signs[idx] > 0 > self.prev_signs[idx] for idx in signs]

        if crossed and all(crossed):
            interval = now - self.last_jump_time
            if interval > self.min_interval:
                self.count += 1
                self.last_jump_time = now
                logger.info(f"[JUMP!] ++count -> {self.count}")

        # 更新状态
        for idx in signs:
            self.prev_signs[idx] = signs[idx]

        return self.count


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

    def render(self, frame, buffers, jump_count):
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
            buf = np.array(buffers[r])
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


class MainApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.cap_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 定义各个“区域”对应的关键点索引
        self.mp_pose = mp.solutions.pose
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
            # "legs": [
            #     self.mp_pose.PoseLandmark.LEFT_KNEE,
            #     self.mp_pose.PoseLandmark.RIGHT_KNEE,
            #     self.mp_pose.PoseLandmark.LEFT_ANKLE,
            #     self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            # ],
        }

        self.pipe = FeaturePipeline(self.cap, window_size=128)
        self.detector = MultiRegionJumpDetector(min_interval=0.1)
        self.renderer = DebugRenderer(
            frame_h=self.cap_h,
            buffer_len=128,
            regions=list(self.REGION_LANDMARKS.keys()),
            zoom=5.0,
            bar_ratio=0.2,
            time_zoom=3.0,
        )

        regions = list(self.REGION_LANDMARKS.keys())
        self.speed_bufs = {r: deque(maxlen=128) for r in regions}

    def run(self):
        frame_idx = 0
        while True:
            if not self.pipe.success_process_frame(frame_idx):
                break

            # 1) 全量可见性 mask
            vis = np.array([self.pipe.fs.rec[f'vis_{i}'] for i in range(33)])
            mask = vis > 0.5
            # 2) 全量速度 array
            vy = np.array([self.pipe.fs.rec[f'vy_{i}'] for i in range(33)])
            # 3) 掩码应用
            vy_masked = vy * mask.astype(float)
            # 4) 构造速度字典
            speed_dict = {i: float(vy_masked[i]) for i in range(33)}
            # Compute per-region average vertical speed
            region_speeds = {}
            for region, lm_list in self.REGION_LANDMARKS.items():
                # extract speeds for each landmark index in the region
                speeds = [speed_dict[lm.value] for lm in lm_list if lm.value in speed_dict]
                region_speeds[region] = sum(speeds) / len(speeds) if speeds else 0.0

            # 更新每个区域的速度缓冲区
            for r, v in region_speeds.items():
                self.speed_bufs[r].append(v)

            # 5) 调用 detect
            cnt = self.detector.detect(region_speeds, frame_idx)

            # 渲染并显示
            out = self.renderer.render(self.pipe.fs.raw_frame, self.speed_bufs, cnt)
            cv2.imshow('JumpCounter', out)
            # 按 Esc 键退出
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_idx += 1

        # 清理
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    MainApp().run()
