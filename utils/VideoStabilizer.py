import cv2
import numpy as np


class VideoStabilizer:
    """
    基于 LK 光流 + 仿射估计的视频稳定器，每次调用 stabilize(frame) 都会把当前帧对齐到上一帧的参考坐标系中。

    算法流程：
      1. 首帧：在灰度图上检测一批角点，直接返回原图。
      2. 后续帧：
         a. 用 calcOpticalFlowPyrLK 跟踪上一帧角点到当前灰度图。
         b. 筛选状态为成功的点对 (pts0->pts1)。
         c. 点对数量足够时用 RANSAC 估计 2×3 仿射变换 M2x3，否则用单位矩阵。
         d. 将 M2x3 转成 3×3 累积矩阵并 warpAffine 到上一帧坐标系。
         e. 每隔 N 帧重新检测新角点，否则继续用跟踪得到的 pts1。
    """

    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: float = 30,
        reinit_interval: int = 30,
    ):
        """
        Args:
            max_corners:       最多检测多少个角点用于跟踪
            quality_level:     GoodFeaturesToTrack 的 qualityLevel 参数
            min_distance:      GoodFeaturesToTrack 的 minDistance 参数
            reinit_interval:   每隔多少帧重检测一次新角点
        """
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.reinit_interval = reinit_interval

        self.prev_gray = None        # 上一帧灰度图
        self.prev_pts = None         # 上一帧待跟踪角点
        self.transforms = []         # 累积的 3×3 同质仿射矩阵列表
        self.frame_count = 0         # 帧计数，用于控制何时重检测角点

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """
        对齐当前帧到上一帧坐标系中，输出补偿抖动之后的新帧。

        Args:
            frame: BGR 彩色图

        Returns:
            stabilized: BGR 彩色图，已做逆仿射对齐
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ----- 第 1 帧：初始化 -----
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance
            )
            # 第一帧无需对齐，累积一个单位矩阵
            I2x3 = np.array([[1, 0, 0],
                             [0, 1, 0]], dtype=np.float32)
            M3x3 = np.vstack([I2x3, [0, 0, 1]])
            self.transforms.append(M3x3)
            return frame

        # ----- 后续帧：跟踪 + 估计仿射 -----
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray,
            self.prev_pts, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        # 只保留跟踪成功的点对
        mask = (status.flatten() == 1)
        pts0 = self.prev_pts[mask].reshape(-1, 2)
        pts1 = curr_pts[mask].reshape(-1, 2)

        # 如果点对太少，直接用单位 2×3 矩阵
        if len(pts0) < 10:
            M2x3 = np.array([[1, 0, 0],
                             [0, 1, 0]], dtype=np.float32)
        else:
            M2x3, inliers = cv2.estimateAffine2D(
                pts0, pts1,
                method=cv2.RANSAC,
                ransacReprojThreshold=3,
                maxIters=200
            )
            if M2x3 is None:
                M2x3 = np.array([[1, 0, 0],
                                 [0, 1, 0]], dtype=np.float32)

        # 累积到 3×3 同质矩阵并保存
        M3x3 = np.vstack([M2x3, [0, 0, 1]]).astype(np.float32)
        self.transforms.append(M3x3)

        # 对当前帧做逆变换：将其对齐到上一帧坐标系
        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(
            frame,
            M2x3,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )

        # ----- 更新角点：每隔 reinit_interval 帧重检测，否则直接用刚跟踪到的 pts1 -----
        self.frame_count += 1
        next_gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
        if self.frame_count % self.reinit_interval == 0 or pts1.shape[0] < 10:
            # 重检测新角点
            self.prev_pts = cv2.goodFeaturesToTrack(
                next_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance
            )
        else:
            # 继续用刚跟踪得到的 pts1
            self.prev_pts = pts1.reshape(-1, 1, 2)

        # 更新上一帧灰度图
        self.prev_gray = next_gray

        return stabilized