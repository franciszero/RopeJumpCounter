import cv2
import numpy as np


class VideoStabilizer:
    """
    基于 LK 光流 + 仿射估计的视频稳定器。
    每次调用 stabilize(frame) 都会把当前帧对齐到上一帧的参考坐标系中。
    """

    def __init__(self, max_corners=200, quality_level=0.01, min_distance=30):
        # 用于跟踪的角点数目和质量
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance

        # 上一帧的灰度图和角点
        self.prev_gray = None
        self.prev_pts = None

        # 仿射矩阵累积结果（可选）
        self.transforms = []

    def stabilize(self, frame):
        """
        输入 BGR 彩色图，输出补偿抖动之后的新帧。
        """
        # 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 第一帧：初始化角点
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance
            )
            # 第一帧无需变换
            self.transforms.append(np.eye(3, dtype=np.float32))
            return frame

        # 计算光流：上一帧的点 -> 当前帧的新位置
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray,
            self.prev_pts, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        # 只保留跟踪成功的点对
        mask = status.flatten() == 1
        pts0 = self.prev_pts[mask].reshape(-1, 2)
        pts1 = curr_pts[mask].reshape(-1, 2)

        # 如果点对太少，跳过补偿
        if len(pts0) < 10:
            M = np.eye(3, dtype=np.float32)
        else:
            # 用 RANSAC 估计仿射变换
            M2x3, inliers = cv2.estimateAffine2D(
                pts0, pts1,
                method=cv2.RANSAC,
                ransacReprojThreshold=3,
                maxIters=200
            )
            if M2x3 is None:
                M = np.eye(3, dtype=np.float32)
            else:
                # 转成 3×3 同质矩阵
                M = np.vstack([M2x3, [0, 0, 1]]).astype(np.float32)

        # 累积变换（可用于平滑或调试）
        self.transforms.append(M)

        # 对当前帧做逆变换（将当前帧对齐到上一帧坐标系）
        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(
            frame,
            M2x3,  # 2×3 部分就行
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )

        # 更新上一帧信息
        self.prev_gray = gray
        # 用对齐后的帧再找新的角点跟踪，或者继续用旧的 pts1
        self.prev_pts = cv2.goodFeaturesToTrack(
            cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY),
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance
        )

        return stabilized
