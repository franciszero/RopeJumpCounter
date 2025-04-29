import cv2
import numpy as np


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
