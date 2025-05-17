# features.py
import time
import math
from collections import deque
import numpy as np
import cv2

from PoseDetection.feature_mode import Feature, get_feature_mode
from utils.Perf import PerfStats
from utils.vision import PoseEstimator
from utils.VideoStabilizer import VideoStabilizer
from utils.Differentiator import get_differentiator
from utils.FrameSample import FrameSample


class FeaturePipeline:
    def __init__(self, cap, window_size):
        self.window_size = window_size
        self.fs = FrameSample(cap, self.window_size)

        self.stabilizer = VideoStabilizer()
        self.pose_est = PoseEstimator()
        self.diff = get_differentiator()
        self.dist_calc = DistanceCalculator()
        self.ang_calc = AngleCalculator()

        self.stats = PerfStats(window_size=10)

    def process_frame(self, frame, frame_idx):
        self.fs.raw_frame = frame
        self.fs.init_current_frame(frame_idx)
        stable = self.stabilizer.stabilize(self.fs.raw_frame)
        lm = self.pose_est.get_pose_landmarks(stable)

        mode = get_feature_mode()
        if Feature.RAW in mode:
            self.fs.compute_raw(lm)
        if Feature.RAW_PX in mode:
            self.fs.compute_raw_px(lm)
        if Feature.DIFF in mode:
            self.fs.compute_diff(self.diff)
        if Feature.SPATIAL in mode:
            self.fs.compute_spatial(lm, self.dist_calc, self.ang_calc)
        if Feature.WINDOW in mode:
            self.fs.windowed_features()


class DistanceCalculator:
    """计算若干关节点对之间的欧氏距离。"""

    def __init__(self):
        """
        pairs: list of (idx_a, idx_b) 元组，索引对应 landmarks 列表的序号
        """

        self.pairs = [(24, 26), (26, 28), (11, 13), (13, 15)]  # 臀-膝，膝-踝，肩-肘，肘-腕

    def compute(self, landmarks) -> list:
        """返回每对关节点的三维距离。"""
        dists = []
        for a, b in self.pairs:
            pa, pb = landmarks[a], landmarks[b]
            dx = pa.x - pb.x
            dy = pa.y - pb.y
            dz = pa.z - pb.z
            dists.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        return dists


class AngleCalculator:
    """计算若干三点定义的关节夹角。"""

    def __init__(self):
        """
        triplets: list of (idx_a, idx_b, idx_c) ，以 b 为顶点算 ∠ABC
        """
        self.triplets = [(24, 26, 28), (11, 13, 15), (23, 11, 13)]  # 髋-膝-踝，肩-肘-腕，躯干倾斜

    def compute(self, landmarks) -> list:
        """返回每组三点夹角（度）。"""
        angles = []
        for a, b, c in self.triplets:
            p1, p2, p3 = landmarks[a], landmarks[b], landmarks[c]
            # 向量 p2->p1, p2->p3
            v1 = (p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)
            v2 = (p3.x - p2.x, p3.y - p2.y, p3.z - p2.z)
            dot = sum(v1[i] * v2[i] for i in range(3))
            mag1 = math.sqrt(sum(v1[i] * v1[i] for i in range(3)))
            mag2 = math.sqrt(sum(v2[i] * v2[i] for i in range(3)))
            angle = math.degrees(math.acos(dot / (mag1 * mag2))) if mag1 * mag2 > 0 else 0.0
            angles.append(angle)
        return angles
