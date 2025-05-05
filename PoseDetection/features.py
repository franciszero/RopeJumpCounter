# features.py
import time

import math
from collections import deque
import numpy as np
import cv2
from utils.vision import PoseEstimator
from utils.VideoStabilizer import VideoStabilizer
from utils.Differentiator import get_differentiator


class FeaturePipeline:
    def __init__(self, window_size):
        self.stabilizer = VideoStabilizer()
        self.pose_est = PoseEstimator()
        self.diff = get_differentiator()
        self.dist_calc = DistanceCalculator()
        self.ang_calc = AngleCalculator()
        self.window_size = window_size

    def process_frame(self, cap, frame_idx):
        fs = FrameSample(cap, frame_idx, self.window_size)
        if not fs.ret:
            return None
        # 共享实例
        stable = self.stabilizer.stabilize(fs.raw_frame)
        lm = self.pose_est.get_pose_landmarks(stable)
        fs.compute_raw(lm)
        fs.compute_diff(self.diff)
        fs.compute_spatial(lm, self.dist_calc, self.ang_calc)
        fs.windowed_features()
        return fs.rec


class FrameSample:
    """
    Encapsulates all per-frame data and feature computations for a single video frame.

    This class stores raw image data, pose landmarks, and computes various features such as
    normalized and pixel coordinates, velocity, acceleration, distances between key joints,
    and joint angles. It also supports maintaining a sliding window buffer of features for
    temporal models.
    """

    def __init__(self, cap, frame_idx, window_size: int = 1):
        """
        Initialize a FrameSample instance.

        Args:
            cap
            window_size (int, optional): Size of the sliding window for temporal features.
                                         Defaults to 1 (no windowing).
        """
        self.ret, self.raw_frame = cap.read()  # Original BGR frame
        if not self.ret:
            return
        self.h, self.w = self.raw_frame.shape[:2]

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if timestamp_ms == 0.0:
            self.timestamp = time.time()
        else:
            self.timestamp = timestamp_ms / 1000.0
        self.frame_idx = frame_idx
        # 拼接到一条记录
        self.rec = {
            'frame': self.frame_idx,
            'timestamp': self.timestamp
        }

        self.window_size = window_size  # Window size for temporal feature buffering

        # To be assigned later during processing:
        self.heights = {}  # Dictionary to hold height measurements if needed

        # Feature buffers initialized as empty lists:
        self.raw_norm = []  # Normalized landmark coordinates + visibility
        self.raw_px = []  # Landmark coordinates in pixel space
        self.vel = []  # Velocity features (first-order temporal differences)
        self.acc = []  # Acceleration features (second-order temporal differences)
        self.dists = []  # Distances between selected joint pairs
        self.angs = []  # Angles between selected joint triplets

        # Sliding window buffer to hold sequences of feature vectors for models that require temporal context
        self.buffer = deque(maxlen=self.window_size)

        # 6) 拼接到一条记录
        self.rec = {
            'frame': frame_idx,
            'timestamp': self.timestamp
        }

    def compute_raw(self, lm):
        """
        Compute normalized and pixel coordinates of landmarks.

        This method populates self.raw_norm and self.raw_px based on the current landmarks.
        If landmarks are missing, fills features with zeros.
        """
        if not lm:
            # No landmarks detected: fill with zeros
            self.raw_norm = [0.0] * 33 * 4  # 33 landmarks * 4 values each (x,y,z,visibility)
            self.raw_px = [0.0] * 33 * 2  # 33 landmarks * 2 values each (x_px, y_px)
        else:
            # Extract normalized coordinates and visibility
            self.raw_norm = []
            for lm in lm.landmark:
                self.raw_norm.extend([lm.x, lm.y, lm.z, lm.visibility])
            # Determine frame size for pixel conversion
            self.raw_px = []
            for lm in lm.landmark:
                # Convert normalized coords to pixel coords
                self.raw_px.extend([lm.x * self.w, lm.y * self.h])
        # raw x,y,z,vis
        for i in range(33):
            self.rec[f'x_{i}'] = self.raw_norm[4 * i]
            self.rec[f'y_{i}'] = self.raw_norm[4 * i + 1]
            self.rec[f'z_{i}'] = self.raw_norm[4 * i + 2]
            self.rec[f'vis_{i}'] = self.raw_norm[4 * i + 3]
        # 像素坐标特征
        for i in range(33):
            self.rec[f'x_px_{i}'] = self.raw_px[2 * i]
            self.rec[f'y_px_{i}'] = self.raw_px[2 * i + 1]

    def compute_diff(self, diff):
        """
        Compute velocity and acceleration features using a Differentiator instance.

        Args:
            diff (Differentiator): Instance to compute velocity and acceleration.
        """
        # Compute velocity and acceleration given current and previous raw_norm and timestamps
        self.vel, self.acc = diff.compute(self.raw_norm, self.timestamp)
        # velocity features
        for i in range(33):
            self.rec[f'vx_{i}'] = self.vel[4 * i]
            self.rec[f'vy_{i}'] = self.vel[4 * i + 1]
            self.rec[f'vz_{i}'] = self.vel[4 * i + 2]
            self.rec[f'vvis_{i}'] = self.vel[4 * i + 3]
        # acceleration features
        for i in range(33):
            self.rec[f'ax_{i}'] = self.acc[4 * i]
            self.rec[f'ay_{i}'] = self.acc[4 * i + 1]
            self.rec[f'az_{i}'] = self.acc[4 * i + 2]
            self.rec[f'avis_{i}'] = self.acc[4 * i + 3]

    def compute_spatial(self, lm, dist_calc, ang_calc):
        """
        Compute spatial features: distances and angles of key joints.

        Args:
            lm
            dist_calc (DistanceCalculator): Computes distances between joint pairs.
            ang_calc (AngleCalculator): Computes angles between joint triplets.
        """
        if not lm:
            # Missing landmarks: fill distances and angles with zeros
            self.dists = [0.0] * len(dist_calc.pairs)
            self.angs = [0.0] * len(ang_calc.triplets)
        else:
            # Compute distances and angles from landmarks
            self.dists = dist_calc.compute(lm.landmark)
            self.angs = ang_calc.compute(lm.landmark)
        # distances
        for idx, (a, b) in enumerate(dist_calc.pairs):
            self.rec[f'dist_{a}_{b}'] = self.dists[idx]
        # angles
        for idx, (a, b, c) in enumerate(ang_calc.triplets):
            self.rec[f'angle_{a}_{b}_{c}'] = self.angs[idx]

    def windowed_features(self):
        """
        Returns a windowed tensor of features suitable for sequence models.

        The output shape depends on window_size:
          - If window_size > 1, returns a tensor shaped (1, window_size, feature_dim),
            where feature_dim is the length of concatenated features.
            If the buffer is not yet full, returns zeros.
          - If window_size == 1, returns a tensor shaped (1, 1, feature_dim) for a single frame.

        Returns:
            np.ndarray: Feature tensor with batch dimension 1.
        """
        # Concatenate all feature components into a single feature vector
        feat = self.raw_norm + self.raw_px + self.vel + self.acc + self.dists + self.angs

        if self.window_size > 1:
            # Append current feature vector to sliding window buffer
            self.buffer.append(feat)
            if len(self.buffer) < self.window_size:
                # Not enough frames collected yet: return zero tensor with correct shape
                return np.zeros((1, self.window_size, len(feat)), dtype=np.float32)
            # Stack buffered features along time axis and add batch dimension
            return np.stack(self.buffer, axis=0)[np.newaxis, ...]
        else:
            # Single frame: add batch and time dimensions for consistent shape
            return np.array(feat, dtype=np.float32)[np.newaxis, np.newaxis, :]


class PoseFrame:
    """封装单帧的原始关键点数据和帧索引/时间戳。"""

    def __init__(self, frame_idx: int, timestamp: float, landmarks, frame_size=None):
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.diff = get_differentiator()
        self.dist_calc = DistanceCalculator()
        self.ang_calc = AngleCalculator()
        self.window_size = window_size
        self.buffer = deque(maxlen=self.window_size)
        # 将原始数据平坦化为 [x0,y0,z0,vis0, ..., x32,y32,z32,vis32]
        self.raw = []
        for lm in landmarks:
            self.raw.extend([lm.x, lm.y, lm.z, lm.visibility])
        if frame_size is not None:
            h, w = frame_size
            self.raw_px = []
            for lm in landmarks:
                self.raw_px.extend([lm.x * w, lm.y * h])

    def extract(self, frame_idx, timestamp, lm, frame_size):
        """
        Given landmarks and frame metadata, returns a tuple (feat_vector, skip_model)
        - feat_vector: 469-dim vector ready for inference
        - skip_model: boolean, True if landmarks missing (will skip inference)
        """
        height, width = frame_size
        if lm is None:
            # no landmarks: zero vector, skip inference
            feat = [0.0] * 469
            return np.array(feat, dtype=np.float32), True

        # assemble raw, raw_px
        raw = self.raw  # 132 dims
        raw_px = self.raw_px  # 66 dims
        # velocity & acceleration
        vel, acc = self.diff.compute(raw, timestamp)
        # distances & angles
        dists = self.dist_calc.compute(lm.landmark)
        angs = self.ang_calc.compute(lm.landmark)

        feat_full = raw + raw_px + vel + acc + dists + angs  # total 469 dims

        if self.window_size > 1:
            self.buffer.append(feat_full)
            if len(self.buffer) < self.window_size:
                # not enough history
                return np.zeros_like(feat_full), True
            # stack into window
            windowed = np.stack(self.buffer, axis=0)[np.newaxis, ...]
            return windowed, False
        else:
            # single-frame format: [1,1,469]
            single = np.array(feat_full, dtype=np.float32)[np.newaxis, np.newaxis, :]
            return single, False


# class Differentiator:
#     """计算一阶（速度）和二阶（加速度）差分特征，支持动态 dt。"""
#
#     def __init__(self):
#         self.prev_raw = None
#         self.prev_vel = None
#         self.prev_ts = None
#
#     def compute(self, raw: list, ts: float) -> (list, list):
#         """
#         输入:
#           raw: 当前帧的原始扁平化坐标列表
#           timestamp: 当前帧的时间戳（秒）
#         输出:
#           vel, acc — 本帧的速度和加速度列表
#         """
#         if self.prev_raw is None or self.prev_ts is None:
#             # 第一帧 or 丢失时间信息：速度、加速度均为 0
#             vel = [0.0] * len(raw)
#             acc = [0.0] * len(raw)
#         else:
#             dt = ts - self.prev_ts
#             if dt <= 0:
#                 # 防止除以 0 或负时间
#                 vel = [0.0] * len(raw)
#                 acc = [0.0] * len(raw)
#             else:
#                 # 一阶差分
#                 vel = [(raw[i] - self.prev_raw[i]) / dt for i in range(len(raw))]
#                 # 二阶差分
#                 if self.prev_vel is None:
#                     acc = [0.0] * len(raw)
#                 else:
#                     acc = [(vel[i] - self.prev_vel[i]) / dt for i in range(len(raw))]
#
#         # 更新状态
#         self.prev_raw = raw.copy()
#         self.prev_vel = vel.copy()
#         self.prev_ts = ts
#
#         return vel, acc


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


class CSVFeatureWriter:
    """负责把一帧的所有特征写入 CSV。"""

    def __init__(self, csv_path, distance_pairs, angle_triplets):
        import csv
        self.writer = csv.writer(open(csv_path, 'w', newline=''))
        header = ['frame', 'timestamp']
        # 原始点 33×4
        for i in range(33):
            header += [f'x_{i}', f'y_{i}', f'z_{i}', f'vis_{i}']
        # 速度、加速度
        for name in ('vx', 'vy', 'vz', 'vvis'):
            for i in range(33):
                header.append(f'{name}_{i}')
        for name in ('ax', 'ay', 'az', 'avis'):
            for i in range(33):
                header.append(f'{name}_{i}')
        # 距离
        for a, b in distance_pairs:
            header.append(f'dist_{a}_{b}')
        # 角度
        for a, b, c in angle_triplets:
            header.append(f'angle_{a}_{b}_{c}')
        self.writer.writerow(header)

    def write_row(self, pose_frame: PoseFrame, vel, acc, dists, angles):
        row = [pose_frame.frame_idx, pose_frame.timestamp]
        row += pose_frame.raw
        row += vel
        row += acc
        row += dists
        row += angles
        self.writer.writerow(row)
