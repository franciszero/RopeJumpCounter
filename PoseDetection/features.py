# features.py

import math
from collections import deque
import cv2
from utils.vision import PoseEstimator


class PoseFrame:
    """封装单帧的原始关键点数据和帧索引/时间戳。"""

    def __init__(self, frame_idx: int, timestamp: float, landmarks, frame_size=None):
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        # 将原始数据平坦化为 [x0,y0,z0,vis0, ..., x32,y32,z32,vis32]
        self.raw = []
        for lm in landmarks:
            self.raw.extend([lm.x, lm.y, lm.z, lm.visibility])
        if frame_size is not None:
            h, w = frame_size
            self.raw_px = []
            for lm in landmarks:
                self.raw_px.extend([lm.x * w, lm.y * h])


class Differentiator:
    """计算一阶（速度）和二阶（加速度）差分特征。"""

    def __init__(self, dt: float):
        self.dt = dt
        self.prev_raw = None
        self.prev_vel = None

    def compute(self, raw: list) -> (list, list):
        """输入原始 flat list，输出 (vel, acc)。"""
        if self.prev_raw is None:
            vel = [0.0] * len(raw)
            acc = [0.0] * len(raw)
        else:
            vel = [(raw[i] - self.prev_raw[i]) / self.dt for i in range(len(raw))]
            if self.prev_vel is None:
                acc = [0.0] * len(raw)
            else:
                acc = [(vel[i] - self.prev_vel[i]) / self.dt for i in range(len(raw))]
        self.prev_raw = raw.copy()
        self.prev_vel = vel.copy()
        return vel, acc


class DistanceCalculator:
    """计算若干关节点对之间的欧氏距离。"""

    def __init__(self, pairs):
        """
        pairs: list of (idx_a, idx_b) 元组，索引对应 landmarks 列表的序号
        """
        self.pairs = pairs

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

    def __init__(self, triplets):
        """
        triplets: list of (idx_a, idx_b, idx_c) ，以 b 为顶点算 ∠ABC
        """
        self.triplets = triplets

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
