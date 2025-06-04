"""
Frame sampling and feature extraction module

Provides comprehensive frame-level feature extraction for jump rope counting.
Handles pose landmark processing, temporal features, spatial relationships,
and windowed feature aggregation for machine learning models.
"""

import time
from collections import deque
import numpy as np
import cv2
import mediapipe as mp

PoseLandmark = mp.solutions.pose.PoseLandmark

# Selected landmarks relevant for rope jump counting (16 key points)
# These landmarks capture the essential body parts needed for jump detection
SELECTED_LM = [
    PoseLandmark.LEFT_EYE,          # Head tracking
    PoseLandmark.RIGHT_EYE,         # Head tracking
    PoseLandmark.LEFT_SHOULDER,     # Upper body structure
    PoseLandmark.RIGHT_SHOULDER,    # Upper body structure
    PoseLandmark.LEFT_ELBOW,        # Arm movement
    PoseLandmark.RIGHT_ELBOW,       # Arm movement
    PoseLandmark.LEFT_WRIST,        # Hand/rope position
    PoseLandmark.RIGHT_WRIST,       # Hand/rope position
    PoseLandmark.LEFT_HIP,          # Core body tracking
    PoseLandmark.RIGHT_HIP,         # Core body tracking
    PoseLandmark.LEFT_KNEE,         # Leg movement
    PoseLandmark.RIGHT_KNEE,        # Leg movement
    PoseLandmark.LEFT_HEEL,         # Foot position
    PoseLandmark.RIGHT_HEEL,        # Foot position
    PoseLandmark.LEFT_FOOT_INDEX,   # Foot landing detection
    PoseLandmark.RIGHT_FOOT_INDEX,  # Foot landing detection
]


class FrameSample:
    """Comprehensive frame-level feature extraction for jump rope analysis

    This class encapsulates all per-frame data processing and feature computation
    for video frames in jump rope counting applications. It handles multiple types
    of features including pose landmarks, temporal derivatives, spatial relationships,
    and windowed aggregations.

    Features computed:
    - Raw normalized coordinates (x, y, z, visibility)
    - Pixel coordinates for selected landmarks
    - Velocity features (first-order temporal differences)
    - Acceleration features (second-order temporal differences)
    - Distance features between joint pairs
    - Angle features between joint triplets
    - Windowed feature sequences for temporal models

    The class maintains a sliding window buffer for temporal models that require
    sequences of feature vectors as input.
    """

    def __init__(self, cap, window_size: int = 1):
        """Initialize frame sample processor

        Sets up feature extraction components and sliding window buffer
        for temporal feature aggregation.

        Args:
            cap: Video capture object for frame dimension extraction
            window_size: Size of sliding window for temporal features (default: 1)
        """
        self.cap = cap
        self.window_size = window_size  # Window size for temporal feature buffering
        # Sliding window buffer to hold sequences of feature vectors for models that require temporal context
        self.buffer = deque(maxlen=self.window_size)

        self.raw_frame = None
        self.h, self.w = None, None
        self.timestamp = None

        # Feature buffers initialized as empty lists:
        self.raw_norm = []  # Normalized landmark coordinates + visibility
        self.raw_px = []  # Landmark coordinates in pixel space
        self.vel = []  # Velocity features (first-order temporal differences)
        self.acc = []  # Acceleration features (second-order temporal differences)
        self.dists = []  # Distances between selected joint pairs
        self.angs = []  # Angles between selected joint triplets

        self.rec = None

        self.len = len(SELECTED_LM)

    def init_current_frame(self, frame_idx):
        self.h, self.w = self.raw_frame.shape[:2]

        try:
            timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            if timestamp_ms and timestamp_ms > 0.0:
                self.timestamp = timestamp_ms / 1000.0
            else:
                self.timestamp = time.time()
        except AttributeError:
            # cap has no get(), e.g. PyAVCapture
            self.timestamp = time.time()

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
            self.raw_norm = [0.0] * self.len * 4  # 33 landmarks * 4 values each (x,y,z,visibility)
            self.raw_norm = [0.0] * self.len * 4
        else:
            # Extract normalized coordinates and visibility
            self.raw_norm = []
            for enum_lm in SELECTED_LM:
                m = lm.landmark[enum_lm.value]
                self.raw_norm.extend([m.x, m.y, m.z, m.visibility])
        # raw x,y,z,vis
        for i, enum_lm in enumerate(SELECTED_LM):
            self.rec[f'x_{enum_lm.value}'] = self.raw_norm[4 * i]
            self.rec[f'y_{enum_lm.value}'] = self.raw_norm[4 * i + 1]
            self.rec[f'z_{enum_lm.value}'] = self.raw_norm[4 * i + 2]
            self.rec[f'vis_{enum_lm.value}'] = self.raw_norm[4 * i + 3]

    def compute_raw_px(self, lm):
        """
        Compute normalized and pixel coordinates of landmarks.

        This method populates self.raw_norm and self.raw_px based on the current landmarks.
        If landmarks are missing, fills features with zeros.
        """
        if not lm:
            # No landmarks detected: fill with zeros
            self.raw_px = [0.0] * self.len * 2  # 33 landmarks * 2 values each (x_px, y_px)
        else:
            # Determine frame size for pixel conversion
            self.raw_px = []
            for m in lm.landmark:
                # Convert normalized coords to pixel coords
                self.raw_px.extend([m.x * self.w, m.y * self.h])
        # 像素坐标特征
        for i, enum_lm in enumerate(SELECTED_LM):
            self.rec[f'x_px_{enum_lm.value}'] = self.raw_px[2 * i]
            self.rec[f'y_px_{enum_lm.value}'] = self.raw_px[2 * i + 1]

    def compute_diff(self, diff):
        """
        Compute velocity and acceleration features using a Differentiator instance.

        Args:
            diff (Differentiator): Instance to compute velocity and acceleration.
        """
        # Compute velocity and acceleration given current and previous raw_norm and timestamps
        self.vel, self.acc = diff.diff_compute(self.raw_norm, self.len, self.timestamp)
        # velocity features
        for i, enum_lm in enumerate(SELECTED_LM):
            self.rec[f'vx_{enum_lm.value}'] = self.vel[3 * i]
            self.rec[f'vy_{enum_lm.value}'] = self.vel[3 * i + 1]
            self.rec[f'vz_{enum_lm.value}'] = self.vel[3 * i + 2]
        # acceleration features
        for i, enum_lm in enumerate(SELECTED_LM):
            self.rec[f'ax_{enum_lm.value}'] = self.acc[3 * i]
            self.rec[f'ay_{enum_lm.value}'] = self.acc[3 * i + 1]
            self.rec[f'az_{enum_lm.value}'] = self.acc[3 * i + 2]

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
