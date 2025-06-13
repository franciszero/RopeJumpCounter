"""
Feature extraction pipeline module

Provides comprehensive feature extraction for jump rope counting including
pose landmarks, spatial features, temporal differences, and windowed features.
Integrates video stabilization and pose estimation for robust feature computation.
"""

import math

from src.utils.performance.Perf import PerfStats
from src.utils.vision import PoseEstimator
from src.utils.VideoStabilizer import VideoStabilizer
from src.utils.common.Differentiator import get_differentiator
from src.utils.common.FrameSample import FrameSample
from src.ml.data.builders.feature_mode import Feature, get_feature_mode


class FeaturePipeline:
    """Complete feature extraction pipeline for jump rope analysis

    This class orchestrates the entire feature extraction process from raw video
    frames to structured feature vectors. It handles video stabilization, pose
    estimation, spatial feature computation, temporal differences, and windowed
    aggregation based on configurable feature modes.

    The pipeline supports multiple feature types:
    - RAW: Normalized pose landmarks
    - RAW_PX: Pixel-space pose landmarks
    - DIFF: Temporal differences between frames
    - SPATIAL: Distance and angle features between joints
    - WINDOW: Windowed aggregation features
    """

    def __init__(self, cap, window_size):
        """Initialize the feature extraction pipeline

        Sets up all components needed for feature extraction including
        video stabilization, pose estimation, and feature calculators.

        Args:
            cap: Video capture object for frame dimensions
            window_size: Number of frames for windowed features
        """
        self.window_size = window_size
        self.fs = FrameSample(cap, self.window_size)

        # Initialize processing components
        self.stabilizer = VideoStabilizer()
        self.pose_est = PoseEstimator()
        self.diff = get_differentiator()
        self.dist_calc = DistanceCalculator()
        self.ang_calc = AngleCalculator()

        # Performance monitoring
        self.stats = PerfStats(window_size=10)

        # Store landmarks for visualization
        self.landmarks = None

    def process_frame(self, frame, frame_idx, mode=None):
        """Process a single frame through the feature extraction pipeline

        Applies video stabilization, pose estimation, and feature extraction
        based on the specified feature mode configuration.

        Args:
            frame: Input video frame (BGR format)
            frame_idx: Frame index for temporal tracking
            mode: Feature extraction mode (uses default if None)
        """
        # Initialize frame processing
        self.fs.raw_frame = frame
        self.fs.init_current_frame(frame_idx)

        # Apply video stabilization and pose estimation
        stable = self.stabilizer.stabilize(self.fs.raw_frame)
        lm = self.pose_est.get_pose_landmarks(stable)

        # Store landmarks for visualization purposes
        self.landmarks = lm

        # Extract features based on configured mode
        mode = mode if mode is not None else get_feature_mode()
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
    """Calculate Euclidean distances between joint pairs

    Computes 3D distances between predefined pairs of body landmarks
    to capture spatial relationships important for jump detection.
    These distance features help the model understand body proportions
    and movement patterns.
    """

    def __init__(self):
        """Initialize distance calculator with predefined joint pairs

        Defines pairs of landmark indices for distance computation:
        - Hip to knee (leg segments)
        - Knee to ankle (lower leg segments)
        - Shoulder to elbow (upper arm segments)
        - Elbow to wrist (forearm segments)
        """
        # Joint pairs: (start_idx, end_idx) for MediaPipe pose landmarks
        self.pairs = [
            (24, 26),  # Left hip to left knee
            (26, 28),  # Left knee to left ankle
            (11, 13),  # Left shoulder to left elbow
            (13, 15)  # Left elbow to left wrist
        ]

    def compute(self, landmarks) -> list:
        """Compute 3D Euclidean distances for all joint pairs

        Args:
            landmarks: MediaPipe pose landmarks object

        Returns:
            list: Distance values for each joint pair in the same order as self.pairs
        """
        dists = []
        for a, b in self.pairs:
            pa, pb = landmarks[a], landmarks[b]
            # Calculate 3D Euclidean distance
            dx = pa.x - pb.x
            dy = pa.y - pb.y
            dz = pa.z - pb.z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            dists.append(distance)
        return dists


class AngleCalculator:
    """Calculate joint angles from three-point configurations

    Computes angles at specific joints to capture body posture and
    movement dynamics. These angular features are crucial for
    understanding jump mechanics and body positioning.
    """

    def __init__(self):
        """Initialize angle calculator with predefined joint triplets

        Defines triplets of landmark indices for angle computation:
        - Hip-knee-ankle angle (leg bend)
        - Shoulder-elbow-wrist angle (arm bend)
        - Hip-shoulder-elbow angle (torso lean)
        """
        # Joint triplets: (point_a, vertex_b, point_c) for angle at vertex_b
        self.triplets = [
            (24, 26, 28),  # Hip-knee-ankle angle (left leg)
            (11, 13, 15),  # Shoulder-elbow-wrist angle (left arm)
            (23, 11, 13)  # Hip-shoulder-elbow angle (torso lean)
        ]

    def compute(self, landmarks) -> list:
        """Compute angles for all joint triplets

        Calculates the angle at the middle point of each triplet using
        vector dot product and arc cosine.

        Args:
            landmarks: MediaPipe pose landmarks object

        Returns:
            list: Angle values in degrees for each triplet
        """
        angles = []
        for a, b, c in self.triplets:
            p1, p2, p3 = landmarks[a], landmarks[b], landmarks[c]

            # Create vectors from vertex to other points
            v1 = (p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)  # Vector p2->p1
            v2 = (p3.x - p2.x, p3.y - p2.y, p3.z - p2.z)  # Vector p2->p3

            # Calculate angle using dot product
            dot = sum(v1[i] * v2[i] for i in range(3))
            mag1 = math.sqrt(sum(v1[i] * v1[i] for i in range(3)))
            mag2 = math.sqrt(sum(v2[i] * v2[i] for i in range(3)))

            # Avoid division by zero and compute angle in degrees
            if mag1 * mag2 > 0:
                cos_angle = dot / (mag1 * mag2)
                # Clamp to valid range for acos
                cos_angle = max(-1.0, min(1.0, cos_angle))
                angle = math.degrees(math.acos(cos_angle))
            else:
                angle = 0.0

            angles.append(angle)
        return angles
