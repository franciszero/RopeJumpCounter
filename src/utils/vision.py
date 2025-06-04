"""
Computer vision utility module

Provides pose estimation capabilities using MediaPipe for human pose detection
and landmark extraction in video frames. Supports region-based analysis and
height calculations for jump rope counting applications.
"""

import cv2
import mediapipe as mp


class PoseEstimator:
    """MediaPipe-based human pose estimation wrapper

    Encapsulates MediaPipe Pose functionality to extract human pose landmarks
    from video frames. Provides methods for landmark detection, region analysis,
    and height calculations for different body parts.

    Features:
    - Real-time pose landmark detection
    - Region-based body part analysis (head, torso, legs)
    - Normalized coordinate extraction
    - Configurable detection and tracking confidence
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
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

    def get_pose_landmarks(self, stable_frame):
        """Extract pose landmarks from video frame

        Processes a BGR video frame to detect human pose landmarks using MediaPipe.

        Args:
            stable_frame: Input video frame in BGR format

        Returns:
            MediaPipe pose landmarks object or None if no pose detected
        """
        img_rgb = cv2.cvtColor(stable_frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        return results.pose_landmarks

    def estimate1(self, stable_frame):
        """Extract body region heights from pose landmarks

        Analyzes pose landmarks to calculate normalized y-coordinates for
        different body regions (head, torso, legs). Used for jump analysis.

        Args:
            stable_frame: Input video frame in BGR format

        Returns:
            dict: Region heights mapping {'head': y, 'torso': y, 'legs': y}
                  where y-values are normalized coordinates (0.0-1.0)
        """
        lm = self.get_pose_landmarks(stable_frame)
        heights = {}
        if lm is not None:
            # normalized y-values for head, torso, legs
            nose = lm.landmark[self.mp_pose.PoseLandmark.NOSE]
            l_sh = lm.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_sh = lm.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hi = lm.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            r_hi = lm.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            heights['head'] = nose.y
            heights['torso'] = (l_sh.y + r_sh.y) / 2
            heights['legs'] = (l_hi.y + r_hi.y) / 2
        return heights
