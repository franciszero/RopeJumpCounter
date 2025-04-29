import cv2
import mediapipe as mp


class PoseEstimator:
    """
    Wrapper around MediaPipe Pose to extract landmarks and region heights.
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

    def estimate(self, frame):
        """
        Run pose estimation on `frame` (BGR).
        Returns: (landmark_results, heights_dict)
        `heights_dict` maps region names ('head','torso','legs') to normalized y-values.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        lm = results.pose_landmarks
        heights = {}
        if lm:
            # normalized y-values for head, torso, legs
            nose = lm.landmark[self.mp_pose.PoseLandmark.NOSE]
            l_sh = lm.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_sh = lm.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hi = lm.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            r_hi = lm.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            heights['head'] = nose.y
            heights['torso'] = (l_sh.y + r_sh.y) / 2
            heights['legs'] = (l_hi.y + r_hi.y) / 2
        return lm, heights
