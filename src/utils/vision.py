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
        img_rgb = cv2.cvtColor(stable_frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        return results.pose_landmarks

    def estimate1(self, stable_frame):
        """
        Run pose estimation on `frame` (BGR).
        Returns: (landmark_results, heights_dict)
        `heights_dict` maps region names ('head','torso','legs') to normalized y-values.
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
