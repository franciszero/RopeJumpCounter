import cv2
import numpy as np


class VideoStabilizer:
    """
    Video stabilizer based on LK optical flow + affine estimation，Each call to stabilize(frame) aligns current frame to previous frame reference coordinate system。

    Algorithm workflow：
      1. First frame：Detect corner points on grayscale image, directly return original image。
      2. Subsequent frames：
         a. Use calcOpticalFlowPyrLK to track previous frame corner points to current grayscale image。
         b. filter successful point pairs (pts0->pts1)。
         c. when enough point pairs, estimate affine transform, otherwise use identity matrix。
         d. convert to cumulative matrix and warp to previous frame coordinate system。
         e. every N frames re-detect new corners, otherwise continue trackingof pts1。
    """

    max_corners = 200
    quality_level = 0.01
    min_distance = 30
    reinit_interval = 30

    def __init__(self):
        self.max_corners = type(self).max_corners
        self.quality_level = type(self).quality_level
        self.min_distance = type(self).min_distance
        self.reinit_interval = type(self).reinit_interval

        self.prev_gray = None  # previous grayscale image
        self.prev_pts = None  # uponeframe待followtrackcornerpoint
        self.transforms = []  # accumulateproductof 3×3 same质imitateprojectmatrixarray列表
        self.frame_count = 0  # framecount，useatcontrolmake何timeagaindetectioncornerpoint

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """
        pair齐currentframetouponeframesitscalarsystemmiddle，outputsupplementcompensateshakemotionofafterofnewframe。

        Args:
            frame: BGR 彩colorimage

        Returns:
            stabilized: BGR 彩colorimage，alreadydo逆imitateprojectpair齐
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ----- 第 1 frame：Initialize -----
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance
            )
            # 第oneframeno需pair齐，accumulateproductoneunitssingleunitmatrixarray
            I2x3 = np.array([[1, 0, 0],
                             [0, 1, 0]], dtype=np.float32)
            M3x3 = np.vstack([I2x3, [0, 0, 1]])
            self.transforms.append(M3x3)
            return frame

        # ----- Subsequent frames：followtrack + estimatecalculateimitateproject -----
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray,
            self.prev_pts, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        # 只保留followtracksuccessofpointpair
        mask = (status.flatten() == 1)
        pts0 = self.prev_pts[mask].reshape(-1, 2)
        pts1 = curr_pts[mask].reshape(-1, 2)

        # ifresultpointpair太少，directlyusesingleunit 2×3 matrixarray
        if len(pts0) < 10:
            M2x3 = np.array([[1, 0, 0],
                             [0, 1, 0]], dtype=np.float32)
        else:
            M2x3, inliers = cv2.estimateAffine2D(
                pts0, pts1,
                method=cv2.RANSAC,
                ransacReprojThreshold=3,
                maxIters=200
            )
            if M2x3 is None:
                M2x3 = np.array([[1, 0, 0],
                                 [0, 1, 0]], dtype=np.float32)

        # accumulateproductto 3×3 same质matrixarrayandsave
        M3x3 = np.vstack([M2x3, [0, 0, 1]]).astype(np.float32)
        self.transforms.append(M3x3)

        # paircurrentframedo逆changetransform：will其to previousframesitscalarsystem
        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(
            frame,
            M2x3,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )

        # ----- updatecornerpoint：eachseparate reinit_interval frameagaindetection，noruledirectlyuse刚followtracktoof pts1 -----
        self.frame_count += 1
        next_gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
        if self.frame_count % self.reinit_interval == 0 or pts1.shape[0] < 10:
            # againdetectionnewcornerpoint
            self.prev_pts = cv2.goodFeaturesToTrack(
                next_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance
            )
        else:
            # continuecontinueuse刚followtrackgettoof pts1
            self.prev_pts = pts1.reshape(-1, 1, 2)

        # updateprevious grayscale image
        self.prev_gray = next_gray

        return stabilized
