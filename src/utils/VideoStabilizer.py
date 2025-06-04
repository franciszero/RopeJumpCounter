import cv2
import numpy as np


class VideoStabilizer:
    """
    Video stabilizer based on LK optical flow + affine estimation. Each call to stabilize(frame) aligns current frame to previous frame reference coordinate system.

    Algorithm workflow:
      1. First frame: Detect corner points on grayscale image, directly return original image.
      2. Subsequent frames:
         a. Use calcOpticalFlowPyrLK to track previous frame corner points to current grayscale image.
         b. Filter successful point pairs (pts0->pts1).
         c. When enough point pairs, estimate affine transform, otherwise use identity matrix.
         d. Convert to cumulative matrix and warp to previous frame coordinate system.
         e. Every N frames re-detect new corners, otherwise continue tracking pts1.
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
        self.prev_pts = None  # previous frame corner points to track
        self.transforms = []  # accumulated list of 3×3 homogeneous transformation matrices
        self.frame_count = 0  # frame count, used to control when to re-detect corner points

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """
        Align current frame to previous frame coordinate system, output new frame after compensating shake motion.

        Args:
            frame: BGR color image

        Returns:
            stabilized: BGR color image, already inverse transformed and aligned
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ----- First frame: Initialize -----
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance
            )
            # First frame needs no alignment, accumulate identity matrix
            I2x3 = np.array([[1, 0, 0],
                             [0, 1, 0]], dtype=np.float32)
            M3x3 = np.vstack([I2x3, [0, 0, 1]])
            self.transforms.append(M3x3)
            return frame

        # ----- Subsequent frames: track + estimate transformation -----
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray,
            self.prev_pts, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        # Only keep successfully tracked point pairs
        mask = (status.flatten() == 1)
        pts0 = self.prev_pts[mask].reshape(-1, 2)
        pts1 = curr_pts[mask].reshape(-1, 2)

        # If too few point pairs, directly use identity 2×3 matrix
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

        # Accumulate to 3×3 homogeneous matrix and save
        M3x3 = np.vstack([M2x3, [0, 0, 1]]).astype(np.float32)
        self.transforms.append(M3x3)

        # Apply inverse transform to current frame: align it to previous frame coordinate system
        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(
            frame,
            M2x3,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )

        # ----- Update corner points: re-detect every reinit_interval frames, otherwise directly use just tracked pts1 -----
        self.frame_count += 1
        next_gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
        if self.frame_count % self.reinit_interval == 0 or pts1.shape[0] < 10:
            # Re-detect new corner points
            self.prev_pts = cv2.goodFeaturesToTrack(
                next_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance
            )
        else:
            # Continue using just tracked pts1
            self.prev_pts = pts1.reshape(-1, 1, 2)

        # Update previous grayscale image
        self.prev_gray = next_gray

        return stabilized
