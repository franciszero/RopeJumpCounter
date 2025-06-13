"""
Graphical user interface module

Provides the main GUI for real-time jump rope counting with video display,
performance monitoring, and optional video recording capabilities.
"""

import cv2
import time
import logging
from collections import deque
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from ..core.pyav_capture import PyAVCapture
from ..utils.performance.Perf import PerfStats
from ..ml.inference.video_predictor import VideoPredictor
from ..core.jump_counter import JumpCounter
from ..core.exceptions import CameraError, ModelError
from ..ml.data.features.features import FeaturePipeline

logger = logging.getLogger(__name__)


class PlayerGUI:
    """Real-time video player with jump counting GUI

    This class provides the main graphical interface for the jump rope counter.
    It handles video capture, real-time processing, display overlay, and
    optional video recording. The GUI shows live video with jump count,
    probability values, and performance metrics.

    Features:
    - Real-time video capture and display
    - Jump counting with visual feedback
    - Performance monitoring (FPS, latency)
    - Optional video recording
    - Error handling and recovery
    """

    def __init__(self, predictor: VideoPredictor, width: int, height: int, fps: int, save_path: str | None = None):
        """Initialize the player GUI

        Sets up video capture, predictor, counter, and optional video recording.

        Args:
            predictor: Trained model predictor for jump detection
            width: Video capture width in pixels
            height: Video capture height in pixels
            fps: Target frames per second for capture
            save_path: Optional directory path for video recording

        Raises:
            CameraError: If camera initialization fails
        """
        try:
            logger.info("Initializing camera...")
            self.cap = PyAVCapture(device_index=0, width=width, height=height, fps=fps)
        except Exception as e:
            raise CameraError(f"Camera initialization failed: {e}")

        self.zoom_height = 920  # Display height for video scaling

        self.stats = PerfStats(window_size=10)
        self.predictor = predictor
        self.counter = JumpCounter()
        self.fps = fps

        # Performance monitoring
        self.proc_times = deque(maxlen=30)  # Processing times for recent frames

        # Setup optional video recording
        if save_path:
            try:
                save_path = Path(save_path)
                save_path.mkdir(parents=True, exist_ok=True)

                time_str = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
                dest_file = save_path / f"jump_{time_str}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.writer = cv2.VideoWriter(str(dest_file), fourcc, fps, (int(width), int(height)))

                if not self.writer.isOpened():
                    logger.error(f"VideoWriter initialization failed: {dest_file}")
                    self.writer = None
                else:
                    logger.info(f"Video will be saved to: {dest_file}")
            except Exception as e:
                logger.error(f"Video writer initialization failed: {e}")
                self.writer = None
        else:
            self.writer = None

    def _overlay(self, frame: np.ndarray, jump_cnt: int, prob: float, is_on_rising: bool, t0) -> np.ndarray:
        """Draw overlay information on video frame

        Adds jump count, probability, rising state indicator, and performance
        metrics to the video frame for real-time feedback.

        Args:
            frame: Input video frame (BGR format)
            jump_cnt: Current jump count
            prob: Model prediction probability
            is_on_rising: Whether person is currently jumping up
            t0: Timestamp for performance calculation

        Returns:
            np.ndarray: Frame with overlay information drawn
        """
        # Draw jump count
        if jump_cnt is not None:
            cv2.putText(frame, f"JUMPS: {jump_cnt}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 255), 2,
                        cv2.LINE_AA)

        # Draw rising state indicator with red overlay
        if prob is not None and is_on_rising:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), thickness=-1)
            alpha = 0.15
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.putText(frame, "RISING", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 255), 2,
                        cv2.LINE_AA)

        # Draw prediction probability
        if prob is not None:
            cv2.putText(frame, f"p={prob:.2f}", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2,
                        cv2.LINE_AA)

        # Draw performance metrics
        if self.stats.proc_fps is not None and self.stats.last_latency_ms is not None:
            txt = f"{self.stats.proc_fps:4.1f} FPS | {self.stats.last_latency_ms:3.0f} ms"
            cv2.putText(frame, txt,
                        (frame.shape[1] - 260, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
                        cv2.LINE_AA)
        return frame

    def run(self):
        """Start the main video processing loop

        Runs the real-time jump counting application with video capture,
        feature extraction, model inference, counting, and display.
        Handles errors gracefully and provides clean shutdown.
        """
        logger.info("Starting video processing loop...")
        pipe = FeaturePipeline(self.cap, self.predictor.window_size)
        frame_idx = 0
        error_count = 0
        MAX_ERRORS = 5

        try:
            while True:
                try:
                    # Performance timing array
                    arr_ts = list()

                    # 1) Capture frame
                    arr_ts.append(time.time())
                    ret, frame, _ = self.cap.read()  # Original BGR frame
                    if not ret:
                        logger.warning(f"Frame capture failed ({error_count}/{MAX_ERRORS})")
                        continue
                    error_count = 0  # Reset error count on successful read

                    # 2) Feature extraction
                    arr_ts.append(time.time())
                    pipe.process_frame(frame, frame_idx)
                    frame_idx += 1

                    # 3) Model inference
                    arr_ts.append(time.time())
                    feat_vec = pd.DataFrame([pipe.fs.rec]).iloc[0][2:].values.astype(np.float32)
                    prob = self.predictor.predict(feat_vec)

                    # 4) Jump counting
                    arr_ts.append(time.time())
                    is_on_rising, jump_cnt = self.counter.process_prediction(prob, self.predictor.threshold)

                    # 5) Display overlay
                    frame_vis = self._overlay(pipe.fs.raw_frame.copy(), jump_cnt, prob, is_on_rising, arr_ts[0])

                    # 6) Show video and optional recording
                    cv2.imshow("JumpRope RealTime", frame_vis)
                    if self.writer:
                        self.writer.write(frame)

                    # 7) Update performance statistics
                    arr_ts.append(time.time())
                    self.stats.update("[Main Process]: ", arr_ts, 0)

                    # 8) Check for exit command
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User requested exit")
                        break

                except ModelError as e:
                    logger.error(f"Model error: {e}")
                    break
                except Exception as e:
                    error_count += 1
                    if error_count > MAX_ERRORS:
                        logger.error(f"Consecutive error count exceeded threshold: {e}")
                        break
                    logger.warning(f"Processing error ({error_count}/{MAX_ERRORS}): {e}")
                    continue

        finally:
            logger.info("Cleaning up resources...")
            self.cap.release()
            if self.writer is not None:
                self.writer.release()
            cv2.destroyAllWindows()
