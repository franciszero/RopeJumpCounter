"""
GStreamer Video Capture Module

Provides low-latency video capture using GStreamer pipeline for macOS cameras
with optimized settings for minimal latency.
"""

import cv2
import time


class GStreamerCapture:
    """GStreamer-based video capture for low-latency camera access

    Captures frames from macOS camera via GStreamer pipeline with drop=true
    to ensure minimal latency. Falls back to standard OpenCV VideoCapture
    if GStreamer is not available.

    Args:
        device_index: Camera device index (default: 0)
        width: Frame width in pixels (default: 640)
        height: Frame height in pixels (default: 480)
        fps: Target frame rate (default: 30)
    """

    def __init__(self, device_index=0, width=640, height=480, fps=30):
        pipeline = (
            f"avfvideosrc device-index={device_index} ! "
            f"video/x-raw,format=BGRA,framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1"
        )
        try:
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open GStreamer pipeline")
        except Exception:
            print(f"Warning: Failed to open GStreamer pipeline: {pipeline}, falling back to standard VideoCapture")
            self.cap = cv2.VideoCapture(device_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Unable to open camera device: {device_index}")

    def read(self):
        """Read a single frame from the video stream

        Returns:
            tuple: (success, frame, latency_ms) where:
                - success (bool): True if frame was captured successfully
                - frame (np.ndarray): BGR format frame, or None if failed
                - latency_ms (float): Capture latency in milliseconds
        """
        t0 = time.time()
        ret, frame = self.cap.read()
        latency = (time.time() - t0) * 1000
        return ret, frame, latency

    def release(self):
        """Release video capture resources

        Closes the video capture and frees associated resources.
        Should be called when video capture is no longer needed.
        """
        self.cap.release()