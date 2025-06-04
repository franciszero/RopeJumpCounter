import cv2
import time

class GStreamerCapture:
    """
    Capture frames from macOS camera via GStreamer pipeline, drop=true ensures minimal latency。
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
        """
        return (ret, frame, latency_ms)
        """
        t0 = time.time()
        ret, frame = self.cap.read()
        latency = (time.time() - t0) * 1000
        return ret, frame, latency

    def release(self):
        self.cap.release()