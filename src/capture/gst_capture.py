import cv2
import time

class GStreamerCapture:
    """
    通过 GStreamer 管道从 macOS 摄像头拉帧，drop=true 保证尽可能低延迟。
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
                raise RuntimeError(f"无法打开摄像头设备: {device_index}")

    def read(self):
        """
        返回 (ret, frame, latency_ms)
        """
        t0 = time.time()
        ret, frame = self.cap.read()
        latency = (time.time() - t0) * 1000
        return ret, frame, latency

    def release(self):
        self.cap.release()