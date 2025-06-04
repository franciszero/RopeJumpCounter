import time
from .gst_capture import GStreamerCapture
from .pyav_capture import PyAVCapture

def measure(capture_cls, name, n_frames=100):
    cap = capture_cls(device_index=0, width=640, height=480, fps=30)
    latencies = []
    for _ in range(n_frames):
        ret, frame, lat = cap.read()
        if not ret:
            break
        latencies.append(lat)
    cap.release()
    avg = sum(latencies) / len(latencies) if latencies else float('nan')
    print(f"{name} 平均采集延迟（{len(latencies)} 帧）: {avg:.1f} ms")

if __name__ == "__main__":
    measure(GStreamerCapture, "GStreamerCapture")
    measure(PyAVCapture,   "PyAVCapture")