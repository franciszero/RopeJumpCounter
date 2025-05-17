import av
import av.error
import time

class PyAVCapture:
    """
    使用 PyAV 直接打开 avfoundation 设备，得到最原生的视频包并解码。
    """
    def __init__(self, device_index=0, width=640, height=480, fps=30):
        opts = {'framerate': str(fps), 'video_size': f'{width}x{height}'}
        # 在 avfoundation 下，file=str(device_index)
        self.container = av.open(format='avfoundation', file=str(device_index), options=opts)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = 'AUTO'

    def read(self):
        """
        遍历 demux 和 decode，只取第一帧，返回 (ret, frame, latency_ms)。
        若底层管道暂时无数据，将捕获 BlockingIOError 并返回 (False, None, None)。
        """
        t0 = time.time()
        try:
            for packet in self.container.demux(self.stream):
                for frame in packet.decode():
                    img = frame.to_ndarray(format='bgr24')
                    latency = (time.time() - t0) * 1000
                    return True, img, latency
        except (av.error.BlockingIOError, BlockingIOError):
            # 非阻塞模式下无数据就绪
            return False, None, None
        return False, None, None

    def release(self):
        self.container.close()