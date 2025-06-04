import av
import av.error
import time

class PyAVCapture:
    """
    Use PyAV to directly open avfoundation device, get native video packets and decode。
    """
    def __init__(self, device_index=0, width=640, height=480, fps=30):
        opts = {'framerate': str(fps), 'video_size': f'{width}x{height}'}
        # in avfoundation，file=str(device_index)
        self.container = av.open(format='avfoundation', file=str(device_index), options=opts)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = 'AUTO'

    def read(self):
        """
        Iterate through demux and decode, take only first frame, return (ret, frame, latency_ms)。
        If underlying pipeline temporarily has no data, catch BlockingIOError and return (False, None, None)。
        """
        t0 = time.time()
        try:
            for packet in self.container.demux(self.stream):
                for frame in packet.decode():
                    img = frame.to_ndarray(format='bgr24')
                    latency = (time.time() - t0) * 1000
                    return True, img, latency
        except (av.error.BlockingIOError, BlockingIOError):
            # no data ready in non-blocking modedataready
            return False, None, None
        return False, None, None

    def release(self):
        self.container.close()