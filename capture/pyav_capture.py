"""
PyAV Video Capture Module

Provides low-latency video capture using PyAV library with direct access
to AVFoundation devices for optimal performance on macOS.
"""

import av
import av.error
import time


class PyAVCapture:
    """PyAV-based video capture for low-latency camera access

    Uses PyAV to directly open AVFoundation devices, providing access to
    native video packets and decoding for minimal latency video capture.

    This implementation bypasses OpenCV's video capture overhead and provides
    direct access to the camera hardware through AVFoundation.

    Args:
        device_index: Camera device index (default: 0)
        width: Frame width in pixels (default: 640)
        height: Frame height in pixels (default: 480)
        fps: Target frame rate (default: 30)
    """

    def __init__(self, device_index=0, width=640, height=480, fps=30):
        opts = {'framerate': str(fps), 'video_size': f'{width}x{height}'}
        # In AVFoundation, file parameter is the device index as string
        self.container = av.open(format='avfoundation', file=str(device_index), options=opts)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = 'AUTO'

    def read(self):
        """Read a single frame from the video stream

        Iterates through demux and decode operations, taking only the first
        available frame to minimize latency.

        Returns:
            tuple: (success, frame, latency_ms) where:
                - success (bool): True if frame was captured successfully
                - frame (np.ndarray): BGR24 format frame, or None if failed
                - latency_ms (float): Capture latency in milliseconds, or None if failed

        Note:
            If the underlying pipeline temporarily has no data available,
            catches BlockingIOError and returns (False, None, None).
        """
        t0 = time.time()
        try:
            for packet in self.container.demux(self.stream):
                for frame in packet.decode():
                    img = frame.to_ndarray(format='bgr24')
                    latency = (time.time() - t0) * 1000
                    return True, img, latency
        except (av.error.BlockingIOError, BlockingIOError):
            # No data ready in non-blocking mode
            return False, None, None
        return False, None, None

    def release(self):
        """Release video capture resources

        Closes the AVFoundation container and frees associated resources.
        It Should be called when video capture is no longer needed.
        """
        self.container.close()
