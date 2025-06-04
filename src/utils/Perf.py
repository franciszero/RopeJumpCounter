"""
Performance monitoring utility module

Provides real-time performance statistics tracking for video processing
applications including FPS calculation, latency measurement, and timing analysis.
"""

import collections
from typing import Deque

import numpy as np


class PerfStats:
    """Real-time performance statistics tracker

    Monitors processing performance by tracking frame processing times,
    calculating average FPS, and measuring latency. Uses a sliding window
    approach to provide smooth, real-time performance metrics.

    Features:
    - Sliding window FPS calculation
    - Latency measurement in milliseconds
    - Detailed timing breakdown for pipeline stages
    - Configurable reporting intervals
    """

    def __init__(self, window_size=10):
        """Initialize performance statistics tracker

        Args:
            window_size: Number of recent measurements to keep for averaging
        """
        self.times: Deque[float] = collections.deque(maxlen=window_size)
        self.last_latency_ms: float = 0.0
        self.proc_fps: float = 0.0
        self.cnt = 0

    def update(self, msg: str, arr_ts: list, limit: int = 10):
        """Update performance statistics with new timing measurements

        Processes an array of timestamps to calculate stage-wise timing
        and overall processing latency. Updates FPS and latency metrics.

        Args:
            msg: Description message for this measurement
            arr_ts: Array of timestamps marking processing stages
            limit: Number of frames between detailed logging (default: 10)
        """
        # Calculate time differences between stages in milliseconds
        time_elapses = np.diff(np.array(arr_ts)) * 1000
        total_time = sum(time_elapses)

        # Add to sliding window
        self.times.append(total_time)

        if self.times:
            if self.cnt >= limit:
                # Log detailed timing breakdown
                timing_breakdown = "+".join(f"{x:.1f}" for x in time_elapses)
                print(f"{msg} : {timing_breakdown} = {total_time:.1f}")

                # Update metrics
                self.last_latency_ms = total_time
                avg_time = sum(self.times) / len(self.times)
                self.proc_fps = 1000 / avg_time if avg_time > 0 else 0.0
                self.cnt = 0
            else:
                self.cnt += 1

    def info_text(self, video_fps: float) -> str:
        """Generate formatted performance information string

        Args:
            video_fps: Input video frame rate for comparison

        Returns:
            str: Formatted string with video FPS, processing FPS, and latency
        """
        return f"video FPS {video_fps:.1f} | proc FPS {self.proc_fps:.1f} | latency {self.last_latency_ms:.1f} ms"
