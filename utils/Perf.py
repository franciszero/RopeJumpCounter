import collections
import time
from typing import Deque

import numpy as np


class PerfStats:
    def __init__(self, window_size=10):
        self.times: Deque[float] = collections.deque(maxlen=window_size)
        self.last_latency_ms: float = 0.0
        self.proc_fps: float = 0.0
        self.cnt = 0

    def update(self, msg, arr_ts, limit=10):
        time_elapses = np.diff(np.array(arr_ts)) * 1000
        total_time = sum(time_elapses)

        self.times.append(total_time)
        if self.times:
            if self.cnt >= limit:
                s = "+".join(f"{x:.1f}" for x in time_elapses)
                print(f"{msg} : {s} = {total_time:.1f}")
                self.last_latency_ms = total_time
                self.proc_fps = 1000 / (sum(self.times) / len(self.times))
                self.cnt = 0
            else:
                self.cnt += 1

    def info_text(self, video_fps: float) -> str:
        return f"video FPS {video_fps:.1f} | proc FPS {self.proc_fps:.1f} | latency {self.last_latency_ms:.1f} ms"
