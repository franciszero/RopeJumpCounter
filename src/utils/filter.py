from collections import deque
import numpy as np


# =========================
# 3. TrendFilter：Exponential smoothing + moving average trend separation
# =========================
class TrendFilter:
    def __init__(self, buffer_len=320, alpha=0.2, trend_win=64, baseline=50):
        self.alpha = alpha
        self.trend_win = trend_win
        self.baseline = baseline
        self.raw_buf = deque(maxlen=buffer_len)
        self.smooth_buf = deque(maxlen=buffer_len)
        self.trend_buf = deque(maxlen=buffer_len)
        self.fluct_buf = deque(maxlen=buffer_len)

    """

    inputRelative velocity after background removal and frame number idx
    returnHigh frequency fluctuation f

    """
    def update(self, rel_speed, idx):
        if idx <= self.baseline:
            for buf in (self.raw_buf,
                        self.smooth_buf,
                        self.trend_buf,
                        self.fluct_buf):
                buf.append(0.0)
            return 0.0

        # Original velocity
        self.raw_buf.append(rel_speed)
        # Exponential smoothing
        last_s = self.smooth_buf[-1]
        s = self.alpha * rel_speed + (1 - self.alpha) * last_s
        # moving average trend
        t = np.mean(list(self.smooth_buf)[-self.trend_win:])
        # high frequency component
        f = s - t

        # updatecache
        self.smooth_buf.append(s)
        self.trend_buf.append(t)
        self.fluct_buf.append(f)
        return f