# rope_jump_counter/differentiator.py

"""
Module: differentiator
Provides a stable and consistent Differentiator for generating velocity and acceleration
features using either moving average + differentiation or Savitzky–Golay filtering.
"""

import numpy as np
from collections import deque
from scipy.signal import savgol_filter

# Default configuration for the Differentiator
_DEFAULT_DIFF_CONFIG = {
    'mode': 'ma',  # 'ma' for Moving Average + Differentiation, 'sg' for Savitzky–Golay
    'ma_window': 3,  # Window size for moving average (must be odd)
    'sg_window': 5,  # Window length for Savitzky–Golay filter (must be odd and >= sg_poly + 2)
    'sg_poly': 2,  # Polynomial order for Savitzky–Golay filter
}


class Differentiator:
    """
    Differentiator computes first-order (velocity) and second-order (acceleration)
    derivatives of a time series of feature vectors.

    It supports two modes:
    - Moving Average + Differentiation ('ma'): smooth raw input with a moving average
      then compute discrete differences.
    - Savitzky–Golay ('sg'): apply a Savitzky–Golay filter to obtain smoothed signal
      and its derivatives.

    Args:
        mode (str): 'ma' or 'sg'.
        ma_window (int): Window size for moving average (odd integer).
        sg_window (int): Window size for Savitzky–Golay filter (odd integer).
        sg_poly (int): Polynomial order for Savitzky–Golay filter.
    """

    def __init__(self, mode, ma_window, sg_window, sg_poly):
        # Validate mode selection
        assert mode in ('ma', 'sg'), "mode must be 'ma' or 'sg'"
        self.mode = mode
        # Store config parameters
        self.ma_window = ma_window
        self.sg_window = sg_window
        self.sg_poly = sg_poly

        # Buffers for sliding window operations
        self.raw_buffer = deque(maxlen=self.ma_window)
        self.sg_buffer = deque(maxlen=self.sg_window)

        # Previous state variables for difference calculations
        self.prev_raw = None
        self.prev_vel = None
        self.prev_ts = None

    def diff_compute(self, raw, ts):
        """
        Compute velocity and acceleration for the given raw feature vector at time ts.

        Args:
            raw (list[float]): Flattened feature vector of current frame.
            ts (float): Timestamp of current frame in seconds.

        Returns:
            vel (list[float]): First-order derivative (velocity) vector.
            acc (list[float]): Second-order derivative (acceleration) vector.
        """
        raw_remove_vis = [
            raw[4 * i + j]  # remove visibility
            for i in range(33)
            for j in (0, 1, 2)
        ]
        if self.mode == 'ma':
            return self._compute_ma(raw_remove_vis, ts)
        else:
            return self._compute_sg(raw_remove_vis, ts)

    def _compute_ma(self, raw, ts):
        """
        Internal method: apply moving average to raw, then differentiate.

        1. Append current raw to buffer.
        2. Once buffer is full, compute average across buffer.
        3. Compute vel/acc via discrete differences on smoothed signal.
        """
        # Add new data point
        self.raw_buffer.append(raw.copy())
        # If not enough frames, return zeros
        if len(self.raw_buffer) < self.ma_window:
            vel = [0.0] * len(raw)
            acc = [0.0] * len(raw)
        else:
            # Compute element-wise average across buffer
            arr = np.array(self.raw_buffer)
            smooth = arr.mean(axis=0).tolist()
            # Compute differences from previous
            vel, acc = self._diff_from(smooth, ts)
        return vel, acc

    def _compute_sg(self, raw, ts):
        """
        Internal method: apply Savitzky-Golay filter to raw to obtain smooth, vel, acc.

        1. Append raw to buffer.
        2. Once buffer is full, apply savgol_filter for signal, first and second derivatives.
        """
        self.sg_buffer.append(raw.copy())
        if len(self.sg_buffer) < self.sg_window:
            vel = [0.0] * len(raw)
            acc = [0.0] * len(raw)
        else:
            arr = np.array(self.sg_buffer)
            # Determine delta time for derivative scale
            dt = ts - self.prev_ts if self.prev_ts else 1.0
            # Zero-th derivative: smoothed signal
            smooth = savgol_filter(
                arr, self.sg_window, self.sg_poly,
                deriv=0, axis=0
            ).tolist()[-1]
            # First derivative: velocity
            vel = savgol_filter(
                arr, self.sg_window, self.sg_poly,
                deriv=1, delta=dt, axis=0
            ).tolist()[-1]
            # Second derivative: acceleration
            acc = savgol_filter(
                arr, self.sg_window, self.sg_poly,
                deriv=2, delta=dt, axis=0
            ).tolist()[-1]
            # Update timestamp after SG computation
            self.prev_ts = ts
        return vel, acc

    def _diff_from(self, smooth_raw, ts):
        """
        Internal helper: compute discrete differences given smoothed raw.

        Args:
            smooth_raw (list[float]): Smoothed feature vector.
            ts (float): Current timestamp.

        Returns:
            vel, acc (tuple of lists): Computed velocity and acceleration vectors.
        """
        # If no previous data, output zeros
        if self.prev_raw is None or self.prev_ts is None:
            vel = [0.0] * len(smooth_raw)
            acc = [0.0] * len(smooth_raw)
        else:
            dt = ts - self.prev_ts
            if dt <= 0:
                # Guard against zero or negative time intervals
                vel = [0.0] * len(smooth_raw)
                acc = [0.0] * len(smooth_raw)
            else:
                # First-order diff
                vel = [
                    (smooth_raw[i] - self.prev_raw[i]) / dt
                    for i in range(len(smooth_raw))
                ]
                # Second-order diff
                if self.prev_vel is None:
                    acc = [0.0] * len(smooth_raw)
                else:
                    acc = [
                        (vel[i] - self.prev_vel[i]) / dt
                        for i in range(len(smooth_raw))
                    ]
                # Do not differentiate visibility channel
                for idx in range(3, len(vel), 4):
                    vel[idx] = 0.0
                    acc[idx] = 0.0
        # Update state
        self.prev_raw = smooth_raw.copy()
        self.prev_vel = vel.copy()
        self.prev_ts = ts
        return vel, acc


def get_differentiator():
    """
    Factory function returning a Differentiator configured with project-wide defaults.

    Returns:
        Differentiator: Instance locked to the standard configuration.
    """
    cfg = _DEFAULT_DIFF_CONFIG
    return Differentiator(
        mode=cfg['mode'],
        ma_window=cfg['ma_window'],
        sg_window=cfg['sg_window'],
        sg_poly=cfg['sg_poly']
    )
