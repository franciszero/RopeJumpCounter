"""
Video capture module

Provides various video capture implementations for different performance
and compatibility requirements.
"""

from .gst_capture import GStreamerCapture
from .pyav_capture import PyAVCapture

__all__ = ['GStreamerCapture', 'PyAVCapture']