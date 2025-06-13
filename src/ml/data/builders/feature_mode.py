"""
Feature_Mode data processing module

Handles data processing and feature extraction for jump rope analysis.
"""

# feature_mode.py

from enum import IntFlag


class Feature(IntFlag):
    RAW = 1 << 4  # 10000
    RAW_PX = 1 << 3  # 01000
    DIFF = 1 << 2  # 00100
    SPATIAL = 1 << 1  # 00010
    WINDOW = 1 << 0  # 00001


# Default enabled features (can be changed to any combination)
_default_mode = Feature.RAW | Feature.DIFF


def get_feature_mode() -> Feature:
    """Get Feature Mode

    Performs get feature mode operation.

    Returns:
        Result of the operation
    """
    # return Feature.RAW | Feature.RAW_PX | Feature.DIFF | Feature.SPATIAL | Feature.WINDOW
    return _default_mode


def get_feature_mode_all():
    return Feature.RAW | Feature.RAW_PX | Feature.DIFF | Feature.SPATIAL | Feature.WINDOW


def set_feature_mode(mode: Feature) -> None:
    """Get Feature Mode All

    Performs get feature mode all operation.

    Returns:
        Result of the operation
    """
    global _default_mode
    _default_mode = mode


def mode_to_str(mode: Feature) -> str:
    # Convert each bit to string in order: RAW,RAW_PX,DIFF,SPATIAL,WINDOW -> '10100'
    bits = []
    for flag in (Feature.RAW, Feature.RAW_PX, Feature.DIFF, Feature.SPATIAL, Feature.WINDOW):
        bits.append('1' if mode & flag else '0')
    return ''.join(bits)
