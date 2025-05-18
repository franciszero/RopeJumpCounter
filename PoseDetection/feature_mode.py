# feature_mode.py

from enum import IntFlag, auto


class Feature(IntFlag):
    RAW = 1 << 4  # 10000
    RAW_PX = 1 << 3  # 01000
    DIFF = 1 << 2  # 00100
    SPATIAL = 1 << 1  # 00010
    WINDOW = 1 << 0  # 00001


# 默认打开哪些特征（你可以改成任何组合）
_default_mode = Feature.RAW | Feature.DIFF


def get_feature_mode() -> Feature:
    """
    工厂方法：返回当前全局的 Feature 掩码
    """
    return _default_mode


def get_feature_mode_all():
    return Feature.RAW | Feature.RAW_PX | Feature.DIFF | Feature.SPATIAL | Feature.DIFF


def set_feature_mode(mode: Feature) -> None:
    """
    如果需要的话，可以在程序启动时动态修改默认掩码
    """
    global _default_mode
    _default_mode = mode


def mode_to_str(mode: Feature) -> str:
    # 把每一位按 RAW,RAW_PX,DIFF,SPATIAL,WINDOW 的顺序拼成 '10100'
    bits = []
    for flag in (Feature.RAW, Feature.RAW_PX, Feature.DIFF, Feature.SPATIAL, Feature.WINDOW):
        bits.append('1' if mode & flag else '0')
    return ''.join(bits)
