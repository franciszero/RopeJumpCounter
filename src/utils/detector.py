import time


# =========================
# 4. MultiRegionJumpDetector：多区域同相位跳跃检测
# =========================
class MultiRegionJumpDetector:
    """
    regions: list of region names, e.g. ["head","torso","legs"]
    """

    def __init__(self, regions, min_interval=0.1):
        self.regions = regions
        self.min_interval = min_interval
        self.prev_signs = {r: -1 for r in regions}
        self.last_jump_time = 0.0
        self.count = 0

    """
    f_dict: {region: f_value}
    仅当所有 region 同时从负过零到正 且间隔足够时计数
    """

    def detect(self, f_dict):
        now = time.time()
        signs = {r: (1 if f_dict[r] > 0 else -1) for r in self.regions}

        # 判断所有区域是否都负→正
        if all(signs[r] > 0 > self.prev_signs[r] for r in self.regions):
            if (now - self.last_jump_time) > self.min_interval:
                self.count += 1
                self.last_jump_time = now

        self.prev_signs = signs
        return self.count
