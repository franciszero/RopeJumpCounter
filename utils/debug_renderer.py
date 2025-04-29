import numpy as np
import cv2


# =========================
# 5. DebugRenderer：画三条波动曲线 + 跳数
# =========================
class DebugRenderer:
    def __init__(self, frame_h, buffer_len, regions):
        self.frame_h = frame_h
        self.buffer_len = buffer_len
        self.regions = regions

    def render(self, frame, filters, jump_count):
        """
        filters: dict region→TrendFilter
        """
        h, w = frame.shape[:2]
        canvas = np.zeros((h, self.buffer_len, 3), np.uint8)
        row_h = h // len(self.regions)

        for i, r in enumerate(self.regions):
            buf = filters[r].fluct_buf
            arr = np.array(buf)
            y0, y1 = i * row_h, (i + 1) * row_h

            if len(arr) >= 2:
                mn, mx = arr.min(), arr.max()
                norm = (arr - mn) / (mx - mn) if mx > mn else np.full_like(arr, 0.5)
                pts = [(x, int(y1 - norm[x] * row_h)) for x in range(len(norm))]
                for p0, p1 in zip(pts, pts[1:]):
                    cv2.line(canvas, p0, p1, (200, 200, 200), 1)
            cv2.putText(canvas, r, (5, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 绘制跳绳计数，自动调整位置以确保完整显示
        text = f"Jumps: {jump_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4
        thickness = 15
        # 获取文本尺寸，避免超出画面
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = 10
        y = text_height + 10  # 将文本基线设置在高度 text_height + 10 处，确保完整显示
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 255), thickness)
        return cv2.hconcat([frame, canvas])
