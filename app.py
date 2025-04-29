import cv2
from utils.vision import PoseEstimator
from utils.flow import BackgroundTracker
from utils.filter import TrendFilter
from utils.detector import MultiRegionJumpDetector
from utils.debug_renderer import DebugRenderer


class MainApp:
    def __init__(self, regions=None):
        regions = regions or ["head", "torso"]
        self.cap = cv2.VideoCapture(0)
        _, tmp = self.cap.read()
        h, _ = tmp.shape[:2]
        self.pose = PoseEstimator()
        self.bg = BackgroundTracker()
        self.filters = {r: TrendFilter() for r in regions}
        self.detector = MultiRegionJumpDetector(regions)
        self.renderer = DebugRenderer(frame_h=h,
                                      buffer_len=self.filters[regions[0]].raw_buf.maxlen,
                                      regions=regions)
        self.prev_heights = {r: None for r in regions}

    def run(self):
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_idx += 1
            lm, heights = self.pose.estimate(frame)
            if not heights:
                heights = {r: (self.prev_heights[r] or 0.5) for r in self.prev_heights}
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bg_dy_norm = self.bg.compensate(gray)
            f_vals = {}
            for r, filt in self.filters.items():
                prev = self.prev_heights[r]
                curr = heights[r]
                body_dy = 0.0 if prev is None else (curr - prev)
                self.prev_heights[r] = curr
                f_vals[r] = filt.update(body_dy - bg_dy_norm, frame_idx)
            count = self.detector.detect(f_vals)
            output = self.renderer.render(frame, self.filters, count)
            cv2.imshow("Multi-Region JumpRope Debug", output)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    MainApp().run()
