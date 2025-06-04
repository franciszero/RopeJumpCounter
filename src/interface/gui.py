import cv2
import time
import logging
from collections import deque
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from capture.pyav_capture import PyAVCapture
from utils.Perf import PerfStats
from ..core.video_predictor import VideoPredictor
from ..core.jump_counter import JumpCounter
from ..core.exceptions import CameraError, ModelError
from ..ml.data.features.features import FeaturePipeline

logger = logging.getLogger(__name__)

class PlayerGUI:
    """视频播放和显示界面"""

    def __init__(self, predictor: VideoPredictor, width: int, height: int, fps: int, save_path: str | None = None):
        try:
            logger.info("初始化摄像头...")
            self.cap = PyAVCapture(device_index=0, width=width, height=height, fps=fps)
        except Exception as e:
            raise CameraError(f"摄像头初始化失败: {e}")
            
        self.zoom_height = 920  # 原始 cv2 图像，高度变成 zoom_height，放大一点

        self.stats = PerfStats(window_size=10)
        self.predictor = predictor
        self.counter = JumpCounter()
        self.fps = fps

        # ---- simple FPS meter ----
        self.proc_times = deque(maxlen=30)  # ms of recent frames

        if save_path:
            try:
                save_path = Path(save_path)
                save_path.mkdir(parents=True, exist_ok=True)
                
                time_str = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
                dest_file = save_path / f"jump_{time_str}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.writer = cv2.VideoWriter(str(dest_file), fourcc, fps, (int(width), int(height)))
                
                if not self.writer.isOpened():
                    logger.error(f"VideoWriter打开失败: {dest_file}")
                    self.writer = None
                else:
                    logger.info(f"视频将保存至: {dest_file}")
            except Exception as e:
                logger.error(f"视频写入器初始化失败: {e}")
                self.writer = None
        else:
            self.writer = None

    def _overlay(self, frame: np.ndarray, jump_cnt: int, prob: float, is_on_rising: bool, t0) -> np.ndarray:
        """在 frame 上绘制概率/标签"""
        if jump_cnt is not None:
            cv2.putText(frame, f"JUMPS: {jump_cnt}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 255), 2,
                        cv2.LINE_AA)
        if prob is not None and is_on_rising:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), thickness=-1)
            alpha = 0.15
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.putText(frame, "RISING", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 255), 2,
                        cv2.LINE_AA)
        if prob is not None:
            cv2.putText(frame, f"p={prob:.2f}", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2,
                        cv2.LINE_AA)
        if self.stats.proc_fps is not None and self.stats.last_latency_ms is not None:
            txt = f"{self.stats.proc_fps:4.1f} FPS | {self.stats.last_latency_ms:3.0f} ms"
            cv2.putText(frame, txt,
                        (frame.shape[1] - 260, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
                        cv2.LINE_AA)
        return frame

    def run(self):
        logger.info("启动视频处理循环...")
        pipe = FeaturePipeline(self.cap, self.predictor.window_size)
        frame_idx = 0
        error_count = 0
        MAX_ERRORS = 5

        try:
            while True:
                try:
                    arr_ts = list()

                    arr_ts.append(time.time())
                    ret, frame, _ = self.cap.read()  # Original BGR frame
                    if not ret:
                        # error_count += 1
                        # if error_count > MAX_ERRORS:
                        #     raise CameraError("连续帧读取失败超过最大次数")
                        logger.warning(f"帧读取失败 ({error_count}/{MAX_ERRORS})")
                        continue
                    error_count = 0  # 重置错误计数

                    arr_ts.append(time.time())
                    # 1) 拉帧 + 特征抽取
                    pipe.process_frame(frame, frame_idx)
                    frame_idx += 1

                    arr_ts.append(time.time())
                    # 2) 模型推理
                    feat_vec = pd.DataFrame([pipe.fs.rec]).iloc[0][2:].values.astype(np.float32)
                    prob = self.predictor.predict(feat_vec)

                    arr_ts.append(time.time())
                    # 3) 计数处理
                    is_on_rising, jump_cnt = self.counter.process_prediction(prob, self.predictor.threshold)
                    
                    # 4) 叠加显示
                    frame_vis = self._overlay(pipe.fs.raw_frame.copy(), jump_cnt, prob, is_on_rising, arr_ts[0])

                    # 5) 显示 & 可选录制
                    cv2.imshow("JumpRope RealTime", frame_vis)
                    if self.writer:
                        self.writer.write(frame)

                    arr_ts.append(time.time())
                    # 6) 更新性能统计
                    self.stats.update("[Main Process]: ", arr_ts, 0)

                    # 7) 检查退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("用户请求退出")
                        break
                        
                except ModelError as e:
                    logger.error(f"模型错误: {e}")
                    break
                except Exception as e:
                    error_count += 1
                    if error_count > MAX_ERRORS:
                        logger.error(f"连续错误次数超过阈值: {e}")
                        break
                    logger.warning(f"处理错误 ({error_count}/{MAX_ERRORS}): {e}")
                    continue
                    
        finally:
            logger.info("清理资源...")
            self.cap.release()
            if self.writer is not None:
                self.writer.release()
            cv2.destroyAllWindows() 