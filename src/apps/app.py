# """
# app.py
#
# main control script：based on pose estimation and deep model for real-time jump rope action detection.
#
# functions:
# 1. open camera, read video frames in real-time.
# 2. use MediaPipe Pose to extract human landmarks and perform background shake compensation.
# 3. build features (landmark heights, trend filtering, etc.) and feed to deep learning model for jump detection.
# 4. real-time calculate rising edge jump count and write to CSV file.
# 5. can define output directory, countdown time, model path etc. via command line parameters.
#
# Usage:
# python app.py --out <outputdirectory> [--countdown N] [--model modelfilepath]
#     python app.py --out output_dir --countdown 3
#     python app.py --out output_dir --countdown 3 --model models/lstm_jump_classifier.h5
#
# Parameter description:
# --out       camera data and result save path (default: record_output)
# --countdown countdown seconds before recording (default: 3)
# --model     jump detection model file, supports .keras or .h5 format (default: models/best_crnn.keras)
# """
#
# import cv2
# import os
# import csv
# import time
# import argparse
# import sys
# from utils.vision import PoseEstimator
# from utils.flow import BackgroundTracker
# from utils.filter import TrendFilter
#
# import tensorflow as tf
# import numpy as np
# from collections import deque
#
# from src.ml.data.features import PoseFrame, Differentiator, DistanceCalculator, AngleCalculator
#
# import logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# logger = logging.getLogger(__name__)
#
#
# def record_session(output_dir, regions=None, countdown=3, model_path='src/ml/models/lstm_jump_classifier.h5'):
# # Create output directory
#     regions = regions or ["head", "torso"]
#     os.makedirs(output_dir, exist_ok=True)
#
# # Initialize video capture and model modules
#     # cap = cv2.VideoCapture(0)
#     pose = PoseEstimator()
#     bg = BackgroundTracker()
#     filters = {r: TrendFilter() for r in regions}
#     prev_heights = {r: None for r in regions}
#
# # load jump rope action recognition model, and decide whether to use window mode or single frame mode
#     if not model_path.endswith(('.keras', '.h5')):
#         model_path += '.keras'
#     model = tf.keras.models.load_model(model_path)
#     logger.debug(f"Loaded model from: {model_path}")
#
# # Decide whether to use window mode or single frame mode based on model.input_shape
#     input_shape = model.input_shape  # e.g. (None, W, F) or (None, F)
#     if len(input_shape) == 3 and input_shape[1] is not None:
#         window_size = int(input_shape[1])
#         use_window = window_size > 1
#     else:
#         window_size = 1
#         use_window = False
#     logger.debug(f"Model input_shape: {input_shape}, window_size: {window_size}, use_window: {use_window}")
#
# # === Feature extraction initialization ===
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     dt = 1.0 / fps
#     diff = Differentiator()
#     dist_calc = DistanceCalculator()
#     ang_calc = AngleCalculator()
#     feature_buffer = deque(maxlen=window_size)
#     prev_pred = 0
#
# # Countdown, prompt user to prepare
#     for i in range(countdown, 0, -1):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         cv2.putText(
#             frame,
#             f"Starting in {i}",
#             (50, 50),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             2,
#             (0, 0, 255),
#             3,
#         )
#         cv2.imshow("Recorder", frame)
#         cv2.waitKey(1000)
#
# # Create CSV file and write table header
#     csv_path = os.path.join(output_dir, "data.csv")
#     with open(csv_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         header = ["frame", "timestamp"]
#         header += [f"{r}_height" for r in regions]
#         header += [f"{r}_fluct" for r in regions]
#         header += ["jump_count"]
#         writer.writerow(header)
#
#         frame_idx = 0
#         prev_timestamp = None
#         jump_count = 0  # total jumps so far
# # Main loop: read video frame, estimate pose, compensate background shake motion, build features, model inference, detect jumps
#         logger.debug("Starting recording loop")
#         while True:
#             ret, frame = cap.read()
#             logger.debug(f"Read frame {frame_idx + 1}, ret={ret}")
#             if not ret:
#                 break
#             frame_idx += 1
#             timestamp = time.time()
#
#             # Compute FPS
#             if prev_timestamp is None:
#                 fps_display = 0.0
#             else:
#                 fps_display = 1.0 / (timestamp - prev_timestamp)
#             prev_timestamp = timestamp
#
#             logger.debug(f"Timestamp: {timestamp:.3f}, FPS display: {fps_display:.1f}")
#
# # Pose estimation, get landmark heights
#             lm, heights = pose.estimate(frame)
#             logger.debug(f"Pose landmarks: {'detected' if lm else 'none'}, heights: {heights}")
#             if not heights:
#                 heights = {r: prev_heights[r] or 0.0 for r in regions}
#
# # Background shake motion compensation, compute background displacement
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             bg_dy = bg.compensate(gray)
#             logger.debug(f"Background displacement dy: {bg_dy:.3f}")
#
# # Build data row: frame number, timestamp
#             row = [frame_idx, timestamp]
#
# # === Full 469-dimensional feature extraction ===
#             lm, heights = pose.estimate(frame)
# # Get frame dimensions
#             height, width = frame.shape[:2]
# # If no person detected, fill with zero values
#             if lm is None:
#                 raw = [0.0] * (33 * 4)
#                 raw_px = [0.0] * (33 * 2)
#                 vel = [0.0] * (33 * 4)
#                 acc = [0.0] * (33 * 4)
#                 dists = [0.0] * len(dist_calc.pairs)
#                 angs = [0.0] * len(ang_calc.triplets)
#
#                 pred = 0.0
#                 label = 0
#                 # skip model inference when no landmarks detected
#                 jump_flag = False
#             else:
#                 logger.debug(f"Raw 132-dim, raw_px 66-dim, computing vel/acc/dist/ang")
#                 pf = PoseFrame(frame_idx, timestamp, lm.landmark, frame_size=(height, width))
#                 raw = pf.raw  # 132 dims
#                 raw_px = pf.raw_px  # 66 dims
#                 vel, acc = diff.compute(raw)
#                 dists = dist_calc.compute(lm.landmark)
#                 angs = ang_calc.compute(lm.landmark)
#
# # Concatenate feature vectors to match model expectations
#                 feat = raw + raw_px + vel + acc + dists + angs  # 132+66+132+132+4+3 = 469 dims
#
# # Model inference and jump counting
#                 if use_window:
#                     feature_buffer.append(feat)
#                     if len(feature_buffer) == window_size:
#                         inp = np.stack(feature_buffer, axis=0)[np.newaxis, ...]
#                         pred = model.predict(inp, verbose=0)[0, 0]
#                         label = 1 if pred > 0.5 else 0
#                     else:
#                         pred = 0.0
#                         label = 0
#                 else:
#                     inp = np.array(feat, dtype=np.float32)[np.newaxis, np.newaxis, :]
#                     pred = model.predict(inp, verbose=0)[0, 0]
#                     label = 1 if pred > 0.5 else 0
#
#                 logger.debug(f"Model prediction: {pred:.3f}, label: {label}")
#
# # Detect rising edge, accumulate jump rope count
#                 jump_flag = (prev_pred == 0 and label == 1)
#
#             if jump_flag:
#                 jump_count += 1
#             logger.debug(f"Jump flag: {jump_flag}, total jump_count: {jump_count}")
#
#             row.append(1 if jump_flag else 0)
#             prev_pred = label
#
#             logger.debug(f"Writing CSV row: {row}")
# # Write to CSV file
#             writer.writerow(row)
#
#             # Overlay debug info
#             debug_texts = [
#                 f"FPS: {fps_display:.1f}",
#                 f"P(jump): {pred:.2f}",
#                 f"Jump Count: {jump_count}",
#             ]
#             y0, dy = 100, 100
#             for i, txt in enumerate(debug_texts):
#                 y = y0 + i * dy
#                 cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
# # Display camera image
#             cv2.imshow("Recorder", frame)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break
#
# # Cleaning up resources, close camera and windows
#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"Recording complete. Data saved to: {csv_path}")
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Pose-based data recorder")
#     parser.add_argument("--out", default="record_output", help="Output directory for data")
#     parser.add_argument("--countdown", type=int, default=3, help="Countdown seconds before recording starts")
#     parser.add_argument(
#         '--model',
#         default='src/ml/models/best_crnn.keras',
#         help='Path to .keras or .h5 model file (default: src/ml/models/best_crnn.keras)'
#     )
#     args = parser.parse_args()
#     record_session(args.out, countdown=args.countdown, model_path=args.model)
