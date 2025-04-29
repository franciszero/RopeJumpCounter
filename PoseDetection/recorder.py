# recorder.py
"""
python -m PoseDetection.recorder \
  --output_dir ./raw_videos_2 \
  --prefix jump \
  --segments 3 \
  --duration 10 \
  --countdown 3 \
  --width 640 \
  --height 480 \
  --fps 30

•	--output_dir：保存视频的目录，脚本会自动创建
•	--prefix：文件名前缀，例如 jump，生成 jump_001.avi、jump_002.avi…
•	--segments：要录制的片段总数
•	--duration：每段录制时长（秒）
•	--countdown：每次录制前的倒计时长度（秒）
•	--width/--height/--fps：摄像头采集分辨率和帧率
"""

import cv2
import os
import csv
import time
import argparse
import glob
from utils.vision import PoseEstimator
from utils.flow import BackgroundTracker
from utils.filter import TrendFilter
from utils.detector import MultiRegionJumpDetector


def record_segment(args, seg_idx):
    # Initialize video capture for this segment
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    # Prepare video writer
    video_filename = f"{args.prefix}_{seg_idx:03d}.avi"
    video_path = os.path.join(args.output_dir, video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (args.width, args.height))

    pose = PoseEstimator()
    bg = BackgroundTracker()
    filters = {r: TrendFilter() for r in args.regions}
    detector = MultiRegionJumpDetector(args.regions)
    prev_heights = {r: None for r in args.regions}

    # Countdown before recording
    start_cd = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cd = args.countdown - (time.time() - start_cd)
        txt = f"{int(cd) + 1}" if cd > 0 else "GO!"
        cv2.putText(frame, txt,
                    (args.width // 2 - 50, args.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 5.0,
                    (0, 0, 255), 10, cv2.LINE_AA)
        cv2.imshow("Recorder", frame)
        if cv2.waitKey(1) & 0xFF == 27 or cd <= 0:
            break

    # Prepare CSV file
    filename = f"{args.prefix}_{seg_idx:03d}.csv"
    filepath = os.path.join(args.output_dir, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["frame", "timestamp"]
        header += [f"{r}_height" for r in args.regions]
        header += [f"{r}_fluct" for r in args.regions]
        header += ["jump_count"]
        writer.writerow(header)

        # Record data for the specified duration
        start_rec = time.time()
        frame_idx = 0
        while time.time() - start_rec < args.duration:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            timestamp = time.time()

            # Pose estimation
            lm, heights = pose.estimate(frame)
            if not heights:
                heights = {r: (prev_heights[r] or 0.0) for r in args.regions}

            # Background compensation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bg_dy = bg.compensate(gray)

            # Build row data
            row = [frame_idx, timestamp]
            fluctuations = {}
            for r in args.regions:
                prev = prev_heights[r]
                curr = heights.get(r, prev or 0.0)
                body_dy = 0.0 if prev is None else (curr - prev)
                prev_heights[r] = curr

                f_val = filters[r].update(body_dy - bg_dy, frame_idx)
                fluctuations[r] = f_val
                row.append(curr)
            for r in args.regions:
                row.append(fluctuations[r])

            jump_count = detector.detect(fluctuations)
            row.append(jump_count)

            writer.writerow(row)

            video_writer.write(frame)

            # Optional live display
            cv2.imshow("Recorder", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    video_writer.release()
    print(f"  Video segment saved: {video_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Segment {seg_idx:03d} saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Pose-based data recorder with segments")
    parser.add_argument("--output_dir", required=True, help="Directory to save CSV files")
    parser.add_argument("--prefix", required=True, help="Filename prefix, e.g. jump")
    parser.add_argument("--segments", type=int, default=1, help="Number of segments to record")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration per segment (seconds)")
    parser.add_argument("--countdown", type=float, default=3.0, help="Countdown before each segment (seconds)")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS")
    parser.add_argument("--regions", nargs="+", default=["head", "torso"], help="Regions to track")
    args = parser.parse_args()

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # Determine starting index based on existing files
    existing = glob.glob(os.path.join(args.output_dir, f"{args.prefix}_*.csv"))
    if existing:
        max_idx = max(int(os.path.splitext(os.path.basename(p))[0].rsplit("_", 1)[1]) for p in existing)
    else:
        max_idx = 0

    # Record each segment
    for i in range(1, args.segments + 1):
        seg_idx = max_idx + i
        record_segment(args, seg_idx)


if __name__ == "__main__":
    main()
