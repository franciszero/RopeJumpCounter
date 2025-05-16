"""
This script records multiple video segments from the default camera device.
Usage:
  python -m PoseDetection.recorder \
    --output_dir ./raw_video_2 \
    --prefix jump \
    --segments number_of_segments \
    --width frame_width \
    --height frame_height \
    --fps frames_per_second

Main parameters:
  --output_dir: Directory to save the recorded video files (created if not exists)
  --prefix: Prefix for the output video filenames (e.g., 'jump' creates jump_001.avi, jump_002.avi, etc.)
  --segments: Total number of video segments to record
  --width/--height/--fps: Video capture resolution and frame rate
"""

import cv2  # OpenCV library for video capture and processing
import os  # Operating system interface for file and directory operations
import time  # Time-related functions for countdown and duration handling
import argparse  # Command-line argument parsing
from datetime import datetime

from utils.Perf import PerfStats


def record_segment(prefix, output_dir, seg_idx, width, height, fps):
    """
    Records a single video segment from the default camera.

    Args:
        prefix (str): Filename prefix for the output video.
        output_dir (str): Directory where the video will be saved.
        seg_idx (int): Segment index used in the filename.
        width (int): Frame width for video capture.
        height (int): Frame height for video capture.
        fps (int): Frames per second for video capture.

    Behavior:
        Opens the camera, waits for user to start recording by pressing 's',
        records until user presses 'e', and saves the video segment.
    """
    # 1. Opening and configuring the camera.
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # 2. Ensuring the output directory exists and initializing the video writer.
    os.makedirs(output_dir, exist_ok=True)
    # Use timestamp to avoid filename collisions
    time_str = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    video_filename = f"{prefix}_{time_str}.avi"
    video_path = os.path.join(output_dir, video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # 显示提示，等待用户按 's' 键启动录制
    print("按 's' 键开始录制，按 'e' 键停止录制")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Recorder", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break

    stats = PerfStats(window_size=10)
    # 开始录制，直到用户按 'e' 停止
    while True:
        arr_ts = list()

        #
        arr_ts.append(time.time())
        ret, frame = cap.read()
        if not ret:
            break

        arr_ts.append(time.time())
        # 5) 更新性能统计
        stats.update("[Main Process]: ", arr_ts)

        #
        writer.write(frame)
        cv2.imshow("Recorder", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            break

    # 5. Cleanup: releasing resources and printing the saved path.
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved video segment: {video_path}")


# Parses command-line arguments and invokes recording for each segment
def main():
    parser = argparse.ArgumentParser(description="Record multiple video segments")
    parser.add_argument("--output_dir", default="./raw_videos_3", help="Directory to save video files")
    parser.add_argument("--prefix", default="jump", help="Filename prefix, e.g. jump")
    parser.add_argument("--segments", type=int, default=1, help="Number of segments to record")
    parser.add_argument("--width", type=int, default=640, help="Video frame width")
    parser.add_argument("--height", type=int, default=480, help="Video frame height")
    parser.add_argument("--fps", type=int, default=30, help="Capture frames per second")
    args = parser.parse_args()

    # Iterate through segments and pass parameters to record_segment
    for i in range(1, args.segments + 1):
        record_segment(args.prefix, args.output_dir, i,
                       args.width, args.height, args.fps)


if __name__ == "__main__":
    main()
