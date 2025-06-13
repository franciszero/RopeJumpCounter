"""
Live_Pose_Countdown_Recorder data processing module

Handles data processing and feature extraction for jump rope analysis.
"""

import cv2  # OpenCV library for video capture and processing
import os  # Operating system interface for file and directory operations
import time  # Time-related functions for countdown and duration handling
import argparse  # Command-line argument parsing
from datetime import datetime

from src.utils.performance.Perf import PerfStats


def record_segment(prefix, output_dir, seg_idx, width, height, fps):
    """Record Segment

    Performs record segment operation.

    Returns:
        Result of the operation
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

    stats = PerfStats(window_size=10)
    # show prompt，wait foruserpress 's' keystartrecording
    print("press 's' keystartrecording，press 'e' keystoprecording")
    while True:
        arr_ts = list()

        #
        arr_ts.append(time.time())
        ret, frame = cap.read()
        if not ret:
            break

        arr_ts.append(time.time())
        stats.update("[Main Process]: ", arr_ts)
        cv2.imshow("Recorder", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break

        time.sleep(0.02)

    # startrecording，untiluserpress 'e' stop
    while True:
        arr_ts = list()

        #
        arr_ts.append(time.time())
        ret, frame = cap.read()
        if not ret:
            break

        arr_ts.append(time.time())
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
    parser.add_argument("--output_dir", default="../data/raw_videos_3", help="Directory to save video files")
    parser.add_argument("--prefix", default="jump", help="Filename prefix, e.g. jump")
    parser.add_argument("--segments", type=int, default=1, help="Number of segments to record")
    parser.add_argument("--width", type=int, default=640, help="Video frame width")
    parser.add_argument("--height", type=int, default=480, help="Video frame height")
    parser.add_argument("--fps", type=int, default=30, help="Capture frames per second")
    args = parser.parse_args()

    # Iterate through segments and pass parameters to record_segment
    for i in range(1, args.segments + 1):
        record_segment(args.prefix, args.output_dir, i, args.width, args.height, args.fps)


if __name__ == "__main__":
    main()
