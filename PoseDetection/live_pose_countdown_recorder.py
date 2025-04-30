"""
This script records multiple video segments from the default camera device.
Usage:
  python -m PoseDetection.recorder \
    --output_dir ./output_directory \
    --prefix filename_prefix \
    --segments number_of_segments \
    --duration duration_per_segment_in_seconds \
    --countdown countdown_seconds_before_recording \
    --width frame_width \
    --height frame_height \
    --fps frames_per_second

Main parameters:
  --output_dir: Directory to save the recorded video files (created if not exists)
  --prefix: Prefix for the output video filenames (e.g., 'jump' creates jump_001.avi, jump_002.avi, etc.)
  --segments: Total number of video segments to record
  --duration: Length of each video segment in seconds
  --countdown: Countdown time in seconds before each recording starts
  --width/--height/--fps: Video capture resolution and frame rate
"""

import cv2  # OpenCV library for video capture and processing
import os  # Operating system interface for file and directory operations
import time  # Time-related functions for countdown and duration handling
import argparse  # Command-line argument parsing


def record_segment(prefix, output_dir, seg_idx, width, height, fps, countdown, duration):
    """
    Records a single video segment from the default camera.

    Args:
        prefix (str): Filename prefix for the output video.
        output_dir (str): Directory where the video will be saved.
        seg_idx (int): Segment index used in the filename.
        width (int): Frame width for video capture.
        height (int): Frame height for video capture.
        fps (int): Frames per second for video capture.
        countdown (int): Countdown time in seconds before recording starts.
        duration (float): Duration in seconds to record the video segment.

    Behavior:
        Opens the camera, performs a countdown with overlay, records the video segment,
        and saves it to the specified output directory.
    """
    # 1. Opening and configuring the camera.
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # 2. Ensuring the output directory exists and initializing the video writer.
    os.makedirs(output_dir, exist_ok=True)
    video_filename = f"{prefix}_{seg_idx:03d}.avi"
    video_path = os.path.join(output_dir, video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # 3. Counting down before recording, with note that it displays the countdown overlay.
    for i in range(countdown, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, str(i),
                    (width // 2 - 50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 5.0,
                    (0, 0, 255), 10, cv2.LINE_AA)
        cv2.imshow("Recorder", frame)
        cv2.waitKey(1000)

    # 4. Recording loop: writing frames, displaying live feed, and checking for abort key.
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        cv2.imshow("Recorder", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 5. Cleanup: releasing resources and printing the saved path.
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved video segment: {video_path}")


# Parses command-line arguments and invokes recording for each segment
def main():
    parser = argparse.ArgumentParser(description="Record multiple video segments")
    parser.add_argument("--output_dir", required=True, help="Directory to save video files")
    parser.add_argument("--prefix", required=True, help="Filename prefix, e.g. jump")
    parser.add_argument("--segments", type=int, default=1, help="Number of segments to record")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration per segment in seconds")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown before each segment in seconds")
    parser.add_argument("--width", type=int, default=640, help="Video frame width")
    parser.add_argument("--height", type=int, default=480, help="Video frame height")
    parser.add_argument("--fps", type=int, default=30, help="Capture frames per second")
    args = parser.parse_args()

    # Iterate through segments and pass parameters to record_segment
    for i in range(1, args.segments + 1):
        record_segment(args.prefix, args.output_dir, i,
                       args.width, args.height, args.fps,
                       args.countdown, args.duration)


if __name__ == "__main__":
    main()
