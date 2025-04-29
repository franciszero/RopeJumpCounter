#!/bin/bash
python -m PoseDetection.recorder \
  --output_dir ./PoseDetection/raw_videos_2 \
  --prefix jump \
  --segments 3 \
  --duration 10 \
  --countdown 3 \
  --width 640 \
  --height 480 \
  --fps 30