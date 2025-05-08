# 在项目根目录外任意位置：
export PYTHONPATH=/Users/francis/Documents/Georgian_College/GitHub/RopeJumpCounter

# 然后切到 PoseDetection：
cd ${PYTHONPATH}/PoseDetection

# 再运行：
python3 builder.py \
  --videos_dir ./raw_videos_3 \
  --labels_dir ./raw_videos_3 \
  --output_dir ./dataset_3 \
  --window_size 32 \
  --stride 1