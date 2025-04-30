# 在项目根目录外任意位置：
export PYTHONPATH=/Users/francis/Documents/Georgian_College/GitHub/RopeJumpCounter

# 然后切到 PoseDetection：
cd ${PYTHONPATH}/PoseDetection

# 再运行：
python3 pose_sequence_dataset_builder.py \
  --videos_dir ./raw_videos \
  --labels_dir ./raw_videos \
  --output_dir ./dataset \
  --window_size 32 \
  --stride 1