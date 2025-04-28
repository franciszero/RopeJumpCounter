# live_pose_countdown_recorder.py
"""
python live_pose_countdown_recorder.py \
    --output_dir ./raw_videos \
    --prefix jump \
    --segments 1 \
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
import glob
import os
import cv2
import time
import argparse
import mediapipe as mp


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_args():
    p = argparse.ArgumentParser(
        description="实时火柴人姿势+倒计时批量录制工具")
    p.add_argument("--output_dir", required=True,
                   help="保存视频文件的目录")
    p.add_argument("--prefix", required=True,
                   help="视频文件前缀，如 jump, rest 等")
    p.add_argument("--segments", type=int, default=3,
                   help="要录制多少段视频")
    p.add_argument("--duration", type=float, default=10.0,
                   help="每段录制时长（秒）")
    p.add_argument("--countdown", type=float, default=3.0,
                   help="每次录制前倒计时（秒）")
    p.add_argument("--width", type=int, default=640,
                   help="采集分辨率宽度")
    p.add_argument("--height", type=int, default=480,
                   help="采集分辨率高度")
    p.add_argument("--fps", type=int, default=30,
                   help="采集帧率")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    # MediaPipe 初始化
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Match any extension (e.g., .avi, .mp4) for existing segments
    existing = glob.glob(os.path.join(args.output_dir, f"{args.prefix}_*.*"))
    if existing:
        max_idx = max(
            int(os.path.splitext(os.path.basename(p))[0].rsplit("_", 1)[1])
            for p in existing
        )
        start_idx = max_idx + 1
    else:
        start_idx = 1

    # Record the specified number of segments, continuing numbering
    for i in range(args.segments):
        segment = start_idx + i
        # 倒计时
        start_cd = time.time()
        while True:
            ret, frame = cap.read()
            if not ret: break
            cd = args.countdown - (time.time() - start_cd)
            # 叠加火柴人
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # 大字倒计时
            if cd > 0:
                txt = f"{int(cd) + 1}"
            else:
                txt = "GO!"
            cv2.putText(frame, txt,
                        (args.width // 2 - 50, args.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 5.0,
                        (0, 0, 255), 10, cv2.LINE_AA)
            cv2.imshow("Recorder", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                return
            if cd <= 0:
                break

        # 开始录制这一段
        filename = f"{args.prefix}_{segment:03d}.avi"
        filepath = os.path.join(args.output_dir, filename)
        writer = cv2.VideoWriter(filepath, fourcc,
                                 args.fps,
                                 (args.width, args.height))
        print(f"=== Recording segment {segment}/{start_idx + args.segments - 1}: {filename}")

        start_rec = time.time()
        while True:
            ret, frame = cap.read()
            if not ret: break
            # 叠加骨架
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            writer.write(frame)
            cv2.imshow("Recorder", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            if time.time() - start_rec >= args.duration:
                break

        writer.release()
        print(f"=== Finished segment {segment}, duration {args.duration}s")

    # 收工
    cap.release()
    cv2.destroyAllWindows()
    print("All segments recorded. Goodbye!")


if __name__ == "__main__":
    main()
