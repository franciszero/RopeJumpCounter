#!/usr/bin/env python3
"""
label_helper_gui.py

用途：
    提供一个图形化界面，用按钮标注视频中的“跳跃上升段”起始和结束帧，
    并导出标签 CSV。

环境要求：
    - Python 3.x
    - 安装依赖：
        pip install opencv-python PySimpleGUIQt PySide6

输入：
    --workdir: 工作目录，包含视频文件
    --input:   输入视频文件名（支持 AVI、MP4 等格式）

输出：
    在工作目录下生成 <输入视频名>_labels.csv，
    列名：start_frame,end_frame，每行记录一个标注区间。

调用方式：
    python label_helper_gui.py --workdir ../raw_videos --input jump_001.avi

界面说明：
    - 视频显示区：当前帧画面
    - 按钮：
        Prev         ：回退一帧
        Next         ：前进一步
        Mark Start   ：标记上升段起始帧
        Mark End     ：标记上升段结束帧并保存区间
        Save & Quit  ：保存标签并退出
    - 状态栏：实时显示当前帧索引、时间戳、当前起始帧（如果已标记）

注意事项：
    - 退出后 CSV 会自动生成，若需继续标注，可手动编辑该 CSV。
    - 确保视频帧率一致，以正确计算时间戳。
"""

import PySimpleGUIQt as sg  # 用于创建 Qt GUI 界面
import cv2  # 用于视频读取和帧操作
import csv  # 用于写入标签 CSV
import os  # 用于路径操作
import argparse  # 用于解析命令行参数
import tempfile
import copy  # 用于深拷贝和刷新右侧标注列表

# 临时图像文件路径，用于 GUI 显示
# 该文件用于将每帧写为 PNG 格式，供 sg.Image(filename=...) 加载显示

def main():
    # 解析命令行参数：工作目录和输入视频文件名
    parser = argparse.ArgumentParser(description="带按钮的跳跃上升段标注工具")
    parser.add_argument("--workdir", default="../raw_videos", help="工作目录，包含视频文件")
    parser.add_argument("--input", default="jump_005.avi", help="输入视频文件名（如 jump.mp4）")
    args = parser.parse_args()

    # 构造输入视频和输出 CSV 路径
    video_path = os.path.join(args.workdir, args.input)
    base, _ = os.path.splitext(args.input)
    output_path = os.path.join(args.workdir, f"{base}_labels.csv")

    tmp_img_path = os.path.join(args.workdir, f"{base}_tmp.png")

    # 初始化标注数据
    labels = []  # 存储 (start_frame, end_frame) 列表

    # 如果已有标签文件，加载已有标注以便重编辑
    if os.path.exists(output_path):
        with open(output_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append((int(row['start_frame']), int(row['end_frame'])))
        # 将加载的标签按起始帧排序
        labels.sort(key=lambda x: x[0])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sg.popup_error(f"无法打开视频: {video_path}")
        return

    # 获取视频总帧数和帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    curr_start = None  # 当前起始帧索引
    frame_idx = 0  # 当前显示的帧索引

    # 左侧视频与操作，右侧滚动标注列表
    left_col = [
        [sg.Image(filename='', key='-IMAGE-')],
        [sg.Text('Frame: 0 / ' + str(total_frames - 1), key='-FRAME-'),
         sg.Text('Time: 0.00s', key='-TIME-'),
         sg.Text('Start: None', key='-START-')],
        [sg.Button('Prev', size=(10,2), font=('Helvetica',14)),
         sg.Button('Next', size=(10,2), font=('Helvetica',14)),
         sg.Button('Mark Start', size=(10,2), font=('Helvetica',14)),
         sg.Button('Mark End', size=(10,2), font=('Helvetica',14)),
         sg.Button('Save & Quit', size=(10,2), font=('Helvetica',14))]
    ]

    # 右侧标签列表（滚动列表框），每个子列表代表一行
    label_listbox = [
        [sg.Listbox(values=[f"{s}-{e}" for s,e in labels],
                    size=(20,20), key='-LIST-', enable_events=True)]
    ]

    # 主窗口布局：左侧视频与控制，右侧标签列表；底部按钮行
    layout = [
        [sg.Column(left_col), sg.VSeparator(),
         sg.Column(label_listbox, scrollable=True, size=(200,400), key='-LIST_COL-')],
        [sg.Button('Goto', size=(8,2), font=('Helvetica',12)),
         sg.Button('Delete', size=(8,2), font=('Helvetica',12)),
         sg.Button('Save & Quit', size=(10,2), font=('Helvetica',14))]
    ]

    # 创建并显示窗口
    window = sg.Window('Jump Rise Label Helper', layout, finalize=True, return_keyboard_events=True)

    # 主循环：显示帧并响应按钮事件
    while True:
        # 定位到当前帧并读取
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # 将当前帧写入临时文件，用于 GUI 显示
        cv2.imwrite(tmp_img_path, frame)
        # 更新界面元素：图像（通过文件）、帧信息、时间戳、起始帧
        window['-IMAGE-'].update(filename=tmp_img_path)

        window['-FRAME-'].update(f'Frame: {frame_idx} / {total_frames - 1}')
        window['-TIME-'].update(f'Time: {frame_idx / fps:.2f}s')
        window['-START-'].update(f'Start: {curr_start if curr_start is not None else "None"}')

        # 读取用户操作
        event, values = window.read()

        # 保存或关闭
        if event in (sg.WIN_CLOSED, 'Save & Quit', 'special 16777216'):
            break
        # 方向键或原按钮逻辑...
        elif event in ('Left', '<Left>', 'special 16777234'):
            # 回退一帧
            frame_idx = max(0, frame_idx - 1)
        elif event in ('Right', '<Right>', 'special 16777236'):
            # 前进一步
            frame_idx = min(total_frames - 1, frame_idx + 1)
        elif event in ('Up', '<Up>', 'special 16777235'):
            # 标记起始帧
            curr_start = frame_idx
        elif event in ('Down', '<Down>', 'special 16777237'):
            # 标记结束帧并保存区间
            if curr_start is not None:
                labels.append((curr_start, frame_idx))
                # 按起始帧排序
                labels.sort(key=lambda x: x[0])
                curr_start = None
                # 更新列表框显示
                window['-LIST-'].update([f"{s}-{e}" for s,e in labels])
            else:
                sg.popup('请先标记起始帧')
        elif event == 'Prev':
            # 回退一帧
            frame_idx = max(0, frame_idx - 1)
        elif event == 'Next':
            # 前进一步
            frame_idx = min(total_frames - 1, frame_idx + 1)
        elif event == 'Mark Start':
            # 标记起始帧
            curr_start = frame_idx
        elif event == 'Mark End':
            # 标记结束帧并保存区间
            if curr_start is not None:
                labels.append((curr_start, frame_idx))
                # 按起始帧排序
                labels.sort(key=lambda x: x[0])
                curr_start = None
                # 更新列表框显示
                window['-LIST-'].update([f"{s}-{e}" for s,e in labels])
            else:
                sg.popup('请先标记起始帧')
        elif event == 'Goto':
            selection = values['-LIST-']
            if selection:
                s,e = map(int, selection[0].split('-'))
                frame_idx = s
        elif event == 'Delete':
            selection = values['-LIST-']
            if selection:
                s,e = map(int, selection[0].split('-'))
                # 删除该区间
                labels = [(a,b) for a,b in labels if not (a==s and b==e)]
                # 删除后按起始帧排序
                labels.sort(key=lambda x: x[0])
                window['-LIST-'].update([f"{a}-{b}" for a,b in labels])
        # 其它事件忽略

    # 释放视频和关闭窗口
    cap.release()
    window.close()

    # 清理临时帧图像文件
    if os.path.exists(tmp_img_path):
        os.remove(tmp_img_path)

    # 保存标签到 CSV
    os.makedirs(args.workdir, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start_frame', 'end_frame'])
        writer.writerows(labels)

    # 提示完成
    sg.popup('标注已保存', f'文件: {output_path}')


if __name__ == '__main__':
    main()
