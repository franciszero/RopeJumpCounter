#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_gui.py

用途：
    主控界面，用于管理多个视频的标注与验证，通过弹出独立窗口进行标注。
"""

import os
import sys
import glob
import argparse
import subprocess
import PySimpleGUIQt as sg


def main():
    parser = argparse.ArgumentParser(description="RopeJumpCounter 主控界面")
    parser.add_argument('--workdir', default='../raw_videos', help='视频和标签所在目录')
    args = parser.parse_args()
    workdir = args.workdir

    # 扫描视频文件
    video_paths = sorted(
        glob.glob(os.path.join(workdir, '*.avi')) +
        glob.glob(os.path.join(workdir, '*.mp4'))
    )
    bases = [os.path.splitext(os.path.basename(p))[0] for p in video_paths]
    if not bases:
        sg.popup_error(f"未在目录找到视频: {workdir}")
        return

    # 构建左侧视频按钮与验证按钮的两列布局
    left_layout = []
    for base in bases:
        label_file = os.path.join(workdir, f"{base}_labels.csv")
        # 视频按钮底色：已标注绿色，未标注灰色
        vid_color = ('white', 'green') if os.path.exists(label_file) else ('white', 'gray')
        # 验证按钮可用性：已标注可点击，否则 disabled
        verify_enabled = os.path.exists(label_file)
        left_layout.append([
            sg.Button(base, key=f"VIDEO_{base}", button_color=vid_color, size=(20, 1)),
            sg.Button('验证', key=f"VERIFY_{base}", size=(6, 1), disabled=not verify_enabled)
        ])
    left_col = sg.Column(left_layout, scrollable=True, size=(300, 600), key='-LEFT-')

    # 右侧仅提示
    right_col = sg.Column([
        [sg.Text("请从左侧选择一个视频进行标注", font=('Helvetica', 14), key='-INFO-')]
    ], size=(600, 600), key='-RIGHT-')

    layout = [[left_col, sg.VSeparator(), right_col]]
    window = sg.Window('RopeJumpCounter 主控界面', layout, finalize=True, resizable=True)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, None):
            break

        # 点击视频按钮，弹出标注窗口
        if event and event.startswith("VIDEO_"):
            base = event.split("_", 1)[1]
            # 选择视频文件
            for ext in ('.avi', '.mp4'):
                video_file = os.path.join(workdir, f"{base}{ext}")
                if os.path.exists(video_file):
                    subprocess.Popen([sys.executable,
                                      os.path.abspath('label_helper_gui.py'),
                                      '--workdir', workdir,
                                      '--input', os.path.basename(video_file)])
                    break
            # 更新右侧显示：信息
            window['-INFO-'].update(f"视频：{base}")

        # 处理每行的验证按钮事件
        if event and event.startswith("VERIFY_"):
            base = event.split("_", 1)[1]
            label_csv = os.path.join(workdir, f"{base}_labels.csv")
            video_file = None
            for ext in ('.avi', '.mp4'):
                path = os.path.join(workdir, f"{base}{ext}")
                if os.path.exists(path):
                    video_file = path
                    break
            if video_file and os.path.exists(label_csv):
                subprocess.Popen([sys.executable,
                                  os.path.abspath('verify_labels.py'),
                                  '--video', video_file,
                                  '--labels', label_csv])

    window.close()


if __name__ == '__main__':
    main()
