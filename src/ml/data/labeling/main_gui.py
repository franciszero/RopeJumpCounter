#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main_Gui data processing module

Handles data processing and feature extraction for jump rope analysis.
"""

import os
import sys
import glob
import argparse
import subprocess
import PySimpleGUIQt as sg


def main(source_path):
    # Scan for video files
    video_paths = sorted(
        glob.glob(os.path.join(source_path, '*.avi')) +
        glob.glob(os.path.join(source_path, '*.mp4'))
    )
    bases = [os.path.splitext(os.path.basename(p))[0] for p in video_paths]
    if not bases:
        sg.popup_error(f"No videos found in directory: {source_path}")
        return

    # Build left-side layout with video buttons and verification buttons
    left_layout = []
    for base in bases:
        label_file = os.path.join(source_path, f"{base}_labels.csv")
        # Video button color: green if labeled, gray if not
        vid_color = ('white', 'green') if os.path.exists(label_file) else ('white', 'gray')
        # Verification button availability: enabled if labeled, disabled otherwise
        verify_enabled = os.path.exists(label_file)
        left_layout.append([
            sg.Button(base, key=f"VIDEO_{base}", button_color=vid_color, size=(20, 1)),
            sg.Button('Verify', key=f"VERIFY_{base}", size=(6, 1), disabled=not verify_enabled)
        ])
    left_col = sg.Column(left_layout, scrollable=True, size=(300, 600), key='-LEFT-')

    # Right side information panel
    right_col = sg.Column([
        [sg.Text("Please select a video from the left to start labeling", font=('Helvetica', 14), key='-INFO-')]
    ], size=(600, 600), key='-RIGHT-')

    layout = [[left_col, sg.VSeparator(), right_col]]
    window = sg.Window('RopeJumpCounter Main Interface', layout, finalize=True, resizable=True)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, None):
            break

        # Handle video button clicks to open labeling window
        if event and event.startswith("VIDEO_"):
            base = event.split("_", 1)[1]
            # Find the video file
            for ext in ('.avi', '.mp4'):
                video_file = os.path.join(source_path, f"{base}{ext}")
                if os.path.exists(video_file):
                    label_script = os.path.join(os.path.dirname(__file__), 'label_helper_gui.py')
                    subprocess.Popen([sys.executable,
                                      label_script,
                                      '--workdir', source_path,
                                      '--input', os.path.basename(video_file)])
                    break
            # Update right panel information
            window['-INFO-'].update(f"Video: {base}")

        # Handle verification button events
        if event and event.startswith("VERIFY_"):
            base = event.split("_", 1)[1]
            label_csv = os.path.join(source_path, f"{base}_labels.csv")
            video_file = None
            for ext in ('.avi', '.mp4'):
                path = os.path.join(source_path, f"{base}{ext}")
                if os.path.exists(path):
                    video_file = path
                    break
            if video_file and os.path.exists(label_csv):
                verify_script = os.path.join(os.path.dirname(__file__), 'verify_labels.py')
                subprocess.Popen([sys.executable,
                                  verify_script,
                                  '--video', video_file,
                                  '--labels', label_csv])

    window.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RopeJumpCounter Main Interface")
    parser.add_argument('--workdir', default='../../data/raw_videos_3', help='Directory containing videos and labels')
    args = parser.parse_args()
    workdir = args.workdir

    main(workdir)
