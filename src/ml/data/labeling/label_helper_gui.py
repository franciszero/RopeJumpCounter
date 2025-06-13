#!/usr/bin/env python3
"""
Label_Helper_Gui data processing module

Handles data processing and feature extraction for jump rope analysis.
"""

import PySimpleGUIQt as sg  # used for creating GUI interface
import cv2  # used for video reading and frame operations
import csv  # used for writing labels CSV
import os  # used for path operations
import argparse  # used for parsing command line parameters
import numpy as np  # for memory encoding PNG
import base64  # used for Base64 encoding/decoding
import tempfile
import copy  # used for deep copying and refreshing right-side label list


# Interval merging utility
def merge_overlaps(intervals):
    """Merge Overlaps

    Performs merge overlaps operation.

    Returns:
        Result of the operation
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:  # overlap or adjacent
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


# Temporary image file path, used for GUI display
# This file is used to write each frame as PNG format for sg.Image(filename=...) loading and display

def build_panel(workdir, base):
    """Build Panel

    Performs build panel operation.

    Returns:
        Result of the operation
    """
    # This only generates a simple panel, clicking the button can start the complete labeling interface
    video_file = os.path.join(workdir, f"{base}.avi")
    if not os.path.exists(video_file):
        video_file = os.path.join(workdir, f"{base}.mp4")
    return [
        [sg.Text(f"Current video: {base}", font=('Helvetica', 14))],
        [sg.Button('Start Labeling Tool', key='-LAUNCH_LABEL_TOOL-', size=(12, 2))],
    ]


def main():
    # Parse command line parameters: working directory and input video file name
    parser = argparse.ArgumentParser(description="Jump rise segment labeling tool with buttons")
    parser.add_argument("--workdir", default="../raw_videos", help="Working directory containing video files")
    parser.add_argument("--input", default="jump_005.avi", help="Input video file name (e.g. jump.mp4)")
    args = parser.parse_args()

    # Construct input video and output CSV paths
    video_path = os.path.join(args.workdir, args.input)
    base, _ = os.path.splitext(args.input)
    output_path = os.path.join(args.workdir, f"{base}_labels.csv")

    # Initialize label data
    labels = []  # Store (start_frame, end_frame) list

    # If label file already exists, load existing labels for convenient re-editing
    if os.path.exists(output_path):
        with open(output_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append((int(row['start_frame']), int(row['end_frame'])))
        labels = merge_overlaps(labels)

    # Print and open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sg.popup_error(f"unable to open video: {video_path}")
        return

    # Get video total frame count and frame rate
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    curr_start = None  # current start frame index
    frame_idx = 0  # current displayed frame index

    # Left side video with controls, right side scrolling label list
    left_col = [
        [sg.Image(filename='', key='-IMAGE-')],
        [sg.Text('Frame: 0 / ' + str(total_frames - 1), key='-FRAME-'),
         sg.Text('Time: 0.00s', key='-TIME-'),
         sg.Text('Start: None', key='-START-')],
        [sg.Button('Prev', size=(10, 2), font=('Helvetica', 14)),
         sg.Button('Fast Prev', size=(10, 2), font=('Helvetica', 14)),
         sg.Button('Next', size=(10, 2), font=('Helvetica', 14)),
         sg.Button('Fast Next', size=(10, 2), font=('Helvetica', 14)),
         sg.Button('Mark Start', size=(10, 2), font=('Helvetica', 14)),
         sg.Button('Mark End', size=(10, 2), font=('Helvetica', 14)),
         sg.Button('Save & Quit', size=(10, 2), font=('Helvetica', 14))]
    ]

    # Right side label list (scrolling listbox), each sub-list represents one line
    label_listbox = [
        [sg.Listbox(values=[f"{s}-{e}" for s, e in labels],
                    size=(20, 20), key='-LIST-', enable_events=True)]
    ]

    # Main window layout: left side video with controls, right side label list; bottom button line
    layout = [
        [sg.Column(left_col), sg.VSeparator(),
         sg.Column(label_listbox, scrollable=True, size=(200, 400), key='-LIST_COL-')],
        [sg.Button('Goto', size=(8, 2), font=('Helvetica', 12)),
         sg.Button('Delete', size=(8, 2), font=('Helvetica', 12)),
         sg.Button('Save & Quit', size=(10, 2), font=('Helvetica', 14))]
    ]

    # Create and display window
    window = sg.Window('Jump Rise Label Helper', layout, finalize=True, return_keyboard_events=True)

    # Main loop: display frame and respond to button events
    while True:
        # Set to current frame and read
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Encode current frame as PNG → Base64 bytes, directly update
        ok, buf = cv2.imencode(".png", frame)
        if ok:
            b64_png = base64.b64encode(buf)  # bytes, not str
            window['-IMAGE-'].update(data_base64=b64_png)

        window['-FRAME-'].update(f'Frame: {frame_idx} / {total_frames - 1}')
        window['-TIME-'].update(f'Time: {frame_idx / fps:.2f}s')
        window['-START-'].update(f'Start: {curr_start if curr_start is not None else "None"}')

        # Read user operations
        event, values = window.read()

        # Save or close
        if event in (sg.WIN_CLOSED, 'Save & Quit', 'special 16777216'):
            break
        # Direction key or original button logic...
        elif event in ('Left', '<Left>', 'special 16777234'):
            # Go back one frame
            frame_idx = max(0, frame_idx - 1)
        elif event in ('Right', '<Right>', 'special 16777236'):
            # Go forward one step
            frame_idx = min(total_frames - 1, frame_idx + 1)
        elif event in ('Fast Prev', 'j', 'J'):
            frame_idx = max(0, frame_idx - 5)
        elif event in ('Fast Next', 'l', 'L'):
            frame_idx = min(total_frames - 1, frame_idx + 5)
        elif event in ('Up', '<Up>', 'special 16777235'):
            # Mark start frame
            curr_start = frame_idx
        elif event in ('Down', '<Down>', 'special 16777237'):
            # Mark end frame and save interval
            if curr_start is not None:
                # 保证 start <= end
                if frame_idx < curr_start:
                    curr_start, frame_idx = frame_idx, curr_start
                labels.append((curr_start, frame_idx))
                labels = merge_overlaps(labels)
                curr_start = None
                # Update listbox display
                window['-LIST-'].update([f"{s}-{e}" for s, e in labels])
            else:
                sg.popup('请先Mark start frames')
        elif event == 'Prev':
            # Go back one frame
            frame_idx = max(0, frame_idx - 1)
        elif event == 'Next':
            # Go forward one step
            frame_idx = min(total_frames - 1, frame_idx + 1)
        elif event == 'Mark Start':
            # Mark start frame
            curr_start = frame_idx
        elif event == 'Mark End':
            # Mark end frame and save interval
            if curr_start is not None:
                # 保证 start <= end
                if frame_idx < curr_start:
                    curr_start, frame_idx = frame_idx, curr_start
                labels.append((curr_start, frame_idx))
                labels = merge_overlaps(labels)
                curr_start = None
                # Update listbox display
                window['-LIST-'].update([f"{s}-{e}" for s, e in labels])
            else:
                sg.popup('请先Mark start frames')
        elif event == 'Goto':
            selection = values['-LIST-']
            if selection:
                s, e = map(int, selection[0].split('-'))
                frame_idx = s
        elif event == 'Delete':
            selection = values['-LIST-']
            if selection:
                s, e = map(int, selection[0].split('-'))
                # Delete this interval
                labels = [(a, b) for a, b in labels if not (a == s and b == e)]
                # Sort by start frame after deletion
                labels.sort(key=lambda x: x[0])
                window['-LIST-'].update([f"{a}-{b}" for a, b in labels])
        # Ignore other events

    # Release video and close window
    cap.release()
    window.close()

    # Save labels to CSV
    os.makedirs(args.workdir, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start_frame', 'end_frame'])
        writer.writerows(labels)

    # Show completion message
    sg.popup('Labels have been saved', f'File: {output_path}')


if __name__ == '__main__':
    main()
