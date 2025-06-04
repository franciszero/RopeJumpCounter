"""
Camera Agent Utilities

Provides utilities for detecting camera capabilities including supported
frame rates and resolutions for video capture configuration.
"""

import cv2


def list_supported_fps(device_index=0, fps_range=range(5, 121, 5)):
    """List supported frame rates for a camera device

    Tests various frame rate settings to determine which ones are
    actually supported by the camera hardware.

    Args:
        device_index: Camera device index (default: 0)
        fps_range: Range of frame rates to test (default: 5-120 in steps of 5)

    Returns:
        list: Sorted list of supported frame rates
    """
    cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
    supported = set()
    for target in fps_range:
        cap.set(cv2.CAP_PROP_FPS, target)
        actual = cap.get(cv2.CAP_PROP_FPS)
        # If value read back after setting is close enough to expected, consider it supported
        if actual >= target - 0.5:
            supported.add(int(round(actual)))
    cap.release()
    return sorted(supported)


def detect_supported_resolutions():
    """Detect camera supported resolutions

    Scans through common resolution ranges to determine which resolutions
    are supported by the camera hardware.

    Common supported resolutions:
      640×480
      1280×720
      1552×1552
      1760×1328
      1920×1080

    Returns:
        list: Sorted list of (width, height) tuples for supported resolutions
    """
    # Resolution ranges to scan with step size
    W_RANGE = range(100, 4001, 100)
    H_RANGE = range(100, 3001, 100)
    cap = cv2.VideoCapture(0)  # Open first camera
    supported = set()

    for w in W_RANGE:
        for h in H_RANGE:
            # Request resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            # Give driver time to switch
            cv2.waitKey(50)

            # Get actual resolution
            real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            supported.add((real_w, real_h))

    cap.release()

    # Deduplicate, sort and return
    return sorted(supported)


def print_camera_capabilities():
    """Print camera capabilities to console

    Displays supported frame rates and resolutions for the default camera.
    """
    fps_list = list_supported_fps()
    print("Supported frame rates:", fps_list)

    supported_resolutions = detect_supported_resolutions()
    print("Camera supported resolutions:")
    for w, h in supported_resolutions:
        print(f"  {w}×{h}")


if __name__ == "__main__":
    print_camera_capabilities()
