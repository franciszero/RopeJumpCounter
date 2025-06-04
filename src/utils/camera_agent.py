import cv2


def list_supported_fps(device_index=0, fps_range=range(5, 121, 5)):
    cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
    supported = set()
    for target in fps_range:
        cap.set(cv2.CAP_PROP_FPS, target)
        actual = cap.get(cv2.CAP_PROP_FPS)
        # if value read back after setting is close enough to expected, consider it supported
        if actual >= target - 0.5:
            supported.add(int(round(actual)))
    cap.release()
    return sorted(supported)


def dist_agent():
    """
    cameraSupported resolutions：
      640×480
      1280×720
      1552×1552
      1760×1328
      1920×1080
    """
    # Two ranges you want to scan, and step size
    W_RANGE = range(100, 4001, 100)
    H_RANGE = range(100, 3001, 100)
    cap = cv2.VideoCapture(0)  # Open first camera
    supported = set()

    for w in W_RANGE:
        for h in H_RANGE:
            # Request resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            # give driver time to switch
            cv2.waitKey(50)

            # actual resolution
            real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            supported.add((real_w, real_h))

    cap.release()

    # deduplicate, sort and print
    supported_list = sorted(supported)
    print("cameraSupported resolutions：")
    for w, h in supported_list:
        print(f"  {w}×{h}")


if __name__ == "__main__":
    fps_list = list_supported_fps()
    print("supportholdof fps：", fps_list)
    dist_agent()
