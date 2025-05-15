import cv2

# 你想扫的两个区间，以及步长
W_RANGE = range(100, 4001, 100)
H_RANGE = range(100, 3001, 100)

"""
摄像头支持的分辨率：
  640×480
  1280×720
  1552×1552
  1760×1328
  1920×1080
"""

cap = cv2.VideoCapture(0)  # 打开第一个摄像头
supported = set()

for w in W_RANGE:
    for h in H_RANGE:
        # 请求分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # 给驱动一点时间去切换
        cv2.waitKey(50)

        # 真实返回的分辨率
        real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        supported.add((real_w, real_h))

cap.release()

# 去重后排序、打印
supported_list = sorted(supported)
print("摄像头支持的分辨率：")
for w, h in supported_list:
    print(f"  {w}×{h}")