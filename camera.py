# camera.py
import cv2

def init_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    ret, frame = cap.read()
    if ret:
        print(f"📸 Camera resolution: {frame.shape[1]} x {frame.shape[0]}")
    else:
        print("❌ Could not access camera.")
    return cap
