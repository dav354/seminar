import cv2

CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
FPS = 60


def setup_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        raise RuntimeError(
            "❌ Could not open camera. Check your device index or drivers."
        )

    print("[📷] Camera ready")
    print(
        f"    Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
    )
    print(f"    FPS: {cap.get(cv2.CAP_PROP_FPS)}")

    return cap
