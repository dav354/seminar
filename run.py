import cv2
import time
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions, RunningMode

# === SETTINGS ===
MODEL_PATH = "models/gesture_recognizer.task"
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
FPS = 30

# Landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

def draw_landmarks(frame, landmarks, width, height):
    for lm_list in landmarks:
        # Draw points
        for lm in lm_list:
            cx, cy = int(lm.x * width), int(lm.y * height)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Draw lines
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            x0, y0 = lm_list[start_idx].x * width, lm_list[start_idx].y * height
            x1, y1 = lm_list[end_idx].x * width, lm_list[end_idx].y * height
            p1 = (int(x0), int(y0))
            p2 = (int(x1), int(y1))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

def main():
    options = GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO
    )
    recognizer = GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("❌ Cannot open camera.")
        exit()

    print("[INFO] Starting gesture recognition. Press 'q' to quit.")
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Can't receive frame. Exiting ...")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = recognizer.recognize_for_video(mp_image, timestamp_ms)

        if result.gestures:
            gesture = result.gestures[0][0].category_name
            score = result.gestures[0][0].score
            cv2.putText(frame, f"{gesture} ({score:.2f})", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        if result.hand_landmarks:
            draw_landmarks(frame, result.hand_landmarks, FRAME_WIDTH, FRAME_HEIGHT)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
