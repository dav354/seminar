import cv2
import time
from camera import setup_camera, FRAME_WIDTH, FRAME_HEIGHT
from draw import HAND_CONNECTIONS
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions, RunningMode

MODEL_PATH = "models/gesture_recognizer.task"

def draw_hand_landmarks(frame, hand_landmarks, width, height):
    for landmarks in hand_landmarks:
        points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
        for x, y in points:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

def main():
    options = GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO
    )
    recognizer = GestureRecognizer.create_from_options(options)

    cap = setup_camera()
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return

    print("✅ Running gesture recognizer. Press 'q' to quit.")
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = recognizer.recognize_for_video(mp_image, timestamp)

        if result.gestures:
            gesture = result.gestures[0][0].category_name
            score = result.gestures[0][0].score
            cv2.putText(frame, f"{gesture} ({score:.2f})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        if result.hand_landmarks:
            draw_hand_landmarks(frame, result.hand_landmarks, FRAME_WIDTH, FRAME_HEIGHT)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        cv2.imshow("Gesture Recognition (.task)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
