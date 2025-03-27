import cv2
import time
import numpy as np
from model_config import load_interpreter, HARDWARE
from camera import setup_camera, FRAME_WIDTH, FRAME_HEIGHT
from draw import draw_landmarks

LABELS_PATH = "models/labels.txt"

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f]

def main():
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions, VisionRunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path="../model_training/exported_model/gesture_recognizer.task"),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1
    )
    recognizer = GestureRecognizer.create_from_options(options)

    cap = setup_camera()
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return

    print("[INFO] Running with custom .task model. Press 'q' to quit.")
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Can't read frame.")
            break

        frame = cv2.flip(frame, 1)

        # Inference
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        try:
            result = recognizer.recognize_async(mp_image, int(time.time() * 1000))
        except:
            result = None

        if result and result.gestures:
            gesture = result.gestures[0][0].category_name
            cv2.putText(frame, f"Gesture: {gesture}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        cv2.imshow("Gesture Recognition (MediaPipe Task)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

