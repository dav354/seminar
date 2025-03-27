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
    labels = load_labels(LABELS_PATH)
    interpreter = load_interpreter()
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"][1:3]

    cap = setup_camera()
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return

    print(f"[INFO] Running on {HARDWARE}. Press 'q' to quit.")
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Can't read frame.")
            break

        frame = cv2.flip(frame, 1)

        input_frame = cv2.resize(frame, tuple(input_shape))
        rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)

        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])[0]

        # Parse landmark output (assuming 21 keypoints, each with x, y, z)
        num_points = len(output) // 3
        landmarks = [(output[i*3], output[i*3 + 1], output[i*3 + 2]) for i in range(num_points)]

        draw_landmarks(frame, landmarks, FRAME_WIDTH, FRAME_HEIGHT)

        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        cv2.imshow(f"Gesture Recognition ({HARDWARE})", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

