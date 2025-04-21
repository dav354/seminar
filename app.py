#!/usr/bin/env python3
import cv2
import time
import numpy as np
import mediapipe as mp
from flask import Flask, Response
from camera import setup_camera
from draw import draw_landmarks
from tflite_runtime.interpreter import Interpreter, load_delegate

# === Load Edge TPU Model ===
interpreter = Interpreter(
    model_path="models/model_edgetpu.tflite",
    experimental_delegates=[load_delegate("libedgetpu.so.1")]
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels must match training order
label_map = ['none', 'rock', 'paper', 'scissors']

app = Flask(__name__)
cap = setup_camera()
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)

if not cap.isOpened():
    raise RuntimeError("❌ Failed to open camera.")

def generate_frames():
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb_frame)

        gesture = "none"
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = [[lm.x, lm.y] for lm in hand_landmarks.landmark]

            # Draw landmarks on frame
            draw_landmarks(frame, landmarks, frame.shape[1], frame.shape[0])

            # Prepare model input
            coords = np.array(landmarks).flatten().astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], [coords])
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Get predicted gesture
            predicted_idx = np.argmax(output_data[0])
            gesture = label_map[predicted_idx]

        # Display gesture and FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(frame, f"Gesture: {gesture}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

@app.route("/")
def index():
    return (
        "<html><body>"
        "<h1>Gesture Recognition Stream</h1>"
        "<img src='/video' style='width:100%;height:auto;'/>"
        "</body></html>"
    )

@app.route("/video")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, threaded=True, debug=False)
