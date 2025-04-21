#!/usr/bin/env python3
import cv2
import time
import mediapipe as mp
from flask import Flask, Response
from camera import setup_camera
from draw import draw_landmarks

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    GestureRecognizer,
    GestureRecognizerOptions,
    VisionRunningMode,
)

app = Flask(__name__)
cap = setup_camera()

if not cap.isOpened():
    raise RuntimeError("❌ Failed to open camera.")

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="models/gesture_recognizer.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
)
recognizer = GestureRecognizer.create_from_options(options)

def generate_frames():
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        try:
            result = recognizer.recognize_async(mp_image, int(time.time() * 1000))
        except Exception:
            result = None

        if result and result.gestures:
            gesture = result.gestures[0][0].category_name
            landmarks = [
                (lm.x, lm.y) for lm in result.hand_landmarks[0]
            ] if result.hand_landmarks else []

            draw_landmarks(frame, landmarks, frame.shape[1], frame.shape[0])

            cv2.putText(
                frame, f"Gesture: {gesture}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2
            )

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(
            frame, f"FPS: {fps:.2f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2
        )

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
