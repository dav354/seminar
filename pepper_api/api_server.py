#!/usr/bin/env python
import qi
import time
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

# Set up qi session to connect to the robot
session = qi.Session()
session.connect("tcp://127.0.0.1:9559")

# Get the ALVideoDevice service
video_service = session.service("ALVideoDevice")

# Camera setup
camera_name = "flask_cam"
resolution = 2  # 1920 x 80
color_space = 13  # BGR
fps = 30

# Subscribe to the camera
name_id = video_service.subscribeCamera(camera_name, 0, resolution, color_space, fps)

def generate():
    while True:
        image = video_service.getImageRemote(name_id)
        if image is None:
            continue

        width = image[0]
        height = image[1]
        array = image[6]

        # Convert raw data to numpy array and then JPEG
        img = np.frombuffer(array, dtype=np.uint8).reshape((height, width, 3))

        # Encode to JPEG using cv2
        import cv2
        ret, jpeg = cv2.imencode('.jpg', img)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Camera stream is available at /video_feed"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

