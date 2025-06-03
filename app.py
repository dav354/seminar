#!/usr/bin/env python3
import cv2
import time
import numpy as np
import psutil
import mediapipe as mp
from flask import Flask, Response, render_template, jsonify
from draw import draw_landmarks
from tflite_runtime.interpreter import Interpreter, load_delegate
from collections import deque
from game_logic import game_state, play_round, reset_game
from gesture_buffer import GestureCollector
import threading

# === Configuration ===
PEPPER_IP="http://192.168.3.156:5000"
label_map = ["none", "rock", "paper", "scissors"]
gesture_collector = GestureCollector(duration=2.0)
PREDICTION_HISTORY = deque(maxlen=5)
log_messages = []
latest_stats = {
    "gesture": "unknown",
    "confidence": 0.0,
    "fps": 0.0,
    "cpu": 0.0,
    "ram": "0MB / 0MB",
    "inference_ms": 0.0,
    "cpu_temp": 0.0,
}

# === Load Edge TPU Model ===
interpreter = Interpreter(
    model_path="models/model_edgetpu.tflite",
    experimental_delegates=[load_delegate("libedgetpu.so.1")],
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Setup Camera and Hand Detection ===
cap = cv2.VideoCapture(f"{PEPPER_IP}/video_feed")
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)

# === Flask App ===
app = Flask(__name__)


def log(msg: str):
    print(msg)
    log_messages.append(msg)
    if len(log_messages) > 100:
        log_messages.pop(0)


def get_system_stats():
    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    ram_str = f"{vm.used // (1024*1024)}MB / {vm.total // (1024*1024)}MB"

    cpu_temp = 0.0
    temps = psutil.sensors_temperatures()
    for label in ("cpu_thermal", "cpu-thermal", "coretemp"):
        if label in temps and temps[label]:
            cpu_temp = temps[label][0].current
            break

    return cpu, ram_str, round(cpu_temp, 1)


def run_inference(coords):
    coords -= coords.mean(axis=0)
    s = np.max(np.abs(coords)) or 1
    coords /= s
    inp = coords.flatten().astype(np.float32)

    t0 = time.time()
    in_scale, in_zero_point = input_details[0]["quantization"]
    quantized_input = (inp / in_scale + in_zero_point).astype(np.uint8)
    interpreter.set_tensor(input_details[0]["index"], [quantized_input])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]
    out_scale, out_zero_point = output_details[0]["quantization"]
    dequantized = (output.astype(np.float32) - out_zero_point) * out_scale
    infer_ms = (time.time() - t0) * 1000

    return dequantized, infer_ms


def generate_frames():
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            log("[⚠️] Frame capture failed. Trying to reconnect...")
            cap.release()
            cap.open(f"{PEPPER_IP}/video_feed")
            time.sleep(1)
            continue
        
        frame = cv2.flip(frame, 1)
        gesture = "unknown"
        confidence = 0.0
        infer_ms = 0.0

        if gesture_collector.collecting:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                pts = [[p.x, p.y] for p in lm.landmark]
                draw_landmarks(frame, pts, frame.shape[1], frame.shape[0])

                coords = np.array(pts)
                dequantized, infer_ms = run_inference(coords)

                PREDICTION_HISTORY.append(dequantized)
                avg_pred = np.mean(PREDICTION_HISTORY, axis=0)
                idx = int(np.argmax(avg_pred))
                conf = float(avg_pred[idx])

                if conf > 0.7:
                    gesture = label_map[idx]
                confidence = conf
                gesture_collector.add_gesture(gesture)

            if gesture_collector.is_done():
                final_gesture = gesture_collector.get_most_common()
                log(f"[🧠] Most common gesture: {final_gesture}")
                play_round(final_gesture)
                gesture_collector.reset()

        # Systemdaten (immer aktuell)
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        cpu, ram_str, cpu_temp = get_system_stats()

        latest_stats.update({
            "gesture": gesture,
            "confidence": round(confidence, 2),
            "fps": round(fps, 1),
            "cpu": cpu,
            "ram": ram_str,
            "inference_ms": round(infer_ms, 1),
            "cpu_temp": cpu_temp,
        })

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            log("[⚠️] JPEG encoding failed.")
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )
        # === FPS-Limiter: max 10 FPS ===
        time.sleep(0.1)

# === Routes ===

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/gesture_data")
def gesture_data():
    return jsonify(latest_stats)


@app.route("/logs")
def logs():
    return jsonify(log_messages)


@app.route("/play_round")
def play_game_round():
    player_gesture = latest_stats["gesture"]
    result = play_round(player_gesture)
    return jsonify(result)

@app.route("/start_round")
def start_round():
    if not gesture_collector.collecting:
        gesture_collector.start()
        log("[🎬] Round started: collecting gestures")
        return jsonify({"status": "collecting"})
    else:
        return jsonify({"status": "already collecting"})

@app.route("/game_state")
def get_game_state():
    return jsonify(game_state)

@app.route("/reset_game")
def reset():
    reset_game()
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, threaded=True, debug=False)
