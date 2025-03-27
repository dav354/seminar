import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python.vision import RunningMode

# === SETTINGS ===
MODEL_PATH = "models/gesture_recognizer.task"
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
FPS = 30

# Create Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO
)
recognizer = GestureRecognizer.create_from_options(options)

# Use V4L2 backend and configure camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    print("❌ Cannot open camera.")
    exit()

print("[INFO] Starting gesture recognition. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Can't receive frame. Exiting ...")
        break

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap as MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Inference with timestamp
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    result = recognizer.recognize_for_video(mp_image, timestamp_ms)

    # Show gesture label if found
    if result.gestures:
        gesture = result.gestures[0][0].category_name
        score = result.gestures[0][0].score
        cv2.putText(frame, f"{gesture} ({score:.2f})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Show the frame
    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
