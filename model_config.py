from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions, VisionRunningMode

MODEL_PATH = "../model_training/exported_model/gesture_recognizer.task"

def load_recognizer(model_path=MODEL_PATH):
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1
    )
    return GestureRecognizer.create_from_options(options)
