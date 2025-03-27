from tflite_runtime.interpreter import Interpreter, load_delegate

# Set to "CPU" or "TPU"
HARDWARE = "CPU"

MODEL_PATHS = {
    "CPU": "models/MediaPipeHandDetector.tflite",
    "TPU": "models/model.tflite",
}

def load_interpreter():
    model_path = MODEL_PATHS[HARDWARE]
    if HARDWARE == "TPU":
        return Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate("libedgetpu.so.1")]
        )
    return Interpreter(model_path=model_path)
