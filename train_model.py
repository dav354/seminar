import os
import tensorflow as tf
import matplotlib.pyplot as plt
from mediapipe_model_maker import gesture_recognizer

# Make sure TensorFlow version is compatible
assert tf.__version__.startswith("2"), "TensorFlow 2.x required"

# === Your dataset folder ===
dataset_path = "rps_data_sample"

# === Visualize sample images per class ===
labels = []
for label in os.listdir(dataset_path):
    if os.path.isdir(os.path.join(dataset_path, label)):
        labels.append(label)

print(f"[INFO] Found classes: {labels}")

NUM_EXAMPLES = 5
for label in labels:
    label_dir = os.path.join(dataset_path, label)
    example_filenames = os.listdir(label_dir)[:NUM_EXAMPLES]
    fig, axs = plt.subplots(1, NUM_EXAMPLES, figsize=(10, 2))
    for i in range(NUM_EXAMPLES):
        image_path = os.path.join(label_dir, example_filenames[i])
        axs[i].imshow(plt.imread(image_path))
        axs[i].axis("off")
    fig.suptitle(f"Examples: {label}")
plt.show()

# === Load and preprocess dataset ===
data = gesture_recognizer.Dataset.from_folder(
    dirname=dataset_path,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)

train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

# === Train the model ===
hparams = gesture_recognizer.HParams(export_dir="exported_model")
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)

print("[INFO] Training model...")
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

# === Evaluate ===
loss, acc = model.evaluate(test_data, batch_size=1)
print(f"[INFO] Test loss: {loss:.4f}, accuracy: {acc:.4f}")

# === Export model ===
model.export_model()
print("[✅] Model exported to 'exported_model/gesture_recognizer.task'")
