import ast
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

root = Path(r"D:\Project\OpticalSelectionMachine\OnnxDeployLab")
model_path = root / "models" / "cls" / "hezhuang5.onnx"
image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data" / "cls_test.jpg"

image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError(f"Failed to read image: {image_path}")
if image.shape != (224, 224, 3):
    raise ValueError(f"Expected (224,224,3), got {image.shape}")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
float_image = rgb.astype(np.float32) / np.float32(255.0)
chw = float_image.transpose(2, 0, 1)
input_data = np.ascontiguousarray(chw[None, ...])

session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

metadata = session.get_modelmeta().custom_metadata_map
print("Model metadata:")
for key, value in metadata.items():
    print(f"{key}: {value}")

names = ast.literal_eval(metadata["names"])
if not isinstance(names, dict) or set(names) != set(range(28)):
    raise ValueError("Expected class IDs 0 through 27")
labels = [names[i] for i in range(28)]
if any(not isinstance(label, str) or not label.strip()
       or "\n" in label or "\r" in label for label in labels):
    raise ValueError("Expected non-empty single-line labels")
label_path = model_path.parent / "labels.txt"
label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
print("Labels saved:", label_path)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: input_data})[0]
if output.shape != (1, 28):
    raise ValueError(f"Expected (1,28), got {output.shape}")

scores = output[0]
best_id = int(np.argmax(scores))
print("Python ORT version:", ort.__version__)
print("Input shape:", input_data.shape)
print("Output shape:", output.shape)
for class_id, score in enumerate(scores):
    print(f"Class {class_id}: {score:.9g}")
print("Best class ID:", best_id)
print("Best score:", float(scores[best_id]))
print("Best label:", labels[best_id])
