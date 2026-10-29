from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

root = Path(r"D:\Project\OpticalSelectionMachine\OnnxDeployLab")
model_path = root / "models" / "cls" / "hezhuang5.onnx"
image_path = root / "data" / "cls_test.jpg"

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
