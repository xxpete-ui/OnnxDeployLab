from pathlib import Path

import onnxruntime as ort

model_path = Path(
    r"D:\Project\OpticalSelectionMachine\OnnxDeployLab"
    r"\models\seg\hei_huizong_250428.onnx"
)
session = ort.InferenceSession(
    str(model_path), providers=["CPUExecutionProvider"]
)
metadata = session.get_modelmeta().custom_metadata_map
for key, value in metadata.items():
    print(f"{key}: {value}")
