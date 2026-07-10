from pathlib import Path
import onnxruntime as ort
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "plantnet_mobilenetv3.onnx"

session = ort.InferenceSession(
    str(MODEL),
    providers=["CPUExecutionProvider"]
)

print("=" * 60)
print("Inputs")
print("=" * 60)

for i in session.get_inputs():
    print(f"Name : {i.name}")
    print(f"Shape: {i.shape}")
    print(f"Type : {i.type}")
    print()

print("=" * 60)
print("Outputs")
print("=" * 60)

for o in session.get_outputs():
    print(f"Name : {o.name}")
    print(f"Shape: {o.shape}")
    print(f"Type : {o.type}")
    print()

dummy = np.random.rand(1, 3, 224, 224).astype(np.float32)

outputs = session.run(
    None,
    {"image": dummy}
)

print("=" * 60)
print("Inference")
print("=" * 60)

print("Output shape:", outputs[0].shape)
print("Min:", outputs[0].min())
print("Max:", outputs[0].max())