from pathlib import Path
import onnx
from onnxsim import simplify

ROOT = Path(__file__).resolve().parent.parent

INPUT_MODEL = ROOT / "models" / "plantnet_mobilenetv3.onnx"
OUTPUT_MODEL = ROOT / "output" / "plantnet_mobilenetv3_simplified.onnx"

print("Loading model...")
model = onnx.load(str(INPUT_MODEL))

print("Simplifying...")
model_simp, check = simplify(model)

if not check:
    raise RuntimeError("Simplification failed")

onnx.save(model_simp, str(OUTPUT_MODEL))

print("====================================")
print("Simplification successful!")
print(f"Saved to: {OUTPUT_MODEL}")