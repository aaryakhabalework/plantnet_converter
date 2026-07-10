from pathlib import Path
import onnx

# Project root
ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "plantnet_mobilenetv3.onnx"

print(f"Loading model from:\n{MODEL_PATH}\n")

model = onnx.load(str(MODEL_PATH))

print("=" * 60)
print("Model Information")
print("=" * 60)

print("IR Version:", model.ir_version)

print("\nOpset:")
for opset in model.opset_import:
    print(opset)

print("\nInputs")
print("-" * 60)

for inp in model.graph.input:
    print(inp.name)

    shape = []

    for dim in inp.type.tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(dim.dim_value)
        else:
            shape.append("dynamic")

    print("Shape:", shape)
    print()

print("\nOutputs")
print("-" * 60)

for out in model.graph.output:
    print(out.name)

    shape = []

    for dim in out.type.tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(dim.dim_value)
        else:
            shape.append("dynamic")

    print("Shape:", shape)
    print()