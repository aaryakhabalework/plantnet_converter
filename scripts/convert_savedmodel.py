from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent

INPUT_MODEL = ROOT / "output" / "plantnet_mobilenetv3_simplified.onnx"
OUTPUT_DIR = ROOT / "output" / "saved_model"

cmd = [
    sys.executable,
    "-m",
    "onnx2tf",
    "-i",
    str(INPUT_MODEL),
    "-o",
    str(OUTPUT_DIR),
]

subprocess.run(cmd, check=True)