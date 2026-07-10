from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent

saved_model = ROOT / "output" / "saved_model"
output_file = ROOT / "output" / "plantnet_fp16.tflite"

converter = tf.lite.TFLiteConverter.from_saved_model(
    str(saved_model)
)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_types = [
    tf.float16
]

converter.experimental_enable_resource_variables = True

print("Converting...")

tflite_model = converter.convert()

with open(output_file, "wb") as f:
    f.write(tflite_model)

print("Done!")
print(output_file)