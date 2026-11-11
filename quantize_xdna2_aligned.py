import numpy as np
import onnx
from onnxsim import simplify
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config

# 1. Simplify ONNX graph to fuse constant shapes and padding math
model = onnx.load("htdemucs_drums_xdna2_aligned_fp32.onnx")
model_simp, check = simplify(model)
assert check, "ONNX Simplifier check failed"
onnx.save(model_simp, "htdemucs_drums_xdna2_aligned_simp.onnx")

# 2. Calibration Data Reader (uses 10 random or real STFT/Waveform chunks)
class DemucsCalibrationReader:
    def __init__(self, num_samples=8):
        self.num_samples = num_samples
        self.idx = 0

    def get_next(self):
        if self.idx >= self.num_samples:
            return None
        self.idx += 1
        return {
            "x_spec": np.random.randn(1, 4, 2048, 336).astype(np.float32),
            "x_time": np.random.randn(1, 2, 1, 344064).astype(np.float32),
        }

# 3. Select XDNA 2 Floating-Point Config ("BF16" or "BFP16")
# BFP16 (Block FP16) gives ~2x higher MAC throughput than BF16 on XDNA 2
# while avoiding INT8 phase distortion!
quant_config = get_default_config("BFP16")
config = Config(global_quant_config=quant_config)

quantizer = ModelQuantizer(config)
quantizer.quantize_model(
    model_input="htdemucs_drums_xdna2_aligned_simp.onnx",
    model_output="htdemucs_drums_xdna2_aligned_bfp16.onnx",
    calibration_data_reader=DemucsCalibrationReader(num_samples=8),
)
print("Saved aligned XDNA 2 BFP16 model: htdemucs_drums_xdna2_aligned_bfp16.onnx")
