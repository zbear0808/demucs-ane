import numpy as np
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config

## NEED TO RUN USING AMD RYZEN AI CONDA ENV I THINK

class CalibrationReader:
    def __init__(self, n=4):
        self.n = n
        self.i = 0
    def get_next(self):
        if self.i >= self.n:
            return None
        self.i += 1
        return {
            'x_spec': np.random.randn(1, 4, 2048, 336).astype(np.float32),
            'x_time': np.random.randn(1, 2, 1, 344064).astype(np.float32),
        }

quant_config = get_default_config('BFP16')
quantizer = ModelQuantizer(Config(global_quant_config=quant_config))
quantizer.quantize_model(
    model_input='htdemucs_drums_xdna2_simp.onnx',
    model_output='htdemucs_drums_xdna2_bfp16.onnx',
    calibration_data_reader=CalibrationReader(n=4),
)
print('Successfully quantized: htdemucs_drums_xdna2_bfp16.onnx')