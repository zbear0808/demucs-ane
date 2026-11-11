import os
import time
import glob
import json
import numpy as np
import onnxruntime as ort

os.environ["VAIP_VERBOSITY"] = "1"
os.environ["XLNX_ONNX_EP_VERBOSE"] = "1"
os.environ["XLNX_ENABLE_DUMP_XIR_MODEL"] = "1"
os.environ["XLNX_ENABLE_DUMP"] = "1"

VAIP_CONFIG = r"C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0\Lib\site-packages\misc\vaip_config.json"
cache_dir = os.path.abspath("./xdna2_cache")

print("--- Testing Candidate 1: htdemucs_drums_xdna2_bfp16_vai.onnx ---", flush=True)
so = ort.SessionOptions()
so.log_severity_level = 1

try:
    sess = ort.InferenceSession(
        "htdemucs_drums_xdna2_bfp16_vai.onnx",
        sess_options=so,
        providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        provider_options=[
            {
                "config_file": VAIP_CONFIG,
                "cacheDir": cache_dir,
                "cacheKey": "demucs_bfp16_vai",
            },
            {},
        ],
    )
    print(f"[OK] Session initialized! Providers: {sess.get_providers()}", flush=True)

    inputs = {
        "x_spec": np.zeros((1, 4, 2048, 336), dtype=np.float32),
        "x_time": np.zeros((1, 2, 1, 344064), dtype=np.float32),
    }
    t0 = time.perf_counter()
    out = sess.run(None, inputs)
    print(f"[OK] Inference completed in {(time.perf_counter()-t0)*1000:.1f}ms!", flush=True)
except Exception as e:
    print(f"[FAIL] {e}", flush=True)
