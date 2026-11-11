import os
import sys
import time
import glob
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Ensure UTF-8 output
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Enable Vitis AI compiler partition reports and subgraph dumping
os.environ["VAIP_VERBOSITY"] = "1"
os.environ["XLNX_ONNX_EP_VERBOSE"] = "1"
os.environ["XLNX_ENABLE_DUMP_XIR_MODEL"] = "1"
os.environ["XLNX_ENABLE_DUMP"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
VAIP_CONFIG = r"C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0-demucs\Lib\site-packages\misc\vaip_config.json"
CACHE_DIR = str(SCRIPT_DIR / "xdna2_cache_v2")
CACHE_KEY = "candidate_a_bfp16_v2"
MODEL_PATH = str(SCRIPT_DIR / "models" / "htdemucs_candidate_a_bfp16_v2.onnx")

# Ensure torch/lib from .venv is on PATH for custom ops dependencies
TORCH_LIB = str(WORKSPACE_ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib")
if os.path.exists(TORCH_LIB):
    os.environ["PATH"] = f"{TORCH_LIB};{os.environ.get('PATH', '')}"
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(TORCH_LIB)
        except Exception:
            pass

search_patterns = [
    r"C:\Users\zubair\AppData\Local\torch_extensions\torch_extensions\Cache\*\quark_custom_ops_stable_abi\*.dll",
    r"C:\Users\zubair\AppData\Local\torch_extensions\torch_extensions\Cache\py312_cpu\quark_custom_ops_stable_abi\*.dll",
    str(WORKSPACE_ROOT / ".venv" / "Lib" / "site-packages" / "quark" / "onnx" / "operators" / "custom_ops" / "**" / "*.dll"),
]
quark_dlls = []
for p in search_patterns:
    quark_dlls.extend(glob.glob(p, recursive=True))
quark_dlls = list(dict.fromkeys(quark_dlls))

so = ort.SessionOptions()
so.log_severity_level = 1

if quark_dlls:
    print(f"Registering Quark Custom Ops DLL: {quark_dlls[0]}", flush=True)
    try:
        so.register_custom_ops_library(quark_dlls[0])
    except Exception as e:
        print(f"[WARN] Failed to register custom ops library: {e}", flush=True)

os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 76, flush=True)
print(f"Testing Candidate A BFP16 V2 on AMD Ryzen AI NPU (XDNA 2)", flush=True)
print(f"Model:       {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/(1024*1024):.2f} MB)", flush=True)
print(f"VAIP Config: {VAIP_CONFIG}", flush=True)
print(f"Cache Dir:   {CACHE_DIR}", flush=True)
print("=" * 76, flush=True)

provider_options = [
    {
        "config_file": VAIP_CONFIG,
        "cacheDir": CACHE_DIR,
        "cacheKey": CACHE_KEY,
    },
    {},
]

t0 = time.perf_counter()
try:
    sess = ort.InferenceSession(
        MODEL_PATH,
        sess_options=so,
        providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        provider_options=provider_options,
    )
    compile_sec = time.perf_counter() - t0
    print(f"\n[SUCCESS] Session compiled in {compile_sec:.2f}s!", flush=True)
    print(f"Active Providers: {sess.get_providers()}", flush=True)

    # Test dummy inference
    print("\nRunning dummy inference pass on NPU...", flush=True)
    x_spec = np.random.randn(1, 4, 2048, 336).astype(np.float32)
    x_time = np.random.randn(1, 2, 1, 344064).astype(np.float32)

    t_inf_0 = time.perf_counter()
    outputs = sess.run(None, {"x_spec": x_spec, "x_time": x_time})
    inf_sec = time.perf_counter() - t_inf_0
    print(f"[SUCCESS] Inference completed in {inf_sec * 1000:.2f} ms ({inf_sec:.3f} s)!", flush=True)
    print(f"Output count: {len(outputs)}", flush=True)
    for i, o in enumerate(outputs):
        print(f"  Output {i}: shape={o.shape}, dtype={o.dtype}", flush=True)

except Exception as e:
    compile_sec = time.perf_counter() - t0
    print(f"\n[FAIL] Session compilation/execution failed after {compile_sec:.2f}s:\n{e}", flush=True)
    import traceback
    traceback.print_exc()

# Look for preliminary summary
summary_files = glob.glob(os.path.join(CACHE_DIR, "**", "*preliminary*"), recursive=True)
for sf in summary_files:
    print(f"\n=== Preliminary Summary ({sf}) ===", flush=True)
    with open(sf, "r") as f:
        print(f.read(), flush=True)

# Look for trace
trace_files = glob.glob(os.path.join(CACHE_DIR, "**", "*partition_trace.csv"), recursive=True)
for tf in trace_files:
    print(f"\n=== Partition Trace ({tf}) ===", flush=True)
    import pandas as pd
    df = pd.read_csv(tf)
    print(df['Status'].value_counts(), flush=True)

