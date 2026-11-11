import os
import time
import glob
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

# 1. Enable Vitis AI compiler partition reports and subgraph dumping
os.environ["VAIP_VERBOSITY"] = "1"
os.environ["XLNX_ONNX_EP_VERBOSE"] = "1"
os.environ["XLNX_ENABLE_DUMP_XIR_MODEL"] = "1"
os.environ["XLNX_ENABLE_DUMP"] = "1"

# Automatically find vaip_config.json if path differs slightly
DEFAULT_VAIP_CONFIG = r"C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0\Lib\site-packages\misc\vaip_config.json"
if os.path.exists(DEFAULT_VAIP_CONFIG):
    VAIP_CONFIG = DEFAULT_VAIP_CONFIG
else:
    found = glob.glob(r"C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0\**\vaip_config.json", recursive=True)
    VAIP_CONFIG = found[0] if found else DEFAULT_VAIP_CONFIG

# Ensure torch/lib from .venv is on PATH for any C++ extension dependencies (e.g. torch_cpu.dll)
TORCH_LIB = str(WORKSPACE_ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib")
if os.path.exists(TORCH_LIB):
    os.environ["PATH"] = f"{TORCH_LIB};{os.environ.get('PATH', '')}"
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(TORCH_LIB)
        except Exception:
            pass
WORKSPACE_ROOT = Path(__file__).resolve().parent
search_patterns = [
    r"C:\Users\zubair\AppData\Local\torch_extensions\torch_extensions\Cache\*\quark_custom_ops_stable_abi\*.dll",
    r"C:\Users\zubair\AppData\Local\torch_extensions\torch_extensions\Cache\py312_cpu\quark_custom_ops_stable_abi\*.dll",
    str(WORKSPACE_ROOT / ".venv" / "Lib" / "site-packages" / "quark" / "onnx" / "operators" / "custom_ops" / "**" / "*.dll"),
    str(WORKSPACE_ROOT / ".venv" / "Lib" / "site-packages" / "quark" / "onnx" / "operators" / "custom_ops" / "*.dll"),
]
quark_dlls = []
for p in search_patterns:
    quark_dlls.extend(glob.glob(p, recursive=True))
quark_dlls = list(dict.fromkeys(quark_dlls))  # deduplicate preserving order


def test_model_on_npu(model_path: str, cache_key: str, register_quark_dll: bool = False):
    print("\n" + "=" * 72, flush=True)
    print(f"Testing Model: {model_path} (cacheKey={cache_key})", flush=True)
    print("=" * 72, flush=True)
    if not os.path.exists(model_path):
        print(f"[WARN] Skipping {model_path} (file not found)", flush=True)
        return

    so = ort.SessionOptions()
    so.log_severity_level = 1  # Info/Verbose level to print EP node assignment

    if register_quark_dll:
        if quark_dlls:
            print(f"Registering Quark Custom Ops DLL: {quark_dlls[0]}", flush=True)
            try:
                so.register_custom_ops_library(quark_dlls[0])
            except Exception as e:
                print(f"[WARN] Failed to register custom ops library {quark_dlls[0]}: {e}", flush=True)
        else:
            print("[INFO] No Quark custom ops DLL found to register.", flush=True)

    cache_dir = os.path.abspath("./xdna2_cache")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        print(f"Initializing InferenceSession with VitisAIExecutionProvider for {model_path}...", flush=True)
        t0 = time.perf_counter()
        sess = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            provider_options=[
                {
                    "config_file": VAIP_CONFIG,
                    "cacheDir": cache_dir,
                    "cacheKey": cache_key,
                },
                {},
            ],
        )
        compile_sec = time.perf_counter() - t0
        print(f"[OK] Session initialized in {compile_sec:.2f}s | Providers: {sess.get_providers()}", flush=True)

        # Inspect generated vitisai_ep_report.json inside cacheDir if produced by VOE 1.8.0
        report_files = glob.glob(os.path.join(cache_dir, cache_key, "**", "*report*.json"), recursive=True)
        if not report_files:
            report_files = glob.glob(os.path.join(cache_dir, "**", f"*{cache_key}*report*.json"), recursive=True)

        for rf in report_files:
            print(f"[REPORT] Found Vitis AI Partition Report: {rf}", flush=True)
            try:
                with open(rf, "r") as f:
                    rep = json.load(f)
                    print(json.dumps(rep, indent=2)[:1500], flush=True)
            except Exception as err:
                print(f"  Error reading report: {err}", flush=True)

        # Warmup + Benchmark 1 forward pass of a 7.8s chunk on XDNA 2 NPU
        print("Running forward pass benchmark (7.8s audio chunk: x_spec [1,4,2048,336], x_time [1,2,1,344064])...", flush=True)
        inputs = {
            "x_spec": np.zeros((1, 4, 2048, 336), dtype=np.float32),
            "x_time": np.zeros((1, 2, 1, 344064), dtype=np.float32),
        }
        _ = sess.run(None, inputs)

        t1 = time.perf_counter()
        out = sess.run(None, inputs)
        infer_ms = (time.perf_counter() - t1) * 1000.0
        print(f"[BENCHMARK] XDNA 2 Inference Time: {infer_ms:.1f} ms", flush=True)
        if len(out) >= 2:
            print(f"   Output shapes: spec={out[0].shape}, time={out[1].shape}", flush=True)
        else:
            print(f"   Output length: {len(out)}, shapes: {[o.shape for o in out]}", flush=True)

    except Exception as e:
        print(f"[FAIL] Execution failed on {model_path}:\n   {e}", flush=True)


if __name__ == "__main__":
    print(f"Configured VAIP_CONFIG: {VAIP_CONFIG}")
    print(f"Detected Quark DLLs: {quark_dlls}")

    # 1. Test Standard ONNX QDQ XINT8 (Zero custom ops)
    test_model_on_npu("htdemucs_drums_xdna2_xint8_qdq.onnx", "demucs_xint8_qdq", register_quark_dll=False)

    # 2. Test Standard ONNX QDQ A16W8 (With fallback custom op DLL registered if needed)
    test_model_on_npu("htdemucs_drums_xdna2_a16w8_qdq.onnx", "demucs_a16w8_qdq", register_quark_dll=True)

    # 3. Test Vitis BFPFixNeuron domain-translated model
    test_model_on_npu("htdemucs_drums_xdna2_bfp16_vai.onnx", "demucs_bfp16_vai", register_quark_dll=False)

    # 4. Test Quark BFP16 with Quark custom_ops DLL registered
    test_model_on_npu("htdemucs_drums_xdna2_bfp16.onnx", "demucs_bfp16_quark", register_quark_dll=True)

