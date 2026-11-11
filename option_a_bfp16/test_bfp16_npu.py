import os
import sys
import time
import glob
import json
import traceback
from pathlib import Path
import numpy as np
import onnxruntime as ort

# Ensure UTF-8 output
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

# 1. Enable Vitis AI compiler partition reports and subgraph dumping
os.environ["VAIP_VERBOSITY"] = "1"
os.environ["XLNX_ONNX_EP_VERBOSE"] = "1"
os.environ["XLNX_ENABLE_DUMP_XIR_MODEL"] = "1"
os.environ["XLNX_ENABLE_DUMP"] = "1"

# Exact VAIP config path
VAIP_CONFIG = r"C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0-demucs\Lib\site-packages\misc\vaip_config.json"
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

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


def test_single_model(model_path: str, cache_key: str, cache_dir: str) -> dict:
    result = {
        "model_name": os.path.basename(model_path),
        "model_path": model_path,
        "cache_key": cache_key,
        "compiled": False,
        "compile_time_s": None,
        "inference_success": False,
        "inference_time_ms": None,
        "providers": [],
        "partition_report": None,
        "error": None,
        "dma_offset_error_detected": False,
    }

    print("\n" + "=" * 76, flush=True)
    print(f"Testing Candidate: {os.path.basename(model_path)} (cacheKey={cache_key})", flush=True)
    print("=" * 76, flush=True)

    if not os.path.exists(model_path):
        err = f"Model file not found: {model_path}"
        print(f"[ERROR] {err}", flush=True)
        result["error"] = err
        return result

    so = ort.SessionOptions()
    so.log_severity_level = 1

    if quark_dlls:
        print(f"Registering Quark Custom Ops DLL: {quark_dlls[0]}", flush=True)
        try:
            so.register_custom_ops_library(quark_dlls[0])
        except Exception as e:
            print(f"[WARN] Failed to register custom ops library: {e}", flush=True)
    else:
        print("[INFO] No Quark custom ops DLL found to register.", flush=True)

    target_cache = os.path.join(cache_dir, cache_key)
    os.makedirs(target_cache, exist_ok=True)

    print(f"Compiling / Initializing Session with VitisAIExecutionProvider...", flush=True)
    print(f"VAIP Config: {VAIP_CONFIG}", flush=True)
    print(f"Cache Dir:   {cache_dir}", flush=True)

    t0 = time.perf_counter()
    sess = None
    try:
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
        result["compiled"] = True
        result["compile_time_s"] = round(compile_sec, 2)
        result["providers"] = sess.get_providers()
        print(f"[SUCCESS] Session compiled in {compile_sec:.2f}s | Providers: {sess.get_providers()}", flush=True)

    except Exception as e:
        compile_sec = time.perf_counter() - t0
        err_msg = str(e)
        result["error"] = err_msg
        result["compile_time_s"] = round(compile_sec, 2)
        print(f"[FAIL] Compilation / Session initialization failed in {compile_sec:.2f}s:\n{err_msg}", flush=True)

        if "77-5732" in err_msg or "cannot be adjusted to 32 bit word" in err_msg or "offset 1 and element size 8" in err_msg:
            result["dma_offset_error_detected"] = True
            print("[CRITICAL MATCH] INT8 AIE DMA offset error (77-5732) detected!", flush=True)

    # Search for partition reports and compiler outputs in cacheDir
    report_patterns = [
        os.path.join(cache_dir, cache_key, "**", "*report*.json"),
        os.path.join(cache_dir, "**", f"*{cache_key}*report*.json"),
        os.path.join(cache_dir, cache_key, "**", "*.json"),
    ]
    found_reports = []
    for pat in report_patterns:
        found_reports.extend(glob.glob(pat, recursive=True))
    found_reports = list(dict.fromkeys(found_reports))

    for rf in found_reports:
        print(f"[REPORT] Found compiler partition report: {rf}", flush=True)
        try:
            with open(rf, "r") as f:
                rep = json.load(f)
                result["partition_report"] = rep
                print(json.dumps(rep, indent=2)[:2000], flush=True)
        except Exception as err:
            print(f"  Error reading {rf}: {err}", flush=True)

    # Check for any log files in cache directory
    log_files = glob.glob(os.path.join(cache_dir, cache_key, "**", "*.log"), recursive=True)
    for lf in log_files:
        print(f"[COMPILER LOG] Found compiler log: {lf}", flush=True)
        try:
            with open(lf, "r", errors="ignore") as f:
                content = f.read()
                if "77-5732" in content or "cannot be adjusted to 32 bit word" in content:
                    result["dma_offset_error_detected"] = True
                    print("[CRITICAL MATCH] AIE DMA offset error (77-5732) found in compiler log!", flush=True)
                print(f"--- Log preview ({lf}) ---\n" + content[:1500] + "\n--- End preview ---", flush=True)
        except Exception as err:
            print(f"  Error reading log {lf}: {err}", flush=True)

    # If compiled, run inference benchmark
    if sess is not None:
        try:
            input_feed = {}
            for inp in sess.get_inputs():
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                # Default dimension handling if dynamic
                shape = [1 if s is None or s <= 0 else s for s in shape]
                print(f"Input tensor '{inp.name}': shape={shape}, type={inp.type}", flush=True)
                input_feed[inp.name] = np.zeros(shape, dtype=np.float32)

            print("Warming up inference pass...", flush=True)
            _ = sess.run(None, input_feed)

            print("Benchmarking inference execution time...", flush=True)
            t_inf = time.perf_counter()
            outputs = sess.run(None, input_feed)
            infer_ms = (time.perf_counter() - t_inf) * 1000.0
            result["inference_success"] = True
            result["inference_time_ms"] = round(infer_ms, 2)
            print(f"[BENCHMARK] NPU Inference completed in {infer_ms:.2f} ms", flush=True)
            for i, o in enumerate(outputs):
                print(f"   Output[{i}]: shape={o.shape}, dtype={o.dtype}", flush=True)

        except Exception as e:
            infer_err = str(e)
            result["inference_error"] = infer_err
            print(f"[FAIL] Inference execution failed:\n{infer_err}", flush=True)

    return result


def main():
    cache_dir = os.path.abspath(str(SCRIPT_DIR / "xdna2_cache"))
    os.makedirs(cache_dir, exist_ok=True)
    models_dir = SCRIPT_DIR / "models"

    print("=" * 76)
    print("Option A: HTDemucs Quark BFP16 NPU Evaluation (AMD Ryzen AI XDNA 2)")
    print(f"VAIP Config: {VAIP_CONFIG}")
    print(f"Available Providers: {ort.get_available_providers()}")
    print("=" * 76)

    candidates = [
        ("Candidate A (2-channel audio)", str(models_dir / "htdemucs_candidate_a_bfp16.onnx"), "candidate_a_bfp16"),
        ("Candidate A 4-Channel (4-channel audio)", str(models_dir / "htdemucs_candidate_a_4ch_bfp16.onnx"), "candidate_a_4ch_bfp16"),
    ]

    all_results = []
    for name, path, key in candidates:
        res = test_single_model(path, key, cache_dir)
        res["display_name"] = name
        all_results.append(res)

    results_file = SCRIPT_DIR / "test_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[DONE] All test results written to {results_file}")


if __name__ == "__main__":
    main()
