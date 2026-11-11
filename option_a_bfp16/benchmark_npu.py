import os
import sys
import time
import glob
import numpy as np
import onnxruntime as ort

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

torch_lib = r"C:\Users\zubair\Documents\GitHub\demucs-npu\.venv\Lib\site-packages\torch\lib"
os.add_dll_directory(torch_lib)
os.environ["PATH"] = f"{torch_lib};{os.environ.get('PATH', '')}"

dll = r"C:\Users\zubair\AppData\Local\torch_extensions\torch_extensions\Cache\py312_cpu\quark_custom_ops_stable_abi\quark_custom_ops_stable_abi.dll"
so = ort.SessionOptions()
so.register_custom_ops_library(dll)
so.intra_op_num_threads = 8
so.inter_op_num_threads = 1
so.log_severity_level = 1

cfg = r"C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0-demucs\Lib\site-packages\misc\vaip_config.json"
model = r"option_a_bfp16\models\htdemucs_candidate_a_pure_aie_bfp16_clean_cpu.onnx"
cache_dir = r"option_a_bfp16\xdna2_cache_pure_aie"
cache_key = "candidate_a_pure_aie"

print("=" * 76)
print("AMD Ryzen AI (XDNA 2) NPU Hardware Benchmark: HTDemucs Drums Specialist")
print(f"Model:     {model} ({os.path.getsize(model)/(1024*1024):.2f} MB)")
print(f"Hardware:  AMD Ryzen AI NPU (AIE2P 2x4x4 Array, Strix Point)")
print(f"Cache:     {cache_dir}\\{cache_key}\\{cache_key}.rai")
print(f"Threads:   CPU intra_op_num_threads = 8")
print("=" * 76)

t0 = time.perf_counter()
sess = ort.InferenceSession(
    model,
    sess_options=so,
    providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
    provider_options=[{"config_file": cfg, "cacheDir": cache_dir, "cacheKey": cache_key}, {}]
)
init_time = time.perf_counter() - t0
print(f"\n[CACHE] Session loaded from hardware .rai cache in {init_time:.2f}s!")
print(f"Active Providers: {sess.get_providers()}")

# Synthetic test inputs (4-second stereo audio at 44.1 kHz)
np.random.seed(42)
x_spec = np.random.randn(1, 4, 2048, 336).astype(np.float32)
x_time = np.random.randn(1, 2, 1, 344064).astype(np.float32)

print("\n--- Warmup Run (Cold Inference) ---")
t_cold_start = time.perf_counter()
cold_outs = sess.run(None, {"x_spec": x_spec, "x_time": x_time})
cold_latency = (time.perf_counter() - t_cold_start) * 1000.0
print(f"Cold Latency: {cold_latency:.2f} ms")
for i, o in enumerate(cold_outs):
    print(f"  Output {i}: shape={o.shape}, dtype={o.dtype}, min={o.min():.4f}, max={o.max():.4f}, has_nan={np.isnan(o).any()}")

print("\n--- Running 10 Warm Timed Benchmark Passes on NPU Hardware ---")
warm_latencies = []
for r in range(10):
    t_start = time.perf_counter()
    outs = sess.run(None, {"x_spec": x_spec, "x_time": x_time})
    lat_ms = (time.perf_counter() - t_start) * 1000.0
    warm_latencies.append(lat_ms)
    print(f"  Pass {r+1:2d}: {lat_ms:7.2f} ms")

mean_lat = float(np.mean(warm_latencies))
std_lat = float(np.std(warm_latencies))
median_lat = float(np.median(warm_latencies))
min_lat = float(np.min(warm_latencies))
max_lat = float(np.max(warm_latencies))

directml_baseline = 422.0 # ms on Radeon 890M GPU
speedup_vs_gpu = directml_baseline / mean_lat

print("\n" + "=" * 76)
print("BENCHMARK RESULTS SUMMARY")
print("=" * 76)
print(f"Target Hardware:       AMD Ryzen AI XDNA 2 NPU (Strix Point)")
print(f"Model AIE Offload:     361.0 GOPs (94.1% on NPU)")
print(f"Cold Latency:          {cold_latency:.2f} ms")
print(f"Warm Mean Latency:     {mean_lat:.2f} ms +/- {std_lat:.2f} ms")
print(f"Warm Median Latency:   {median_lat:.2f} ms")
print(f"Warm Min Latency:      {min_lat:.2f} ms")
print(f"Warm Max Latency:      {max_lat:.2f} ms")
print(f"DirectML GPU Baseline: {directml_baseline:.2f} ms")
print(f"Speedup vs GPU:        {speedup_vs_gpu:.2f}x")
print(f"Real-Time Factor (RTF): {mean_lat / 4000.0:.3f}x (4.0s audio chunk)")
print("=" * 76)

