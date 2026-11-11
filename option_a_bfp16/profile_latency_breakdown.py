import os
import sys
import json
import numpy as np
import onnxruntime as ort

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
so.enable_profiling = True

cfg = r"C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0-demucs\Lib\site-packages\misc\vaip_config.json"
model = r"option_a_bfp16\models\htdemucs_candidate_a_pure_aie_bfp16_clean_cpu.onnx"
cache_dir = r"option_a_bfp16\xdna2_cache_pure_aie"
cache_key = "candidate_a_pure_aie"

sess = ort.InferenceSession(
    model,
    sess_options=so,
    providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
    provider_options=[{"config_file": cfg, "cacheDir": cache_dir, "cacheKey": cache_key}, {}],
    disabled_optimizers=["NchwcTransformer"],
)

x_spec = np.random.randn(1, 4, 2048, 336).astype(np.float32)
x_time = np.random.randn(1, 2, 1, 344064).astype(np.float32)

# Warmup run
sess.run(None, {"x_spec": x_spec, "x_time": x_time})

# Profiled run
sess.run(None, {"x_spec": x_spec, "x_time": x_time})

prof_file = sess.end_profiling()
print(f"Profile saved: {prof_file}")

with open(prof_file, "r") as f:
    events = json.load(f)

# Aggregate duration by provider and op_name
provider_time = {}
op_time = {}
npu_nodes_time = {}

for ev in events:
    if ev.get("cat") == "Node":
        dur_us = ev.get("dur", 0)
        dur_ms = dur_us / 1000.0
        args = ev.get("args", {})
        provider = args.get("provider", "Unknown")
        op = args.get("op_name", "Unknown")
        name = ev.get("name", "")

        provider_time[provider] = provider_time.get(provider, 0.0) + dur_ms
        op_time[op] = op_time.get(op, 0.0) + dur_ms

        if "vitis" in provider.lower():
            npu_nodes_time[name] = npu_nodes_time.get(name, 0.0) + dur_ms

total_ms = sum(provider_time.values())

print("\n" + "=" * 70)
print("INFERENCE LATENCY BREAKDOWN BY EXECUTION PROVIDER")
print("=" * 70)
for prov, t in sorted(provider_time.items(), key=lambda x: -x[1]):
    pct = (t / total_ms) * 100.0 if total_ms > 0 else 0
    print(f"  {prov:28s}: {t:8.2f} ms ({pct:5.1f}%)")
print(f"  {'TOTAL':28s}: {total_ms:8.2f} ms (100.0%)")

print("\n" + "=" * 70)
print("TOP 15 OPERATORS BY RUNTIME")
print("=" * 70)
for op, t in sorted(op_time.items(), key=lambda x: -x[1])[:15]:
    pct = (t / total_ms) * 100.0 if total_ms > 0 else 0
    print(f"  {op:28s}: {t:8.2f} ms ({pct:5.1f}%)")

print("\n" + "=" * 70)
print("VITIS AI NPU FUSED PARTITION TIMING")
print("=" * 70)
for node, t in npu_nodes_time.items():
    print(f"  {node}: {t:.2f} ms")

os.remove(prof_file)

