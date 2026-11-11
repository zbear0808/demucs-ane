import os
import sys
import time
import gc
import psutil
import numpy as np
import onnxruntime as ort

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

def get_mem_info():
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    # rss: Working Set (RAM physically resident in physical memory)
    # vms: Virtual Memory Size (Committed / Address space)
    # private: Private working set (RAM unique to this process)
    private = getattr(mem, "private", mem.rss)
    return {
        "rss_mb": mem.rss / (1024 * 1024),
        "vms_mb": mem.vms / (1024 * 1024),
        "private_mb": private / (1024 * 1024),
    }

def print_mem_step(step_name: str, baseline_rss: float = 0.0):
    gc.collect()
    info = get_mem_info()
    delta_rss = info["rss_mb"] - baseline_rss if baseline_rss > 0 else 0.0
    print(f"  {step_name:40s} | Working Set: {info['rss_mb']:8.2f} MB | Private: {info['private_mb']:8.2f} MB | (+{delta_rss:6.2f} MB)")
    return info["rss_mb"]

def main():
    print("=" * 82)
    print("AMD Ryzen AI (XDNA 2) NPU: Exact Host RAM & Hot-Standby Footprint Analysis")
    print("=" * 82)

    sys_mem = psutil.virtual_memory()
    print(f"System Total Physical RAM: {sys_mem.total / (1024**3):.2f} GB")
    print(f"System Available RAM     : {sys_mem.available / (1024**3):.2f} GB ({sys_mem.percent}% used)")
    print("-" * 82)

    # 1. Baseline Python process
    base_rss = print_mem_step("1. Baseline Python Process")

    # 2. Point to .venv PyTorch lib & load Quark DLL
    torch_lib = r"C:\Users\zubair\Documents\GitHub\demucs-npu\.venv\Lib\site-packages\torch\lib"
    os.add_dll_directory(torch_lib)
    os.environ["PATH"] = f"{torch_lib};{os.environ.get('PATH', '')}"
    quark_dll = r"C:\Users\zubair\AppData\Local\torch_extensions\torch_extensions\Cache\py312_cpu\quark_custom_ops_stable_abi\quark_custom_ops_stable_abi.dll"
    so = ort.SessionOptions()
    so.log_severity_level = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_cpu_mem_arena = True
    so.enable_mem_pattern = True
    so.enable_mem_reuse = True
    so.intra_op_num_threads = 8
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.register_custom_ops_library(quark_dll)

    print_mem_step("2. After Loading Custom Ops Library", base_rss)

    # 3. Model path and cache configs
    cfg = r"C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0-demucs\Lib\site-packages\misc\vaip_config.json"
    model_path = r"option_a_bfp16\models\htdemucs_candidate_a_pure_aie_bfp16_clean_cpu.onnx"
    cache_dir = r"option_a_bfp16\xdna2_cache_pure_aie"
    cache_key = "candidate_a_pure_aie"

    # 4. Load InferenceSession from AIE cache
    t0 = time.perf_counter()
    sess = ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        provider_options=[
            {"config_file": cfg, "cacheDir": cache_dir, "cacheKey": cache_key},
            {},
        ],
        disabled_optimizers=["NchwcTransformer"],
    )
    load_time = time.perf_counter() - t0
    sess_rss = print_mem_step("3. After Loading ONNX Session & AIE Cache", base_rss)

    # 5. Pre-allocate aligned IOBinding buffers
    io_binding = sess.io_binding()
    input_buffers = {}
    output_buffers = {}

    def _alloc_aligned(shape):
        n_bytes = int(np.prod(shape)) * 4
        raw = np.empty(n_bytes + 64, dtype=np.uint8)
        offset = (-raw.ctypes.data) % 64
        return raw[offset : offset + n_bytes].view(np.float32).reshape(shape)

    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        buf = _alloc_aligned(shape)
        ort_val = ort.OrtValue.ortvalue_from_numpy(buf, "cpu", 0)
        input_buffers[inp.name] = buf
        io_binding.bind_ortvalue_input(inp.name, ort_val)

    for out in sess.get_outputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in out.shape]
        buf = _alloc_aligned(shape)
        ort_val = ort.OrtValue.ortvalue_from_numpy(buf, "cpu", 0)
        output_buffers[out.name] = buf
        io_binding.bind_ortvalue_output(out.name, ort_val)

    print_mem_step("4. After Pre-Allocating Pinned IOBinding Buffers", base_rss)

    # 6. First Warmup Pass (Locks internal memory pattern arena & XRT BOs)
    t0 = time.perf_counter()
    sess.run_with_iobinding(io_binding)
    io_binding.synchronize_outputs()
    warmup_time = time.perf_counter() - t0
    warm_rss = print_mem_step("5. After Hardware Warmup Pass (Arena Locked)", base_rss)

    # 7. Active Inference Passes
    pass_times = []
    for i in range(3):
        t0 = time.perf_counter()
        sess.run_with_iobinding(io_binding)
        pass_times.append((time.perf_counter() - t0) * 1000.0)

    active_rss = print_mem_step("6. During/After Active Inference Passes", base_rss)

    # 8. Idle Hot-Standby State
    time.sleep(1.0)
    hot_standby_rss = print_mem_step("7. Idle Hot-Standby State (Ready to Run)", base_rss)

    print("\n" + "=" * 82)
    print("EXECUTIVE MEMORY FOOTPRINT & HOT-STANDBY SUMMARY")
    print("=" * 82)
    proc_info = get_mem_info()
    print(f"  One-Time Cold Load Time:             {load_time:.2f} seconds")
    print(f"  First Warmup Pass Time:              {warmup_time:.2f} seconds")
    print(f"  Warm Steady-State Inference Time:    {np.mean(pass_times):.2f} ms per 7.8s chunk")
    print("-" * 82)
    print(f"  Baseline Python Runtime Overhead:    {base_rss:.2f} MB")
    print(f"  Model & AIE Partition RAM Overhead:  {sess_rss - base_rss:.2f} MB")
    print(f"  Pinned IO & Static Memory Arena:     {warm_rss - sess_rss:.2f} MB")
    print(f"  TOTAL RESIDENT RAM (Working Set):    {proc_info['rss_mb']:.2f} MB (~{proc_info['rss_mb']/1024:.2f} GB)")
    print(f"  TOTAL COMMITTED PRIVATE RAM:         {proc_info['private_mb']:.2f} MB (~{proc_info['private_mb']/1024:.2f} GB)")
    print(f"  PERCENTAGE OF SYSTEM RAM ({sys_mem.total/(1024**3):.1f} GB):   {(proc_info['rss_mb'] / (sys_mem.total/(1024*1024))) * 100:.2f}%")
    print("=" * 82)

if __name__ == "__main__":
    main()

