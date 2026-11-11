import ctypes
import gc
import os
import sys
import time
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

# Point to .venv PyTorch lib for custom ops DLL dependencies (without importing torch)
torch_lib = r"C:\Users\zubair\Documents\GitHub\demucs-npu\.venv\Lib\site-packages\torch\lib"
os.add_dll_directory(torch_lib)
os.environ["PATH"] = f"{torch_lib};{os.environ.get('PATH', '')}"

quark_dll = r"C:\Users\zubair\AppData\Local\torch_extensions\torch_extensions\Cache\py312_cpu\quark_custom_ops_stable_abi\quark_custom_ops_stable_abi.dll"
cfg = r"C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0-demucs\Lib\site-packages\misc\vaip_config.json"
model_path = r"option_a_bfp16\models\htdemucs_candidate_a_pure_aie_bfp16_clean_cpu.onnx"
cache_dir = r"option_a_bfp16\xdna2_cache_pure_aie"
cache_key = "candidate_a_pure_aie"

# Windows PSAPI configuration for 64-bit processes
kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi
kernel32.GetCurrentProcess.restype = ctypes.c_void_p
psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
psapi.EmptyWorkingSet.restype = ctypes.c_int


def get_memory_mb() -> tuple[float, float]:
    """Returns (WorkingSet_Physical_MB, PrivateCommit_MB) using psutil."""
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    rss_mb = mem.rss / (1024 * 1024)
    priv_mb = getattr(mem, "private", mem.rss) / (1024 * 1024)
    return rss_mb, priv_mb


def trim_working_set() -> tuple[float, float]:
    """Forces Windows to evict cold initialization pages (.rai, Protobuf, unused DLL pages) from physical RAM."""
    gc.collect()
    h_proc = kernel32.GetCurrentProcess()
    ret = psapi.EmptyWorkingSet(h_proc)
    if ret == 0:
        err = ctypes.get_last_error()
        print(f"[WARN] EmptyWorkingSet failed with error code {err}")
    return get_memory_mb()


class LowRamCachedOrtRunner:
    def __init__(
        self,
        model_path: str,
        vaip_config: str,
        cache_dir: str,
        cache_key: str,
        quark_dll: str | None = None,
        intra_threads: int = 8,
    ):
        so = ort.SessionOptions()
        so.log_severity_level = 1

        # Low-RAM configuration: release intermediate CPU buffers immediately like PyTorch
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        so.enable_mem_reuse = True
        so.intra_op_num_threads = intra_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        # Prevent ORT from duplicating initializers during prepacking
        so.add_session_config_entry("session.disable_prepacking", "1")

        if quark_dll:
            so.register_custom_ops_library(quark_dll)

        print("[INIT] Creating Low-RAM InferenceSession...")
        t0 = time.perf_counter()
        self.sess = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            provider_options=[
                {
                    "config_file": vaip_config,
                    "cacheDir": cache_dir,
                    "cacheKey": cache_key,
                },
                {},
            ],
            disabled_optimizers=["NchwcTransformer"],
        )
        print(f"[INIT] Session loaded from AIE cache in {time.perf_counter() - t0:.2f}s!")

        self.io_binding = self.sess.io_binding()
        self.input_buffers: dict[str, np.ndarray] = {}
        self.input_ortvalues: dict[str, ort.OrtValue] = {}
        self.output_buffers: dict[str, np.ndarray] = {}
        self.output_ortvalues: dict[str, ort.OrtValue] = {}

        for inp in self.sess.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            buf = self._alloc_aligned_f32(shape)
            ort_val = ort.OrtValue.ortvalue_from_numpy(buf, "cpu", 0)
            self.input_buffers[inp.name] = buf
            self.input_ortvalues[inp.name] = ort_val
            self.io_binding.bind_ortvalue_input(inp.name, ort_val)

        for out in self.sess.get_outputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in out.shape]
            buf = self._alloc_aligned_f32(shape)
            ort_val = ort.OrtValue.ortvalue_from_numpy(buf, "cpu", 0)
            self.output_buffers[out.name] = buf
            self.output_ortvalues[out.name] = ort_val
            self.io_binding.bind_ortvalue_output(out.name, ort_val)

    @staticmethod
    def _alloc_aligned_f32(shape: list[int], alignment: int = 64) -> np.ndarray:
        n_bytes = int(np.prod(shape)) * 4
        raw = np.empty(n_bytes + alignment, dtype=np.uint8)
        offset = (-raw.ctypes.data) % alignment
        return raw[offset : offset + n_bytes].view(np.float32).reshape(shape)

    def warmup_and_trim(self):
        """Runs 1 warmup pass, then evicts dead .rai / Protobuf initialization pages from RAM."""
        print("[WARMUP] Running 1st hardware warmup pass...")
        t0 = time.perf_counter()
        self.sess.run_with_iobinding(self.io_binding)
        self.io_binding.synchronize_outputs()
        print(f"[WARMUP] Completed in {(time.perf_counter() - t0)*1000:.2f} ms")

        rss_before, priv_before = get_memory_mb()
        print(f"\n[TRIM] Calling EmptyWorkingSet() to evict dead initialization pages...")
        rss_after, priv_after = trim_working_set()
        print(f"  RAM Before Trim : {rss_before:7.1f} MB Physical RSS | {priv_before:7.1f} MB Commit")
        print(f"  RAM After Trim  : {rss_after:7.1f} MB Physical RSS  | {priv_after:7.1f} MB Commit")
        print(f"  Evicted Dead Mem: {rss_before - rss_after:7.1f} MB Physical RAM freed!")

        # Touch active IO buffers with 1 fast pass so only hot runtime pages stay resident
        print("\n[ACTIVE] Running touch pass to re-fault only active runtime pages...")
        t0 = time.perf_counter()
        self.sess.run_with_iobinding(self.io_binding)
        self.io_binding.synchronize_outputs()
        dt = (time.perf_counter() - t0) * 1000.0
        rss_hot, priv_hot = get_memory_mb()
        print(f"  Pass Latency    : {dt:.2f} ms")
        print(f"  Hot Steady-State: {rss_hot:7.1f} MB Physical RSS (Net Savings: {rss_before - rss_hot:.1f} MB)")
        return rss_before, rss_after, rss_hot

    def run_in_place(self, new_inputs: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        if new_inputs:
            for name, arr in new_inputs.items():
                np.copyto(self.input_buffers[name], arr)
        self.sess.run_with_iobinding(self.io_binding)
        return self.output_buffers


def main():
    print("=" * 76)
    print("TESTING LOW-RAM OPTIMIZATIONS: ARENA DISABLE + POST-WARMUP TRIM")
    print("=" * 76)

    rss_init, priv_init = get_memory_mb()
    print(f"Initial Python Memory: {rss_init:.1f} MB RSS | {priv_init:.1f} MB Commit")

    runner = LowRamCachedOrtRunner(
        model_path=model_path,
        vaip_config=cfg,
        cache_dir=cache_dir,
        cache_key=cache_key,
        quark_dll=quark_dll,
    )

    rss_loaded, priv_loaded = get_memory_mb()
    print(f"Session Loaded Memory: {rss_loaded:.1f} MB RSS | {priv_loaded:.1f} MB Commit")

    rss_before, rss_after, rss_hot = runner.warmup_and_trim()

    # Benchmark 5 active passes to measure latency stability and memory growth
    print("\n" + "=" * 76)
    print("BENCHMARKING 5 WARM INFERENCE PASSES (7.80s Audio Chunks)")
    print("=" * 76)
    np.random.seed(42)
    test_inputs = {
        "x_spec": np.random.randn(1, 4, 2048, 336).astype(np.float32),
        "x_time": np.random.randn(1, 2, 1, 344064).astype(np.float32),
    }

    latencies = []
    for i in range(5):
        t0 = time.perf_counter()
        _ = runner.run_in_place(test_inputs)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        rss_curr, priv_curr = get_memory_mb()
        print(f"  Pass {i+1}: {dt:7.2f} ms | Physical RSS: {rss_curr:7.1f} MB | Commit: {priv_curr:7.1f} MB")

    mean_lat = np.mean(latencies)
    min_lat = np.min(latencies)
    final_rss, final_priv = get_memory_mb()

    print("\n" + "=" * 76)
    print("EXECUTIVE COMPARISON: DEFAULT ARENA vs LOW-RAM TRIMMED RUNNER")
    print("=" * 76)
    old_baseline_ram = 3576.34  # Measured in previous script
    print(f"Original Physical RAM Footprint   : {old_baseline_ram:7.1f} MB (~3.49 GB)")
    print(f"Low-RAM Trimmed Physical Footprint : {final_rss:7.1f} MB (~{final_rss/1024:.2f} GB)")
    print(f"Net Physical RAM Slashed          : {old_baseline_ram - final_rss:7.1f} MB (~{(old_baseline_ram - final_rss)/1024:.2f} GB REDUCTION!)")
    print(f"Percentage of Physical RAM Freed   : {((old_baseline_ram - final_rss) / old_baseline_ram)*100:.1f}%")
    print("-" * 76)
    print(f"Inference Latency (Mean)          : {mean_lat:7.2f} ms (Min: {min_lat:7.2f} ms)")
    print(f"Real-Time Factor (RTF) on 7.80s   : {mean_lat / 7801.9:.3f}x ({7801.9 / mean_lat:.2f}x Real-Time Speed)")
    print("=" * 76)


if __name__ == "__main__":
    main()
