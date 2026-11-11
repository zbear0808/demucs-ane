import os
import sys
import time
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


class CachedOrtRunner:
    """
    Zero-allocation ONNX Runtime runner using pre-pinned OrtValue buffers,
    IOBinding, and static internal memory-pattern arena caching.
    """
    def __init__(
        self,
        model_path: str,
        vaip_config: str,
        cache_dir: str,
        cache_key: str,
        quark_dll: str | None = None,
        intra_threads: int = 8,
        disable_nchwc: bool = True,
    ):
        so = ort.SessionOptions()
        so.log_severity_level = 1

        # 1. Pin internal intermediate buffers (glu_14, pad_6, add_167) in static arena
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.enable_cpu_mem_arena = True
        so.enable_mem_pattern = True
        so.enable_mem_reuse = True
        so.intra_op_num_threads = intra_threads

        # Keep extended optimizations (constant folding, fusions) while avoiding NCHWc reorders
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        if quark_dll:
            so.register_custom_ops_library(quark_dll)

        disabled_opts = ["NchwcTransformer"] if disable_nchwc else []

        # 2. Initialize session with NchwcTransformer explicitly disabled
        print(f"Creating InferenceSession (disable_nchwc={disable_nchwc})...")
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
            disabled_optimizers=disabled_opts,
        )
        print("InferenceSession created successfully!")

        self.io_binding = self.sess.io_binding()
        self.input_buffers: dict[str, np.ndarray] = {}
        self.input_ortvalues: dict[str, ort.OrtValue] = {}
        self.output_buffers: dict[str, np.ndarray] = {}
        self.output_ortvalues: dict[str, ort.OrtValue] = {}

        # 3. Pre-allocate 64-byte aligned contiguous NumPy buffers & pin OrtValues once
        for inp in self.sess.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            buf = self._alloc_aligned_f32(shape)
            ort_val = ort.OrtValue.ortvalue_from_numpy(buf, "cpu", 0)

            self.input_buffers[inp.name] = buf
            self.input_ortvalues[inp.name] = ort_val
            self.io_binding.bind_ortvalue_input(inp.name, ort_val)
            print(f"[IOBind Input]  {inp.name:35s} shape={shape} ptr={hex(ort_val.data_ptr())}")

        for out in self.sess.get_outputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in out.shape]
            buf = self._alloc_aligned_f32(shape)
            ort_val = ort.OrtValue.ortvalue_from_numpy(buf, "cpu", 0)

            self.output_buffers[out.name] = buf
            self.output_ortvalues[out.name] = ort_val
            self.io_binding.bind_ortvalue_output(out.name, ort_val)
            print(f"[IOBind Output] {out.name:35s} shape={shape} ptr={hex(ort_val.data_ptr())}")

    @staticmethod
    def _alloc_aligned_f32(shape: list[int], alignment: int = 64) -> np.ndarray:
        """Allocates a cache-line / AVX-512 / DMA-aligned (64-byte) C-contiguous float32 buffer."""
        n_bytes = int(np.prod(shape)) * 4
        raw = np.empty(n_bytes + alignment, dtype=np.uint8)
        offset = (-raw.ctypes.data) % alignment
        aligned_view = raw[offset : offset + n_bytes].view(np.float32)
        return aligned_view.reshape(shape)

    def warmup(self, runs: int = 2):
        """Warmup pass locks in ORT's internal memory pattern planner and XRT BO mappings."""
        for _ in range(runs):
            self.sess.run_with_iobinding(self.io_binding)
        self.io_binding.synchronize_outputs()

    def run_in_place(self, new_inputs: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """
        Executes inference using pre-bound buffers with zero allocations.
        If new_inputs is provided, copies data in-place into the pinned input buffers.
        """
        if new_inputs:
            for name, arr in new_inputs.items():
                np.copyto(self.input_buffers[name], arr)

        self.sess.run_with_iobinding(self.io_binding)
        # Output data is already written directly into self.output_buffers in-place
        return self.output_buffers


def main():
    print("=" * 76)
    print("AMD Ryzen AI (XDNA 2) NPU Optimization:")
    print("Zero-Copy IOBinding & NchwcTransformer Elimination Benchmark")
    print("=" * 76)

    # Initialize runner with NchwcTransformer disabled
    runner = CachedOrtRunner(
        model_path=model_path,
        vaip_config=cfg,
        cache_dir=cache_dir,
        cache_key=cache_key,
        quark_dll=quark_dll,
        disable_nchwc=True,
    )

    # Populate inputs with random test data
    np.random.seed(42)
    test_inputs = {
        "x_spec": np.random.randn(1, 4, 2048, 336).astype(np.float32),
        "x_time": np.random.randn(1, 2, 1, 344064).astype(np.float32),
    }

    print("\nWarming up session & locking internal memory pattern arena...")
    runner.warmup(runs=1)
    time.sleep(2.0)

    num_runs = 5
    print(f"\n--- Running Interleaved A/B Benchmark (Standard sess.run vs Zero-Copy IOBinding) ({num_runs} pairs) ---")
    feed = {k: v.copy() for k, v in test_inputs.items()}
    std_times = []
    iob_times = []

    for i in range(num_runs):
        # 1. Standard sess.run()
        t0 = time.perf_counter()
        outs_std = runner.sess.run(None, feed)
        dt_std = (time.perf_counter() - t0) * 1000.0
        std_times.append(dt_std)

        time.sleep(1.5)  # Allow APU thermals and clocks to stabilize

        # 2. Zero-Copy IOBinding run_in_place()
        t0 = time.perf_counter()
        outs_iob = runner.run_in_place(test_inputs)
        dt_iob = (time.perf_counter() - t0) * 1000.0
        iob_times.append(dt_iob)

        print(f"  Pair {i+1}: Standard = {dt_std:7.2f} ms | IOBinding = {dt_iob:7.2f} ms | Diff = {dt_std - dt_iob:+6.2f} ms")
        time.sleep(1.5)

    # Verify parity between sess.run() and run_in_place()
    diff_spec = np.max(np.abs(outs_std[0] - outs_iob["out_spec"]))
    diff_time = np.max(np.abs(outs_std[1] - outs_iob["out_time"]))
    print(f"\nVerification between sess.run() and Zero-Copy run_in_place():")
    print(f"  Max Diff Spectrogram: {diff_spec:.6e}")
    print(f"  Max Diff Time-Domain: {diff_time:.6e}")

    # Summary
    mean_std = np.mean(std_times)
    mean_iob = np.mean(iob_times)
    min_iob = np.min(iob_times)
    baseline_old = 2975.97  # Previous benchmark without NchwcTransformer disabled

    print("\n" + "=" * 76)
    print("BENCHMARK COMPARISON SUMMARY")
    print("=" * 76)
    print(f"Baseline (Old sess.run with NCHWc reorders):  {baseline_old:.2f} ms")
    print(f"NchwcTransformer Disabled sess.run() mean  :  {mean_std:.2f} ms (Savings: {baseline_old - mean_std:.2f} ms)")
    print(f"Zero-Copy IOBinding + Memory Arena mean    :  {mean_iob:.2f} ms (Savings: {baseline_old - mean_iob:.2f} ms)")
    print(f"Zero-Copy IOBinding Peak Min Latency       :  {min_iob:.2f} ms")
    print(f"Net Speedup vs Initial Baseline             :  {baseline_old / mean_iob:.3f}x")
    print(f"Real-Time Factor (RTF) on 4.0s Audio Chunk  :  {mean_iob / 4000.0:.3f}x")
    print("=" * 76)


if __name__ == "__main__":
    main()
