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

SAMPLE_RATE = 44100
CHUNK_SAMPLES = 344064  # Exactly 7.8019 seconds of audio at 44.1 kHz
CHUNK_DURATION = CHUNK_SAMPLES / SAMPLE_RATE


class DrumsNpuPureRunner:
    """
    Dedicated Drums-Only NPU Runner using AMD Ryzen AI XDNA 2
    with Zero-Copy IOBinding and NchwcTransformer disabled.
    """
    def __init__(self):
        so = ort.SessionOptions()
        so.log_severity_level = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.enable_cpu_mem_arena = True
        so.enable_mem_pattern = True
        so.enable_mem_reuse = True
        so.intra_op_num_threads = 8
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        so.register_custom_ops_library(quark_dll)

        print("[INIT] Loading Drums Specialist Session on AMD Ryzen AI NPU...")
        t0 = time.perf_counter()
        self.sess = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            provider_options=[
                {"config_file": cfg, "cacheDir": cache_dir, "cacheKey": cache_key},
                {},
            ],
            disabled_optimizers=["NchwcTransformer"],
        )
        self.load_time = time.perf_counter() - t0
        print(f"[INIT] Loaded in {self.load_time:.2f}s from AIE hardware cache!")
        print(f"[INIT] Active Providers: {self.sess.get_providers()} (Zero GPU usage)")

        self.io_binding = self.sess.io_binding()
        self.input_buffers: dict[str, np.ndarray] = {}
        self.output_buffers: dict[str, np.ndarray] = {}

        # Pre-allocate 64-byte aligned buffers
        for inp in self.sess.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            buf = self._alloc_aligned_f32(shape)
            ort_val = ort.OrtValue.ortvalue_from_numpy(buf, "cpu", 0)
            self.input_buffers[inp.name] = buf
            self.io_binding.bind_ortvalue_input(inp.name, ort_val)

        for out in self.sess.get_outputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in out.shape]
            buf = self._alloc_aligned_f32(shape)
            ort_val = ort.OrtValue.ortvalue_from_numpy(buf, "cpu", 0)
            self.output_buffers[out.name] = buf
            self.io_binding.bind_ortvalue_output(out.name, ort_val)

        # Warmup pass
        print("[INIT] Running initial hardware warmup pass...")
        t0 = time.perf_counter()
        self.sess.run_with_iobinding(self.io_binding)
        self.io_binding.synchronize_outputs()
        self.cold_time_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[INIT] Warmup complete in {self.cold_time_ms:.2f} ms!")

    @staticmethod
    def _alloc_aligned_f32(shape: list[int], alignment: int = 64) -> np.ndarray:
        n_bytes = int(np.prod(shape)) * 4
        raw = np.empty(n_bytes + alignment, dtype=np.uint8)
        offset = (-raw.ctypes.data) % alignment
        return raw[offset : offset + n_bytes].view(np.float32).reshape(shape)

    def separate_chunk(self, audio_chunk_stereo: np.ndarray, spec_chunk: np.ndarray) -> np.ndarray:
        """
        Executes drums separation on a single 7.80s chunk.
        audio_chunk_stereo: [2, 344064]
        spec_chunk: [4, 2048, 336]
        """
        np.copyto(self.input_buffers["x_time"][0, :, 0, :], audio_chunk_stereo)
        np.copyto(self.input_buffers["x_spec"][0], spec_chunk)
        self.sess.run_with_iobinding(self.io_binding)
        # Returns separated drums audio: [2, 344064]
        return self.output_buffers["out_time"][0, :, 0, :]


def generate_audio(duration_sec: float) -> np.ndarray:
    """Generates stereo test audio with dynamic drum/percussion content."""
    total_samples = int(duration_sec * SAMPLE_RATE)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False, dtype=np.float32)
    left = 0.5 * np.sin(2 * np.pi * 55 * t) + 0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.random.randn(total_samples).astype(np.float32)
    right = 0.5 * np.sin(2 * np.pi * 55 * t) + 0.3 * np.sin(2 * np.pi * 330 * t) + 0.05 * np.random.randn(total_samples).astype(np.float32)
    return np.stack([left, right], axis=0)


def benchmark_drums_track(runner: DrumsNpuPureRunner, track_duration_sec: float):
    total_samples = int(track_duration_sec * SAMPLE_RATE)
    num_chunks = int(np.ceil(total_samples / CHUNK_SAMPLES))

    print("\n" + "=" * 76)
    print(f"BENCHMARK: DRUMS SEPARATION ON NPU — {track_duration_sec:.1f}s AUDIO TRACK")
    print(f"Total Samples: {total_samples} | Sample Rate: {SAMPLE_RATE} Hz | Chunk Size: {CHUNK_DURATION:.2f}s ({CHUNK_SAMPLES} samples)")
    print(f"Total NPU Chunks: {num_chunks}")
    print("=" * 76)

    audio = generate_audio(track_duration_sec)
    chunk_times = []
    separated_drum_chunks = []

    t_track_start = time.perf_counter()

    for i in range(num_chunks):
        start = i * CHUNK_SAMPLES
        end = start + CHUNK_SAMPLES
        if end <= total_samples:
            chunk = audio[:, start:end]
        else:
            pad_width = end - total_samples
            chunk = np.pad(audio[:, start:], ((0, 0), (0, pad_width)))

        spec = np.random.randn(4, 2048, 336).astype(np.float32)

        t0 = time.perf_counter()
        drum_chunk = runner.separate_chunk(chunk, spec)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        chunk_times.append(dt_ms)
        separated_drum_chunks.append(drum_chunk.copy())

        speed = CHUNK_DURATION / (dt_ms / 1000.0)
        print(f"  Chunk {i+1:2d}/{num_chunks}: {dt_ms:7.2f} ms | Chunk Duration: {CHUNK_DURATION:.2f}s | Speed: {speed:.2f}x Real-Time")

    total_track_time = time.perf_counter() - t_track_start
    mean_chunk_ms = np.mean(chunk_times)
    min_chunk_ms = np.min(chunk_times)
    max_chunk_ms = np.max(chunk_times)
    rtf = total_track_time / track_duration_sec
    speedup = track_duration_sec / total_track_time

    # Audio validity checks
    full_drums = np.concatenate(separated_drum_chunks, axis=-1)[:, :total_samples]
    has_nan = np.isnan(full_drums).any()
    has_inf = np.isinf(full_drums).any()
    min_val = float(np.min(full_drums))
    max_val = float(np.max(full_drums))

    print("-" * 76)
    print("SUMMARY RESULTS (DRUMS ONLY ON NPU):")
    print(f"  Audio Duration:          {track_duration_sec:.2f} seconds")
    print(f"  Total Processing Time:   {total_track_time:.2f} seconds")
    print(f"  Mean Latency per Chunk:  {mean_chunk_ms:.2f} ms (Min: {min_chunk_ms:.2f} ms, Max: {max_chunk_ms:.2f} ms)")
    print(f"  Real-Time Factor (RTF):  {rtf:.3f}x")
    print(f"  Separation Throughput:   {speedup:.2f}x Real-Time Speed")
    print(f"  Signal Validity:         NaN={has_nan}, Inf={has_inf}, Range=[{min_val:.4f}, {max_val:.4f}]")
    if rtf < 1.0:
        print(f"  VERDICT:                 FASTER THAN REAL-TIME ({speedup:.2f}x faster than playback)")
    else:
        print(f"  VERDICT:                 SLOWER THAN REAL-TIME ({rtf:.2f}x RTF)")
    print("=" * 76)

    return {
        "duration": track_duration_sec,
        "time": total_track_time,
        "mean_chunk": mean_chunk_ms,
        "min_chunk": min_chunk_ms,
        "max_chunk": max_chunk_ms,
        "rtf": rtf,
        "speedup": speedup,
    }


def main():
    print("=" * 76)
    print("AMD Ryzen AI (XDNA 2) NPU Hardware Benchmark: 1-Stem (Drums Specialist)")
    print(f"Model:    {model_path} ({os.path.getsize(model_path)/(1024*1024):.2f} MB)")
    print(f"Hardware: AMD Ryzen AI NPU (AIE2P 2x4x4 Grid, Strix Point)")
    print(f"Target:   DRUMS STEM ONLY (100% NPU + Host Glue, ZERO GPU)")
    print("=" * 76)

    runner = DrumsNpuPureRunner()

    # Test cases:
    # 1. 7.80s single chunk
    # 2. 30.0s audio track (4 chunks)
    # 3. 60.0s audio track (8 chunks)
    # 4. 120.0s audio track (2 minutes = 16 chunks)
    test_durations = [7.80, 30.0, 60.0, 120.0]
    results = []

    for dur in test_durations:
        res = benchmark_drums_track(runner, track_duration_sec=dur)
        results.append(res)
        time.sleep(2.0)  # Thermal recovery between test suites

    print("\n" + "=" * 76)
    print("FINAL EXECUTIVE BENCHMARK REPORT: DRUMS ON AMD RYZEN AI NPU")
    print("=" * 76)
    print(f"{'Audio Duration':18s} | {'Total Wall Time':16s} | {'Mean Chunk':14s} | {'RTF':10s} | {'Throughput'}")
    print("-" * 76)
    for r in results:
        print(
            f"{r['duration']:6.1f} seconds      | "
            f"{r['time']:6.2f} seconds     | "
            f"{r['mean_chunk']:7.1f} ms     | "
            f"{r['rtf']:5.3f}x    | "
            f"{r['speedup']:.2f}x Real-Time"
        )
    print("=" * 76)


if __name__ == "__main__":
    main()

