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


class StreamDemucsNpuRunner:
    """
    High-performance zero-copy streaming runner for HTDemucs Candidate A
    on AMD Ryzen AI XDNA 2 NPU.
    """
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
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.enable_cpu_mem_arena = True
        so.enable_mem_pattern = True
        so.enable_mem_reuse = True
        so.intra_op_num_threads = intra_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        if quark_dll:
            so.register_custom_ops_library(quark_dll)

        print("[INIT] Creating NPU session with NchwcTransformer disabled...")
        t0 = time.perf_counter()
        self.sess = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            provider_options=[
                {"config_file": vaip_config, "cacheDir": cache_dir, "cacheKey": cache_key},
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

        # Pre-allocate aligned memory buffers
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

        # Single warmup run
        print("[INIT] Performing initial hardware warmup pass...")
        self.sess.run_with_iobinding(self.io_binding)
        self.io_binding.synchronize_outputs()
        print("[INIT] Hardware warmup complete!")

    @staticmethod
    def _alloc_aligned_f32(shape: list[int], alignment: int = 64) -> np.ndarray:
        n_bytes = int(np.prod(shape)) * 4
        raw = np.empty(n_bytes + alignment, dtype=np.uint8)
        offset = (-raw.ctypes.data) % alignment
        return raw[offset : offset + n_bytes].view(np.float32).reshape(shape)

    def process_chunk_in_place(self, x_spec: np.ndarray, x_time: np.ndarray) -> np.ndarray:
        np.copyto(self.input_buffers["x_spec"], x_spec)
        np.copyto(self.input_buffers["x_time"], x_time)
        self.sess.run_with_iobinding(self.io_binding)
        # Returns reference to time-domain separated drums waveform [1, 2, 1, 344064]
        return self.output_buffers["out_time"]


def generate_synthetic_audio(duration_sec: float) -> np.ndarray:
    """Generates synthetic stereo audio with realistic multi-frequency musical content."""
    total_samples = int(duration_sec * SAMPLE_RATE)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False, dtype=np.float32)
    # Bass kick (60 Hz), snare transient, cymbal sizzle (8 kHz), mid harmony (440 Hz)
    left = (
        0.4 * np.sin(2 * np.pi * 60 * t) * (np.sin(2 * np.pi * 2 * t) ** 2)
        + 0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.1 * np.random.randn(total_samples).astype(np.float32)
    )
    right = (
        0.4 * np.sin(2 * np.pi * 60 * t) * (np.sin(2 * np.pi * 2 * t) ** 2)
        + 0.3 * np.sin(2 * np.pi * 880 * t)
        + 0.1 * np.random.randn(total_samples).astype(np.float32)
    )
    return np.stack([left, right], axis=0)  # [2, total_samples]


def benchmark_audio_stream(runner: StreamDemucsNpuRunner, duration_sec: float, overlap: float = 0.0):
    total_samples = int(duration_sec * SAMPLE_RATE)
    print("\n" + "=" * 76)
    print(f"BENCHMARKING AUDIO STREAM: {duration_sec:.1f} SECONDS ({total_samples} samples at {SAMPLE_RATE} Hz)")
    print(f"Chunk Size: {CHUNK_DURATION:.2f}s ({CHUNK_SAMPLES} samples) | Overlap: {overlap * 100:.0f}%")
    print("=" * 76)

    # Synthesize realistic audio waveform
    audio = generate_synthetic_audio(duration_sec)

    step_samples = int(CHUNK_SAMPLES * (1.0 - overlap))
    num_chunks = int(np.ceil((total_samples - CHUNK_SAMPLES) / step_samples)) + 1 if total_samples > CHUNK_SAMPLES else 1

    print(f"Total Chunks to Process on NPU: {num_chunks}")
    
    chunk_times = []
    t_start = time.perf_counter()

    for idx in range(num_chunks):
        start_samp = idx * step_samples
        end_samp = start_samp + CHUNK_SAMPLES

        # Extract or zero-pad slice
        if end_samp <= total_samples:
            chunk = audio[:, start_samp:end_samp]
        else:
            pad_len = end_samp - total_samples
            chunk = np.pad(audio[:, start_samp:], ((0, 0), (0, pad_len)))

        # Format input shapes for Candidate A:
        # x_time: [1, 2, 1, 344064]
        # x_spec: [1, 4, 2048, 336] (dummy mock spectrogram matching chunk dimensions)
        x_time = chunk[np.newaxis, :, np.newaxis, :]  # [1, 2, 1, 344064]
        x_spec = np.random.randn(1, 4, 2048, 336).astype(np.float32)

        t0 = time.perf_counter()
        out_time = runner.process_chunk_in_place(x_spec, x_time)
        dt = (time.perf_counter() - t0) * 1000.0
        chunk_times.append(dt)

        print(f"  Chunk {idx + 1:2d}/{num_chunks}: {dt:7.2f} ms | Chunk Duration: {CHUNK_DURATION:.2f}s | Speed: {CHUNK_DURATION / (dt / 1000.0):.2f}x Real-Time")

    total_time_sec = time.perf_counter() - t_start
    mean_chunk_ms = np.mean(chunk_times)
    median_chunk_ms = np.median(chunk_times)
    min_chunk_ms = np.min(chunk_times)
    max_chunk_ms = np.max(chunk_times)

    # Real-Time Factor (RTF) = Total Processing Time / Total Audio Duration
    rtf = total_time_sec / duration_sec
    speedup = duration_sec / total_time_sec

    print("-" * 76)
    print("RESULTS:")
    print(f"  Total Audio Duration:      {duration_sec:.2f} seconds")
    print(f"  Total Wall-Clock Runtime:  {total_time_sec:.2f} seconds")
    print(f"  Mean Latency per Chunk:    {mean_chunk_ms:.2f} ms (Min: {min_chunk_ms:.2f} ms, Max: {max_chunk_ms:.2f} ms)")
    print(f"  Real-Time Factor (RTF):    {rtf:.3f}x")
    if rtf < 1.0:
        print(f"  STATUS:                    FASTER THAN REAL-TIME ({speedup:.2f}x Real-Time Speed!)")
    else:
        print(f"  STATUS:                    SLOWER THAN REAL-TIME ({rtf:.2f}x RTF)")
    print("=" * 76)
    return {
        "duration": duration_sec,
        "runtime": total_time_sec,
        "rtf": rtf,
        "speedup": speedup,
        "mean_chunk_ms": mean_chunk_ms,
    }


def main():
    print("=" * 76)
    print("AMD Ryzen AI (XDNA 2) NPU: Long-Form Audio Real-Time Benchmark")
    print(f"Model: {model_path}")
    print(f"Hardware: AMD Ryzen AI NPU (AIE2P 2x4x4 Tile Array)")
    print("=" * 76)

    runner = StreamDemucsNpuRunner(
        model_path=model_path,
        vaip_config=cfg,
        cache_dir=cache_dir,
        cache_key=cache_key,
        quark_dll=quark_dll,
    )

    # Test cases:
    # 1. 15.6-second audio file (2 chunks of 7.8s)
    # 2. 31.2-second audio file (4 chunks of 7.8s)
    # 3. 62.4-second audio file (8 chunks of 7.8s)
    durations = [15.6, 31.2, 62.4]

    results = []
    for dur in durations:
        res = benchmark_audio_stream(runner, duration_sec=dur, overlap=0.0)
        results.append(res)
        time.sleep(2.0)  # Thermal recovery between test suites

    print("\n" + "=" * 76)
    print("LONG AUDIO BENCHMARK EXECUTIVE SUMMARY")
    print("=" * 76)
    print(f"{'Audio Duration':18s} | {'Total Runtime':16s} | {'Mean Chunk':14s} | {'RTF':10s} | {'Throughput'}")
    print("-" * 76)
    for r in results:
        print(
            f"{r['duration']:6.1f} seconds      | "
            f"{r['runtime']:6.2f} seconds     | "
            f"{r['mean_chunk_ms']:7.1f} ms     | "
            f"{r['rtf']:5.3f}x    | "
            f"{r['speedup']:.2f}x Real-Time"
        )
    print("=" * 76)


if __name__ == "__main__":
    main()

