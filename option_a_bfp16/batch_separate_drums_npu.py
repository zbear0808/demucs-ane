import ctypes
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import psutil

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

from htdemucs_dsp import (
    SAMPLE_RATE,
    SEGMENT_LEN,
    cpu_postproc_numpy,
    cpu_preproc_numpy,
)

# Windows PSAPI for memory trimming
kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi
kernel32.GetCurrentProcess.restype = ctypes.c_void_p
psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
psapi.EmptyWorkingSet.restype = ctypes.c_int


def trim_working_set():
    gc.collect()
    h_proc = kernel32.GetCurrentProcess()
    psapi.EmptyWorkingSet(h_proc)


def get_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


class NpuDrumsBatchSeparator:
    def __init__(self):
        so = ort.SessionOptions()
        so.log_severity_level = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        so.enable_mem_reuse = True
        so.intra_op_num_threads = 8
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        so.add_session_config_entry("session.disable_prepacking", "1")
        so.register_custom_ops_library(quark_dll)

        print("[INIT] Loading AMD Ryzen AI NPU Session from AIE Cache...")
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
        print(f"[INIT] Loaded in {time.perf_counter() - t0:.2f}s!")

        # Setup persistent pinned OrtValues
        self.io_binding = self.sess.io_binding()
        self.input_buffers: dict[str, np.ndarray] = {}
        self.output_buffers: dict[str, np.ndarray] = {}

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

        # Hardware warmup & memory trim
        print("[INIT] Running initial hardware warmup...")
        self.sess.run_with_iobinding(self.io_binding)
        self.io_binding.synchronize_outputs()
        print("[INIT] Warmup complete. Trimming dead initialization pages from RAM...")
        trim_working_set()
        print(f"[INIT] Ready for batch separation! Physical RAM: {get_rss_mb():.1f} MB")

    @staticmethod
    def _alloc_aligned_f32(shape: list[int], alignment: int = 64) -> np.ndarray:
        n_bytes = int(np.prod(shape)) * 4
        raw = np.empty(n_bytes + alignment, dtype=np.uint8)
        offset = (-raw.ctypes.data) % alignment
        return raw[offset : offset + n_bytes].view(np.float32).reshape(shape)

    def separate_track(self, audio: np.ndarray, overlap: float = 0.25) -> np.ndarray:
        """
        Runs full-track drums separation with smooth triangular overlap-add.
        audio: [2, total_samples] @ 44.1 kHz float32
        Returns: [2, total_samples] separated drums
        """
        total_length = audio.shape[-1]
        stride = int((1.0 - overlap) * SEGMENT_LEN)

        # Global audio normalization
        ref_mean = audio.mean()
        ref_std = audio.std() + 1e-8
        norm_audio = (audio - ref_mean) / ref_std

        num_chunks = 1 if total_length <= SEGMENT_LEN else int(np.ceil((total_length - SEGMENT_LEN) / stride)) + 1
        pad_len = (num_chunks - 1) * stride + SEGMENT_LEN - total_length
        if pad_len > 0:
            padded_audio = np.pad(norm_audio, ((0, 0), (0, pad_len)))
        else:
            padded_audio = norm_audio

        # Triangular overlap-add window
        half_seg = SEGMENT_LEN // 2
        ramp = np.arange(1, half_seg + 1, dtype=np.float32)
        weight = np.concatenate([ramp, ramp[::-1]])
        weight = (weight / weight.max()).astype(np.float32)

        out_drums = np.zeros_like(padded_audio, dtype=np.float32)
        sum_weight = np.zeros(padded_audio.shape[-1], dtype=np.float32)

        for chunk_idx in range(num_chunks):
            offset = chunk_idx * stride
            chunk = padded_audio[np.newaxis, :, offset : offset + SEGMENT_LEN]

            t0 = time.perf_counter()
            # 1. CPU Pre-processing
            norm_spec, norm_time, stats = cpu_preproc_numpy(chunk)

            # 2. XDNA 2 NPU Inference via Zero-Copy IOBinding
            np.copyto(self.input_buffers["x_spec"], norm_spec)
            np.copyto(self.input_buffers["x_time"], norm_time)
            self.sess.run_with_iobinding(self.io_binding)

            # 3. CPU Post-processing
            pred_chunk = cpu_postproc_numpy(
                self.output_buffers["out_spec"],
                self.output_buffers["out_time"],
                stats,
            )[0]  # [2, 344064]

            # Overlap-add
            out_drums[:, offset : offset + SEGMENT_LEN] += pred_chunk * weight
            sum_weight[offset : offset + SEGMENT_LEN] += weight
            dt = (time.perf_counter() - t0) * 1000.0

            speed = (SEGMENT_LEN / SAMPLE_RATE) / (dt / 1000.0)
            print(
                f"    Chunk {chunk_idx+1:2d}/{num_chunks:2d}: {dt:6.1f} ms ({speed:.2f}x RT) | RAM: {get_rss_mb():5.1f} MB",
                flush=True,
            )

        mask = sum_weight > 1e-8
        out_drums[:, mask] /= sum_weight[mask]

        # Denormalize with global stats
        drums = out_drums[:, :total_length] * ref_std + ref_mean
        return drums


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Decodes any audio file (wav, mp3, flac) to 44.1 kHz stereo float32 via FFmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "panic",
        "-i",
        path,
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "2",
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    audio = raw.reshape(-1, 2).T  # [2, num_samples]
    return audio, SAMPLE_RATE


def save_audio_wav(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Encodes stereo float32 audio to 16-bit PCM WAV using FFmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "panic",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        "2",
        "-i",
        "pipe:0",
        "-c:a",
        "pcm_s16le",
        path,
    ]
    raw = audio.T.astype(np.float32).tobytes()
    subprocess.run(cmd, input=raw, check=True)


def main():
    input_folder = r"C:\Users\zubair\Music\skrillex leaks\LEAKS"
    output_folder = os.path.join(input_folder, "npu_drums")
    os.makedirs(output_folder, exist_ok=True)

    extensions = (".wav", ".mp3", ".flac", ".m4a")
    track_paths = [
        os.path.join(input_folder, f)
        for f in sorted(os.listdir(input_folder))
        if f.lower().endswith(extensions) and not f.startswith(".")
    ]

    print("=" * 80)
    print("AMD RYZEN AI NPU BATCH DRUMS SEPARATOR")
    print(f"Target Directory: {input_folder}")
    print(f"Output Directory: {output_folder}")
    print(f"Total Tracks Found: {len(track_paths)}")
    print("=" * 80)

    separator = NpuDrumsBatchSeparator()

    total_batch_start = time.perf_counter()
    total_audio_sec = 0.0

    for idx, path in enumerate(track_paths, 1):
        filename = os.path.basename(path)
        stem_name = os.path.splitext(filename)[0]
        out_filename = f"{stem_name} (Drums).wav"
        out_path = os.path.join(output_folder, out_filename)

        print(f"\n[{idx:2d}/{len(track_paths):2d}] Processing: {filename}")
        t_load = time.perf_counter()
        audio, sr = load_audio(path)
        dur_sec = audio.shape[-1] / sr
        total_audio_sec += dur_sec
        print(f"  Duration: {dur_sec:.1f}s ({dur_sec/60:.2f} min) | Loaded in {time.perf_counter() - t_load:.2f}s")

        t_sep = time.perf_counter()
        drums = separator.separate_track(audio, overlap=0.25)
        sep_time = time.perf_counter() - t_sep

        rtf = sep_time / dur_sec
        speedup = dur_sec / sep_time
        print(f"  Separated in {sep_time:.1f}s | RTF: {rtf:.3f}x ({speedup:.2f}x Real-Time Speed)")

        t_save = time.perf_counter()
        save_audio_wav(out_path, drums, sr)
        print(f"  Saved -> {out_filename} ({time.perf_counter() - t_save:.2f}s)")

    total_wall_sec = time.perf_counter() - total_batch_start
    print("\n" + "=" * 80)
    print("BATCH SEPARATION COMPLETE!")
    print(f"Total Tracks Processed : {len(track_paths)}")
    print(f"Total Audio Duration   : {total_audio_sec:.1f}s ({total_audio_sec/60:.2f} min)")
    print(f"Total Processing Time  : {total_wall_sec:.1f}s ({total_wall_sec/60:.2f} min)")
    print(f"Overall Average Speed  : {total_audio_sec / total_wall_sec:.2f}x Real-Time")
    print(f"Output Destination     : {output_folder}")
    print("=" * 80)


if __name__ == "__main__":
    main()

