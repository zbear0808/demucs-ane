import argparse
import ctypes
import gc
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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


class IoBindingSlot:
    """Pre-allocated, 64-byte aligned IOBinding buffer slot for zero-copy NPU dispatch."""

    def __init__(self, sess: ort.InferenceSession):
        self.io_binding = sess.io_binding()
        self.input_buffers: dict[str, np.ndarray] = {}
        self.output_buffers: dict[str, np.ndarray] = {}

        for inp in sess.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            buf = self._alloc_aligned_f32(shape)
            ort_val = ort.OrtValue.ortvalue_from_numpy(buf, "cpu", 0)
            self.input_buffers[inp.name] = buf
            self.io_binding.bind_ortvalue_input(inp.name, ort_val)

        for out in sess.get_outputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in out.shape]
            buf = self._alloc_aligned_f32(shape)
            ort_val = ort.OrtValue.ortvalue_from_numpy(buf, "cpu", 0)
            self.output_buffers[out.name] = buf
            self.io_binding.bind_ortvalue_output(out.name, ort_val)

    @staticmethod
    def _alloc_aligned_f32(shape: list[int], alignment: int = 64) -> np.ndarray:
        n_bytes = int(np.prod(shape)) * 4
        raw = np.zeros(n_bytes + alignment, dtype=np.uint8)
        offset = (-raw.ctypes.data) % alignment
        return raw[offset : offset + n_bytes].view(np.float32).reshape(shape)


class PipelinedNpuDrumsSeparator:
    """
    Continuous Multi-Stage Pipelined Drums Separator:
    - Multi-threaded Lookahead CPU Pre-processing
    - Ping-Pong / Multi-Slot Zero-Copy NPU Hardware Execution
    - Background Asynchronous CPU Post-processing & Overlap-Add
    """

    def __init__(
        self,
        num_slots: int = 2,
        prep_workers: int = 4,
        lookahead_depth: int = 1,
        post_workers: int = 2,
        model: str = model_path,
        npu_cache_dir: str = cache_dir,
        npu_cache_key: str = cache_key,
        tail: str | None = None,
    ):
        self.num_slots = max(2, num_slots)
        self.prep_workers = max(1, prep_workers)
        self.lookahead_depth = max(1, lookahead_depth)
        self.post_workers = max(1, post_workers)

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

        print("[INIT] Loading AMD Ryzen AI NPU Session from AIE Hardware Cache...")
        t0 = time.perf_counter()
        self.sess = ort.InferenceSession(
            model,
            sess_options=so,
            providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            provider_options=[
                {"config_file": cfg, "cacheDir": npu_cache_dir, "cacheKey": npu_cache_key},
                {},
            ],
            disabled_optimizers=["NchwcTransformer"],
        )
        print(f"[INIT] Session loaded in {time.perf_counter() - t0:.2f}s!")

        # Optional FP32 CPU tail (decoder.3), see split_cpu_tail.py
        self.tail_sess = None
        self.tail_input = None
        if tail:
            tail_so = ort.SessionOptions()
            # 2 threads: ~300 ms per chunk for ~0.6 core-seconds; more threads barely help
            tail_so.intra_op_num_threads = 2
            self.tail_sess = ort.InferenceSession(tail, sess_options=tail_so, providers=["CPUExecutionProvider"])
            self.tail_input = self.tail_sess.get_inputs()[0].name
            print(f"[INIT] CPU tail loaded: {tail} (input '{self.tail_input}')")

        # Initialize multi-buffer IOBinding slots
        print(f"[INIT] Allocating {self.num_slots} zero-copy IOBinding hardware slots...")
        self.slots = [IoBindingSlot(self.sess) for _ in range(self.num_slots)]

        # Warmup slot 0
        print("[INIT] Running initial hardware warmup pass...")
        self.sess.run_with_iobinding(self.slots[0].io_binding)
        self.slots[0].io_binding.synchronize_outputs()

        print("[INIT] Warmup complete. Evicting dead initialization pages from physical RAM...")
        trim_working_set()
        print(
            f"[INIT] Pipeline Ready! Pinned Slots: {self.num_slots} | "
            f"Prep Workers: {self.prep_workers} | Lookahead Depth: {self.lookahead_depth} | "
            f"Resident RAM: {get_rss_mb():.1f} MB"
        )

    def separate_track(self, audio: np.ndarray, overlap: float = 0.25) -> np.ndarray:
        """
        Executes pipelined overlap-add drums separation:
        - Producer threads precompute STFT & norm into lookahead queue.
        - Hardware thread executes NPU back-to-back across slots.
        - Consumer threads synthesize iSTFT and accumulate overlap-add in parallel.
        """
        total_length = audio.shape[-1]
        stride = int((1.0 - overlap) * SEGMENT_LEN)

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
        accum_lock = threading.Lock()

        # Queues for decoupled producer-consumer pipeline
        prep_queue = queue.Queue(maxsize=self.lookahead_depth)
        post_queue = queue.Queue(maxsize=max(2, self.lookahead_depth))

        # -----------------------------------------------------------------
        # Stage 1: Producer - Preprocessing in background thread
        # -----------------------------------------------------------------
        def prep_worker_fn():
            for idx in range(num_chunks):
                off = idx * stride
                chk = padded_audio[np.newaxis, :, off : off + SEGMENT_LEN]
                spec, time_b, st = cpu_preproc_numpy(chk)
                prep_queue.put((idx, off, spec, time_b, st))
            prep_queue.put(None)  # Sentinel

        prep_thread = threading.Thread(target=prep_worker_fn, daemon=True)
        prep_thread.start()

        # -----------------------------------------------------------------
        # Stage 3: Consumer - Postprocessing in background thread
        # -----------------------------------------------------------------
        def post_worker_fn():
            while True:
                item = post_queue.get()
                if item is None:
                    post_queue.task_done()
                    break
                idx, off, spec_copy, time_copy, st = item
                if self.tail_sess is not None:
                    spec_copy = self.tail_sess.run(None, {self.tail_input: spec_copy})[0]
                pred = cpu_postproc_numpy(spec_copy, time_copy, st)[0]
                with accum_lock:
                    out_drums[:, off : off + SEGMENT_LEN] += pred * weight
                    sum_weight[off : off + SEGMENT_LEN] += weight
                del spec_copy, time_copy, pred
                post_queue.task_done()

        post_thread = threading.Thread(target=post_worker_fn, daemon=True)
        post_thread.start()

        # -----------------------------------------------------------------
        # Stage 2: Hardware NPU Dispatcher (Concurrent Multi-Threaded Slots)
        # -----------------------------------------------------------------
        # Running concurrent worker threads across distinct IOBinding slots
        # eliminates the 1.35s CPU tencoder/tdecoder gap: while Slot 0 is doing
        # CPU glue, Slot 1 is executing on the NPU, keeping hardware at 100% duty cycle.
        print_lock = threading.Lock()
        completed_chunks = 0

        def npu_slot_worker(slot_k: int):
            nonlocal completed_chunks
            curr_slot = self.slots[slot_k]

            while True:
                item = prep_queue.get()
                if item is None:
                    # Put back sentinel for other workers and exit
                    prep_queue.put(None)
                    prep_queue.task_done()
                    break

                chunk_idx, offset, norm_spec, norm_time, stats = item
                prep_queue.task_done()

                t_npu_start = time.perf_counter()

                # Zero-copy buffer fill into this slot's distinct memory
                np.copyto(curr_slot.input_buffers["x_spec"], norm_spec)
                np.copyto(curr_slot.input_buffers["x_time"], norm_time)

                # Hardware inference on AMD Ryzen AI NPU
                self.sess.run_with_iobinding(curr_slot.io_binding)

                dt_npu = (time.perf_counter() - t_npu_start) * 1000.0

                # Copy output tensors before passing slot to next iteration.
                # With a split model the body emits the decoder.3 input instead of out_spec.
                spec_key = "out_spec" if self.tail_sess is None else self.tail_input
                out_spec_copy = curr_slot.output_buffers[spec_key].copy()
                out_time_copy = curr_slot.output_buffers["out_time"].copy()

                # Dispatch to asynchronous post-processing consumer
                post_queue.put((chunk_idx, offset, out_spec_copy, out_time_copy, stats))

                speed = (SEGMENT_LEN / SAMPLE_RATE) / (dt_npu / 1000.0)
                with print_lock:
                    completed_chunks += 1
                    q_ready = prep_queue.qsize()
                    q_post = post_queue.qsize()
                    print(
                        f"    [Zero-Wait NPU] Chunk {chunk_idx+1:2d}/{num_chunks:2d} (Slot {slot_k}): "
                        f"{dt_npu:6.1f} ms ({speed:.2f}x RT) | Ready in Queue: {q_ready:2d} | In Post: {q_post:2d} | "
                        f"Progress: {completed_chunks}/{num_chunks} | RAM: {get_rss_mb():5.1f} MB",
                        flush=True,
                    )

        # Launch concurrent NPU slot workers with staggered start
        npu_threads = []
        for s_idx in range(self.num_slots):
            th = threading.Thread(target=npu_slot_worker, args=(s_idx,), daemon=True)
            npu_threads.append(th)
            th.start()
            if s_idx < self.num_slots - 1:
                # Phase-shift threads by ~1.2s to interleave CPU pre/post with AIE compute
                time.sleep(1.2)

        for th in npu_threads:
            th.join()

        # Signal postprocessing thread to finish
        post_queue.put(None)
        post_thread.join()
        prep_thread.join()

        # Final normalization & unpadding
        mask = sum_weight > 1e-8
        out_drums[:, mask] /= sum_weight[mask]
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
    audio = raw.reshape(-1, 2).T
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


def parse_args():
    parser = argparse.ArgumentParser(description="Pipelined Multi-Buffer NPU Drums Batch Separator")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"C:\Users\zubair\Music\skrillex leaks\LEAKS",
        help="Input folder containing audio tracks",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"C:\Users\zubair\Music\skrillex leaks\LEAKS\npu_drums_pipelined",
        help="Output folder for separated drum stems",
    )
    parser.add_argument(
        "--num-slots",
        type=int,
        default=2,
        help="Number of alternating IOBinding hardware slots (default: 2)",
    )
    parser.add_argument(
        "--prep-workers",
        type=int,
        default=4,
        help="Number of CPU worker threads for lookahead chunk pre-processing (default: 4)",
    )
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        default=1,
        help="Maximum pre-processed chunks buffered ahead of the NPU (default: 1 for minimal RAM)",
    )
    parser.add_argument(
        "--post-workers",
        type=int,
        default=2,
        help="Number of CPU worker threads for iSTFT post-processing (default: 2)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Chunk overlap ratio (default: 0.25 for Demucs)",
    )
    parser.add_argument(
        "--single-file",
        type=str,
        default=None,
        help="Optionally process a single audio file instead of the whole directory",
    )
    parser.add_argument("--model", type=str, default=model_path, help="ONNX model to run")
    parser.add_argument("--cache-dir", type=str, default=cache_dir, help="Vitis AI EP cache directory")
    parser.add_argument("--cache-key", type=str, default=cache_key, help="Vitis AI EP cache key")
    parser.add_argument(
        "--tail", type=str, default=None,
        help="Optional FP32 CPU tail model (split_cpu_tail.py) when --model is an NPU body",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.single_file:
        track_paths = [args.single_file]
    else:
        extensions = (".wav", ".mp3", ".flac", ".m4a")
        track_paths = [
            os.path.join(args.input_dir, f)
            for f in sorted(os.listdir(args.input_dir))
            if f.lower().endswith(extensions) and not f.startswith(".")
        ]

    print("=" * 80)
    print("PIPELINED AMD RYZEN AI NPU BATCH DRUMS SEPARATOR")
    print(f"Target Directory     : {args.input_dir}")
    print(f"Output Directory     : {args.output_dir}")
    print(f"Total Tracks Found   : {len(track_paths)}")
    print(f"Pipeline Config      : {args.num_slots} IOBinding Slots | {args.prep_workers} Prep Workers | "
          f"{args.lookahead_depth} Lookahead Depth | {args.post_workers} Post Workers")
    print("=" * 80)

    separator = PipelinedNpuDrumsSeparator(
        num_slots=args.num_slots,
        prep_workers=args.prep_workers,
        lookahead_depth=args.lookahead_depth,
        post_workers=args.post_workers,
        model=args.model,
        npu_cache_dir=args.cache_dir,
        npu_cache_key=args.cache_key,
        tail=args.tail,
    )

    total_batch_start = time.perf_counter()
    total_audio_sec = 0.0

    for idx, path in enumerate(track_paths, 1):
        filename = os.path.basename(path)
        stem_name = os.path.splitext(filename)[0]
        out_filename = f"{stem_name} (Drums).wav"
        out_path = os.path.join(args.output_dir, out_filename)

        print(f"\n[{idx:2d}/{len(track_paths):2d}] Processing: {filename}")
        t_load = time.perf_counter()
        audio, sr = load_audio(path)
        dur_sec = audio.shape[-1] / sr
        total_audio_sec += dur_sec
        print(f"  Duration: {dur_sec:.1f}s ({dur_sec/60:.2f} min) | Loaded in {time.perf_counter() - t_load:.2f}s")

        t_sep = time.perf_counter()
        drums = separator.separate_track(audio, overlap=args.overlap)
        sep_time = time.perf_counter() - t_sep

        rtf = sep_time / dur_sec
        speedup = dur_sec / sep_time
        print(f"  Separated in {sep_time:.1f}s | RTF: {rtf:.3f}x ({speedup:.2f}x Real-Time Speed)")

        t_save = time.perf_counter()
        save_audio_wav(out_path, drums, sr)
        print(f"  Saved -> {out_filename} ({time.perf_counter() - t_save:.2f}s)")

    total_wall_sec = time.perf_counter() - total_batch_start
    print("\n" + "=" * 80)
    print("PIPELINED BATCH SEPARATION COMPLETE!")
    print(f"Total Tracks Processed : {len(track_paths)}")
    print(f"Total Audio Duration   : {total_audio_sec:.1f}s ({total_audio_sec/60:.2f} min)")
    print(f"Total Processing Time  : {total_wall_sec:.1f}s ({total_wall_sec/60:.2f} min)")
    print(f"Overall Average Speed  : {total_audio_sec / total_wall_sec:.2f}x Real-Time")
    print(f"Output Destination     : {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

