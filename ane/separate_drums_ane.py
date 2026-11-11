"""
Batch drums separation on the Apple Neural Engine.

Same chunking / overlap-add / DSP as option_a_bfp16/batch_separate_drums_npu.py, with the
NPU session swapped for the Core ML model (or the fp16 ONNX via ORT's CoreML EP).
A small pipeline overlaps CPU STFT/iSTFT with the ANE run of the neighbouring chunk.

  .venv/bin/python ane/separate_drums_ane.py --input-dir ~/Music/songs
  .venv/bin/python ane/separate_drums_ane.py --single-file song.wav
  .venv/bin/python ane/separate_drums_ane.py --single-file song.wav --onnx ane/models/htdemucs_drums_ane_fp16.onnx
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ane_tools import (
    ANE_DIR, COMPUTE_UNITS, DEFAULT_MLPACKAGE, SAMPLE_RATE, SEGMENT_LEN, cpu_postproc_numpy, cpu_preproc_numpy,
    fold_time, unfold_time,
)
from parity_ane import CoreMLRunner, OnnxCoreMLRunner

EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".aiff", ".ogg")


def load_audio(path: str) -> np.ndarray:
    """Decode any audio file to 44.1 kHz stereo float32 [2, n] via FFmpeg."""
    cmd = ["ffmpeg", "-loglevel", "panic", "-i", path, "-ar", str(SAMPLE_RATE), "-ac", "2", "-f", "f32le", "pipe:1"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def save_wav(path: str, audio: np.ndarray):
    cmd = ["ffmpeg", "-y", "-loglevel", "panic", "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "2", "-i", "pipe:0",
           "-c:a", "pcm_s16le", path]
    subprocess.run(cmd, input=audio.T.astype(np.float32).tobytes(), check=True)


def separate_track(run, audio: np.ndarray, overlap: float = 0.25, verbose: bool = True) -> np.ndarray:
    """audio: [2, n] -> separated drums [2, n] (triangular-window overlap-add)."""
    total = audio.shape[-1]
    stride = int((1.0 - overlap) * SEGMENT_LEN)
    ref_mean, ref_std = audio.mean(), audio.std() + 1e-8
    norm = (audio - ref_mean) / ref_std

    n_chunks = 1 if total <= SEGMENT_LEN else int(np.ceil((total - SEGMENT_LEN) / stride)) + 1
    pad = (n_chunks - 1) * stride + SEGMENT_LEN - total
    padded = np.pad(norm, ((0, 0), (0, max(pad, 0))))

    ramp = np.arange(1, SEGMENT_LEN // 2 + 1, dtype=np.float32)
    weight = np.concatenate([ramp, ramp[::-1]])
    weight /= weight.max()
    out = np.zeros_like(padded)
    sum_w = np.zeros(padded.shape[-1], dtype=np.float32)

    def prep(idx):
        return cpu_preproc_numpy(padded[np.newaxis, :, idx * stride : idx * stride + SEGMENT_LEN])

    def post(idx, spec, time_, stats):
        wav = cpu_postproc_numpy(spec, unfold_time(time_), stats)[0]
        off = idx * stride
        out[:, off : off + SEGMENT_LEN] += wav * weight
        sum_w[off : off + SEGMENT_LEN] += weight

    # Prep of chunk i+1 and post of chunk i-1 run on CPU threads while chunk i is on the ANE.
    with ThreadPoolExecutor(max_workers=2) as pool:
        nxt = pool.submit(prep, 0)
        pending = None
        for i in range(n_chunks):
            t0 = time.perf_counter()
            x_spec, x_time, stats = nxt.result()
            if i + 1 < n_chunks:
                nxt = pool.submit(prep, i + 1)
            spec, time_ = run(np.ascontiguousarray(x_spec, dtype=np.float32), fold_time(x_time))
            if pending is not None:
                pending.result()
            pending = pool.submit(post, i, spec, time_, stats)
            if verbose:
                dt = time.perf_counter() - t0
                print(f"    chunk {i + 1:3d}/{n_chunks}: {dt * 1000:6.0f} ms ({SEGMENT_LEN / SAMPLE_RATE / dt:5.1f}x RT)",
                      flush=True)
        pending.result()

    mask = sum_w > 1e-8
    out[:, mask] /= sum_w[mask]
    return out[:, :total] * ref_std + ref_mean


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-dir")
    src.add_argument("--single-file")
    p.add_argument("--output-dir", default=None, help="Default: <input>/ane_drums")
    p.add_argument("--model", default=str(DEFAULT_MLPACKAGE))
    p.add_argument("--onnx", default=None, help="Use an ONNX model via ORT's CoreML EP instead of the .mlpackage")
    p.add_argument("--units", choices=COMPUTE_UNITS, default="CPU_AND_NE")
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    if args.single_file:
        tracks = [os.path.abspath(os.path.expanduser(args.single_file))]
        base_dir = os.path.dirname(tracks[0])
    else:
        base_dir = os.path.abspath(os.path.expanduser(args.input_dir))
        tracks = [os.path.join(base_dir, f) for f in sorted(os.listdir(base_dir))
                  if f.lower().endswith(EXTENSIONS) and not f.startswith(".")]
    out_dir = args.output_dir or os.path.join(base_dir, "ane_drums")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    if args.onnx:
        run = OnnxCoreMLRunner(args.onnx, args.units, cache_dir=ANE_DIR / "models" / "ort_coreml_cache")
    else:
        run = CoreMLRunner(args.model, args.units)
    print(f"[INIT] {args.onnx or args.model} [{args.units}] loaded in {time.perf_counter() - t0:.1f}s")

    for n, path in enumerate(tracks, 1):
        name = Path(path).stem
        print(f"\n[{n}/{len(tracks)}] {name}")
        audio = load_audio(path)
        t0 = time.perf_counter()
        drums = separate_track(run, audio, args.overlap, verbose=not args.quiet)
        dt = time.perf_counter() - t0
        dur = audio.shape[-1] / SAMPLE_RATE
        dest = os.path.join(out_dir, f"{name}_drums.wav")
        save_wav(dest, drums)
        print(f"  {dur:.1f}s of audio in {dt:.1f}s ({dur / dt:.1f}x real-time) -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
