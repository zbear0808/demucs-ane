"""
Batch stem separation (drums, vocals, ...) on the Apple Neural Engine.

Same chunking / overlap-add / DSP as option_a_bfp16/batch_separate_drums_npu.py, with the
NPU session swapped for the Core ML model (or the fp16 ONNX via ORT's CoreML EP).
A small pipeline overlaps CPU STFT/iSTFT with the ANE run of the neighbouring chunk.

  .venv/bin/python ane/separate_ane.py --input-dir ~/Music/songs
  .venv/bin/python ane/separate_ane.py --single-file song.wav --source vocals
  .venv/bin/python ane/separate_ane.py --single-file song.wav --onnx ane/models/htdemucs_drums_ane_fp16.onnx
  .venv/bin/python ane/separate_ane.py --single-file song.wav --dsp fused        # whole pipeline in one ANE program
  .venv/bin/python ane/separate_ane.py --single-file song.wav --dsp fused \\
      --onnx ane/models/htdemucs_drums_fused_ane_fp16.onnx                      # same, via ONNX Runtime
"""
import argparse
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import dsp_ane as dsp
from ane_tools import (
    ANE_DIR, COMPUTE_UNITS, FUSED_IO, MODELS_DIR, SAMPLE_RATE, SEGMENT_LEN, SPLIT_IO, cpu_postproc_numpy,
    cpu_preproc_numpy, fold_time, fused_mlpackage_path, load_coreml, mlpackage_path, unfold_time,
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


class CpuDSP:
    """numpy STFT/iSTFT (htdemucs_dsp.py)."""

    def prep(self, chunk):
        x_spec, x_time, stats = cpu_preproc_numpy(chunk)
        return np.ascontiguousarray(x_spec, dtype=np.float32), fold_time(x_time), stats

    def post(self, spec, time_, stats):
        return cpu_postproc_numpy(spec, unfold_time(time_), stats)[0]


class AneDSP:
    """STFT/iSTFT as Core ML models on the ANE (dsp_ane.py); only padding and the crop stay on the CPU."""

    def __init__(self, units):
        self.stft = load_coreml(MODELS_DIR / "htdemucs_stft_ane.mlpackage", units)
        self.istft = load_coreml(MODELS_DIR / "htdemucs_istft_ane.mlpackage", units)

    def prep(self, chunk):
        s = self.stft.predict({"blocks": dsp.reflect_pad_numpy(chunk)})
        x_time, mean_t, std_t = dsp.time_stats_numpy(chunk)
        return s["x_spec"], fold_time(x_time), (s["mean_s"], s["std_s"], mean_t, std_t)

    def post(self, spec, time_, stats):
        mean_s, std_s, mean_t, std_t = stats
        w = self.istft.predict({"out_spec": spec, "mean_s": mean_s, "std_s": std_s})["wav_blocks"]
        return (dsp.crop_istft(w) + unfold_time(time_)[:, :, 0, :] * std_t + mean_t)[0]


class FusedDSP:
    """
    For the fused model (STFT + separation + iSTFT in one ANE program). The CPU only does the
    reflect padding and time-branch statistics before the call, and the crop and the final
    add after it: about 1 ms per chunk.
    """

    def prep(self, chunk):
        x_time, mean_t, std_t = dsp.time_stats_numpy(chunk)
        return dsp.reflect_pad_numpy(chunk), fold_time(x_time), (mean_t, std_t)

    def post(self, wav_blocks, out_time, stats):
        mean_t, std_t = stats
        return (dsp.crop_istft(wav_blocks) + unfold_time(out_time)[:, :, 0, :] * std_t + mean_t)[0]


def separate_track(run, audio: np.ndarray, overlap: float = 0.25, verbose: bool = True, dsp_impl=None,
                   timing: dict = None) -> np.ndarray:
    """audio: [2, n] -> separated stem [2, n] (triangular-window overlap-add). `timing` collects per-chunk ms."""
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

    dsp_impl = dsp_impl or CpuDSP()

    def prep(idx):
        return dsp_impl.prep(padded[np.newaxis, :, idx * stride : idx * stride + SEGMENT_LEN])

    def post(idx, spec, time_, stats):
        wav = dsp_impl.post(spec, time_, stats)
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
            spec, time_ = run(x_spec, x_time)
            if pending is not None:
                pending.result()
            pending = pool.submit(post, i, spec, time_, stats)
            dt = time.perf_counter() - t0
            if timing is not None:
                timing.setdefault("chunk_ms", []).append(1000 * dt)
            if verbose:
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
    p.add_argument("--source", default="drums", help="Stem name: picks the default --model and output naming")
    p.add_argument("--output-dir", default=None, help="Default: <input>/ane_<source>")
    p.add_argument("--model", default=None,
                   help="Default: ane/models/htdemucs_<source>_ane.mlpackage (…_fused_ane.mlpackage with --dsp fused)")
    p.add_argument("--onnx", default=None, help="Use an ONNX model via ORT's CoreML EP instead of the .mlpackage")
    p.add_argument("--units", choices=COMPUTE_UNITS, default="CPU_AND_NE")
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--dsp", choices=("cpu", "ane", "fused"), default="cpu",
                   help="Where the STFT/iSTFT run: cpu = numpy; ane = separate ANE models; "
                        "fused = one ANE program with the whole pipeline (models come from benchmark_pipeline.py)")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()
    fused = args.dsp == "fused"
    args.model = args.model or str(fused_mlpackage_path(args.source) if fused else mlpackage_path(args.source))

    if args.single_file:
        tracks = [os.path.abspath(os.path.expanduser(args.single_file))]
        base_dir = os.path.dirname(tracks[0])
    else:
        base_dir = os.path.abspath(os.path.expanduser(args.input_dir))
        tracks = [os.path.join(base_dir, f) for f in sorted(os.listdir(base_dir))
                  if f.lower().endswith(EXTENSIONS) and not f.startswith(".")]
    out_dir = args.output_dir or os.path.join(base_dir, f"ane_{args.source}")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    if args.onnx:
        run = OnnxCoreMLRunner(args.onnx, args.units, cache_dir=ANE_DIR / "models" / "ort_coreml_cache")
    else:
        run = CoreMLRunner(args.model, args.units, FUSED_IO if fused else SPLIT_IO)
    dsp_impl = {"ane": lambda: AneDSP(args.units), "fused": FusedDSP, "cpu": CpuDSP}[args.dsp]()
    print(f"[INIT] {args.onnx or args.model} [{args.units}, {args.dsp} STFT] loaded in {time.perf_counter() - t0:.1f}s")

    for n, path in enumerate(tracks, 1):
        name = Path(path).stem
        print(f"\n[{n}/{len(tracks)}] {name}")
        audio = load_audio(path)
        t0, cpu0 = time.perf_counter(), time.process_time()
        st = {}
        stem = separate_track(run, audio, args.overlap, verbose=not args.quiet, dsp_impl=dsp_impl, timing=st)
        dt, cpu = time.perf_counter() - t0, time.process_time() - cpu0
        dur = audio.shape[-1] / SAMPLE_RATE
        dest = os.path.join(out_dir, f"{name}_{args.source}.wav")
        save_wav(dest, stem)
        ms = np.array(st["chunk_ms"])
        print(f"  {dur:.1f}s of audio in {dt:.1f}s ({dur / dt:.1f}x real-time) -> {dest}")
        print(f"  this process used {cpu:.1f} s of CPU time ({100 * cpu / dt:.0f}% of one core) and peaked at "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.0f} MB resident memory")
        print(f"  per chunk: median {np.median(ms):.0f} ms, p90 {np.percentile(ms, 90):.0f} ms, max {ms.max():.0f} ms"
              + ("  <- p90 is well above the median: another process was probably using the ANE/CPU"
                 if np.percentile(ms, 90) > 1.25 * np.median(ms) else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
