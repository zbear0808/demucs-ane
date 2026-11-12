"""
Real-audio parity + latency check for the ANE single-source models (drums, vocals, ...).

Uses the same reference data as the XDNA2 flow (option_a_bfp16/parity_real_audio.npz,
built by `option_a_bfp16/parity_real_audio.py make-ref`) and the same metrics:
  spec / time - SNR of the raw model outputs vs the PyTorch FP32 pure-AIE core
  wav         - separated stem chunk vs stock Demucs forward()
  mix         - error level below the input mix (the PASS/FAIL criterion)

With --fused the model is the whole pipeline (STFT -> separation -> iSTFT in one program).
It takes the reflect-padded mix as blocks and returns the spectral branch as waveform
blocks, so there is no spectrogram output and the `spec` metric does not exist.

  .venv/bin/python ane/parity_ane.py                                  # Core ML on the ANE
  .venv/bin/python ane/parity_ane.py --units CPU_ONLY                 # same model, CPU
  .venv/bin/python ane/parity_ane.py --source vocals                  # vocals model + vocals reference
  .venv/bin/python ane/parity_ane.py --onnx ane/models/htdemucs_drums_ane_fp16.onnx   # ONNX via ORT CoreML EP
  .venv/bin/python ane/parity_ane.py --fused                          # fused Core ML model
  .venv/bin/python ane/parity_ane.py --fused --onnx ane/models/htdemucs_drums_fused_ane_fp16.onnx
"""
import argparse
import sys
import time

import numpy as np

import dsp_ane as dsp
from ane_tools import (
    COMPUTE_UNITS, FUSED_IO, SPLIT_IO, cpu_postproc_numpy, fold_time, fused_mlpackage_path, load_coreml, load_ref,
    mlpackage_path, ref_path, snr_db, unfold_time,
)


class CoreMLRunner:
    """Core ML model with two inputs and two outputs; `io` names them (SPLIT_IO or FUSED_IO)."""

    def __init__(self, path, units, io=SPLIT_IO):
        self.model = load_coreml(path, units)
        self.io = io

    def __call__(self, a, b):
        out = self.model.predict(dict(zip(self.io["inputs"], (a, b))))
        return tuple(out[k] for k in self.io["outputs"])


class OnnxCoreMLRunner:
    """ONNX Runtime with the CoreML execution provider (ONNX -> Core ML MLProgram at session load)."""

    def __init__(self, path, units, cache_dir=None):
        import onnxruntime as ort

        units_map = {"CPU_AND_NE": "CPUAndNeuralEngine", "ALL": "ALL", "CPU_AND_GPU": "CPUAndGPU", "CPU_ONLY": "CPUOnly"}
        opts = {"ModelFormat": "MLProgram", "MLComputeUnits": units_map[units], "RequireStaticInputShapes": "1"}
        if cache_dir:
            opts["ModelCacheDirectory"] = str(cache_dir)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=[("CoreMLExecutionProvider", opts), "CPUExecutionProvider"])
        self.dtype = np.float16 if self.sess.get_inputs()[0].type == "tensor(float16)" else np.float32
        self.in_names = [i.name for i in self.sess.get_inputs()]

    def __call__(self, a, b):
        outs = self.sess.run(None, {self.in_names[0]: a.astype(self.dtype), self.in_names[1]: b.astype(self.dtype)})
        return tuple(o.astype(np.float32) for o in outs)


def evaluate(run, ref, min_mix_snr: float) -> int:
    rows = []
    for i, song in enumerate(ref["songs"]):
        x_spec = np.ascontiguousarray(ref["x_spec"][i : i + 1])
        x_time = fold_time(ref["x_time"][i : i + 1])
        t0 = time.perf_counter()
        out_spec, out_time = run(x_spec, x_time)
        dt = time.perf_counter() - t0
        out_time = unfold_time(out_time)
        stats = tuple(ref[k][i] for k in ("mean_s", "std_s", "mean_t", "std_t"))
        wav = cpu_postproc_numpy(out_spec, out_time, stats)[0]
        row = {
            "spec": snr_db(ref["ref_spec"][i : i + 1], out_spec),
            "time": snr_db(ref["ref_time"][i : i + 1], out_time),
            "wav": snr_db(ref["stock_wav"][i], wav),
            "mix": snr_db(ref["chunk"][i], ref["chunk"][i] + (wav - ref["stock_wav"][i])),
        }
        rows.append(row)
        print(f"[{i + 1}] {song}  ({dt:.2f}s)")
        print(f"    spec {row['spec']:6.2f} dB | time {row['time']:6.2f} dB | wav vs stock {row['wav']:6.2f} dB"
              f" | error below mix {row['mix']:6.2f} dB")
    worst = {k: min(r[k] for r in rows) for k in rows[0]}
    print(f"\nWORST: spec {worst['spec']:.2f} dB | time {worst['time']:.2f} dB | wav {worst['wav']:.2f} dB"
          f" | error below mix {worst['mix']:.2f} dB")
    ok = all(np.isfinite(v) for v in worst.values()) and worst["mix"] >= min_mix_snr
    print(f"{'PASS' if ok else 'FAIL'} (threshold: error below mix >= {min_mix_snr} dB)")
    return 0 if ok else 1


def fused_chunk(run, mix):
    """Whole fused pipeline for one [1, 2, 344064] chunk: CPU pads and normalises, the model does the rest."""
    x_time, mean_t, std_t = dsp.time_stats_numpy(mix)
    wav_blocks, out_time = run(dsp.reflect_pad_numpy(mix), fold_time(x_time))
    return (dsp.crop_istft(wav_blocks) + unfold_time(out_time)[:, :, 0, :] * std_t + mean_t)[0], out_time


def evaluate_fused(run, ref, min_mix_snr: float) -> int:
    rows = []
    for i, song in enumerate(ref["songs"]):
        t0 = time.perf_counter()
        wav, out_time = fused_chunk(run, ref["chunk"][i : i + 1])
        dt = time.perf_counter() - t0
        row = {
            "time": snr_db(ref["ref_time"][i : i + 1], unfold_time(out_time)),
            "wav": snr_db(ref["stock_wav"][i], wav),
            "mix": snr_db(ref["chunk"][i], ref["chunk"][i] + (wav - ref["stock_wav"][i])),
        }
        rows.append(row)
        print(f"[{i + 1}] {song}  ({dt:.2f}s incl. CPU pad/crop)")
        print(f"    time {row['time']:6.2f} dB | wav vs stock {row['wav']:6.2f} dB | error below mix {row['mix']:6.2f} dB")
    worst = {k: min(r[k] for r in rows) for k in rows[0]}
    print(f"\nWORST: time {worst['time']:.2f} dB | wav {worst['wav']:.2f} dB | error below mix {worst['mix']:.2f} dB")
    ok = all(np.isfinite(v) for v in worst.values()) and worst["mix"] >= min_mix_snr
    print(f"{'PASS' if ok else 'FAIL'} (threshold: error below mix >= {min_mix_snr} dB)")
    return 0 if ok else 1


def benchmark(run, n: int, fused: bool = False):
    rng = np.random.default_rng(0)
    from ane_tools import SAMPLE_RATE, SEGMENT_LEN, SPEC_SHAPE, TIME_SHAPE

    a_shape = (1, 2, dsp.N_BLOCKS, dsp.HOP) if fused else SPEC_SHAPE
    a = rng.standard_normal(a_shape).astype(np.float32)
    x_time = rng.standard_normal(TIME_SHAPE).astype(np.float32)
    run(a, x_time)  # warmup
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        run(a, x_time)
        times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    what = "whole model incl. STFT/iSTFT" if fused else "model only, excludes STFT/iSTFT"
    print(f"\nModel latency over {n} runs: median {med * 1000:.0f} ms, min {min(times) * 1000:.0f} ms"
          f" -> {SEGMENT_LEN / SAMPLE_RATE / med:.1f}x real-time ({what})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="drums", help="Picks the default --model and --ref")
    p.add_argument("--fused", action="store_true", help="Fused STFT + separation + iSTFT model")
    p.add_argument("--model", default=None, help="Core ML .mlpackage (default: htdemucs_<source>_ane.mlpackage)")
    p.add_argument("--onnx", default=None, help="Evaluate an ONNX model through ORT's CoreML EP instead")
    p.add_argument("--units", choices=COMPUTE_UNITS, default="CPU_AND_NE")
    p.add_argument("--ref", default=None)
    p.add_argument("--min-mix-snr", type=float, default=40.0)
    p.add_argument("--bench", type=int, default=10, help="Latency runs after the parity check (0 = skip)")
    args = p.parse_args()
    args.model = args.model or str(fused_mlpackage_path(args.source) if args.fused else mlpackage_path(args.source))
    args.ref = args.ref or str(ref_path(args.source))

    t0 = time.perf_counter()
    if args.onnx:
        from ane_tools import MODELS_DIR

        run = OnnxCoreMLRunner(args.onnx, args.units, cache_dir=MODELS_DIR / "ort_coreml_cache")
    else:
        run = CoreMLRunner(args.model, args.units, FUSED_IO if args.fused else SPLIT_IO)
    print(f"Model: {args.onnx or args.model} [{args.units}] loaded in {time.perf_counter() - t0:.1f}s\n")

    rc = 0
    ref = load_ref(args.ref)
    if ref is None:
        print(f"[skip] no reference data at {args.ref} (export_htdemucs_ane.py builds it)")
    else:
        rc = (evaluate_fused if args.fused else evaluate)(run, ref, args.min_mix_snr)
    if args.bench:
        benchmark(run, args.bench, args.fused)
    return rc


if __name__ == "__main__":
    sys.exit(main())
