"""
Real-audio parity + latency check for the ANE single-source models (drums, vocals, ...).

Uses the same reference data as the XDNA2 flow (option_a_bfp16/parity_real_audio.npz,
built by `option_a_bfp16/parity_real_audio.py make-ref`) and the same metrics:
  spec / time - SNR of the raw model outputs vs the PyTorch FP32 pure-AIE core
  wav         - separated stem chunk vs stock Demucs forward()
  mix         - error level below the input mix (the PASS/FAIL criterion)

  .venv/bin/python ane/parity_ane.py                                  # Core ML on the ANE
  .venv/bin/python ane/parity_ane.py --units CPU_ONLY                 # same model, CPU
  .venv/bin/python ane/parity_ane.py --source vocals                  # vocals model + vocals reference
  .venv/bin/python ane/parity_ane.py --onnx ane/models/htdemucs_drums_ane_fp16.onnx   # ONNX via ORT CoreML EP
"""
import argparse
import sys
import time

import numpy as np

from ane_tools import (
    COMPUTE_UNITS, compute_unit, mlpackage_path, ref_path, cpu_postproc_numpy, fold_time, load_ref,
    snr_db, unfold_time,
)


class CoreMLRunner:
    def __init__(self, path, units):
        import coremltools as ct

        self.model = ct.models.MLModel(str(path), compute_units=compute_unit(units))

    def __call__(self, x_spec, x_time):
        out = self.model.predict({"x_spec": x_spec, "x_time": x_time})
        return out["out_spec"], out["out_time"]


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

    def __call__(self, x_spec, x_time):
        s, t = self.sess.run(None, {"x_spec": x_spec.astype(self.dtype), "x_time": x_time.astype(self.dtype)})
        return s.astype(np.float32), t.astype(np.float32)


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


def benchmark(run, n: int):
    rng = np.random.default_rng(0)
    from ane_tools import SPEC_SHAPE, TIME_SHAPE, SAMPLE_RATE, SEGMENT_LEN

    x_spec = rng.standard_normal(SPEC_SHAPE).astype(np.float32)
    x_time = rng.standard_normal(TIME_SHAPE).astype(np.float32)
    run(x_spec, x_time)  # warmup
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        run(x_spec, x_time)
        times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    print(f"\nModel latency over {n} runs: median {med * 1000:.0f} ms, min {min(times) * 1000:.0f} ms"
          f" -> {SEGMENT_LEN / SAMPLE_RATE / med:.1f}x real-time (model only, excludes STFT/iSTFT)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="drums", help="Picks the default --model and --ref")
    p.add_argument("--model", default=None, help="Core ML .mlpackage (default: htdemucs_<source>_ane.mlpackage)")
    p.add_argument("--onnx", default=None, help="Evaluate an ONNX model through ORT's CoreML EP instead")
    p.add_argument("--units", choices=COMPUTE_UNITS, default="CPU_AND_NE")
    p.add_argument("--ref", default=None)
    p.add_argument("--min-mix-snr", type=float, default=40.0)
    p.add_argument("--bench", type=int, default=10, help="Latency runs after the parity check (0 = skip)")
    args = p.parse_args()
    args.model = args.model or str(mlpackage_path(args.source))
    args.ref = args.ref or str(ref_path(args.source))

    t0 = time.perf_counter()
    if args.onnx:
        from ane_tools import MODELS_DIR

        run = OnnxCoreMLRunner(args.onnx, args.units, cache_dir=MODELS_DIR / "ort_coreml_cache")
    else:
        run = CoreMLRunner(args.model, args.units)
    print(f"Model: {args.onnx or args.model} [{args.units}] loaded in {time.perf_counter() - t0:.1f}s\n")

    rc = 0
    ref = load_ref(args.ref)
    if ref is None:
        print(f"[skip] no reference data at {args.ref} (export_htdemucs_ane.py builds it)")
    else:
        rc = evaluate(run, ref, args.min_mix_snr)
    if args.bench:
        benchmark(run, args.bench)
    return rc


if __name__ == "__main__":
    sys.exit(main())
