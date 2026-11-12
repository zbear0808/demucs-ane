"""
Where does a chunk's time go: Neural Engine vs CPU, with the STFT/iSTFT on the CPU or on the ANE.

Three pipelines per 7.8 s chunk, all on real audio (the parity reference chunks):

  cpu-dsp  numpy STFT -> ANE separation model -> numpy iSTFT          (current runner)
  ane-dsp  ANE STFT model -> ANE separation model -> ANE iSTFT model  (3 Core ML calls)
  fused    ANE [STFT + separation + iSTFT] in one Core ML program     (1 call)

Every stage is timed separately. The fixed cost of a Core ML call (fp32<->fp16 input/output
conversion and dispatch, all on the CPU) is measured with "passthrough" models that have
the same I/O and almost no compute. Accuracy (error below the mix vs stock
Demucs) is reported for each pipeline so that speed is not bought with quality.

  .venv/bin/python ane/benchmark_pipeline.py                # drums
  .venv/bin/python ane/benchmark_pipeline.py --source vocals --runs 20
  .venv/bin/python ane/benchmark_pipeline.py --rebuild      # re-export the DSP / fused models
"""
import argparse
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

import dsp_ane as dsp  # noqa: E402
from ane_modules import SPEC_SHAPE, TIME_SHAPE, build_ane_model  # noqa: E402
from ane_tools import (  # noqa: E402
    MODELS_DIR, SAMPLE_RATE, SEGMENT_LEN, compute_plan_summary, convert_fp16, cpu_postproc_numpy,
    cpu_preproc_numpy, fold_time, load_ref, mlpackage_path, ref_path, snr_db, unfold_time,
)

CHUNK_SEC = SEGMENT_LEN / SAMPLE_RATE


class Passthrough(nn.Module):
    """Same inputs/outputs as a real model, trivial compute: measures Core ML call overhead."""

    def __init__(self, n_out):
        super().__init__()
        self.n_out = n_out

    def forward(self, *xs):
        return tuple(x[..., :1] * 0 + x[..., -1:] * 0 + x * 1.0001 for x in xs[: self.n_out])


def _load(path, units="CPU_AND_NE"):
    import coremltools as ct

    return ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, units))


def build_models(source: str, rebuild: bool):
    paths = {
        "stft": MODELS_DIR / "htdemucs_stft_ane.mlpackage",
        "istft": MODELS_DIR / "htdemucs_istft_ane.mlpackage",
        "fused": MODELS_DIR / f"htdemucs_{source}_fused_ane.mlpackage",
        "pt_model": MODELS_DIR / "passthrough_model_io.mlpackage",
        "pt_stft": MODELS_DIR / "passthrough_stft_io.mlpackage",
    }
    blocks = torch.zeros(1, 2, dsp.N_BLOCKS, dsp.HOP)
    spec, xt = torch.zeros(SPEC_SHAPE), torch.zeros(TIME_SHAPE)
    todo = {
        "stft": lambda: (dsp.STFTANE(), (blocks,), ["blocks"], ["x_spec", "mean_s", "std_s"]),
        # distinct tensors: torch.export merges an input passed twice, which would tie mean_s to std_s
        "istft": lambda: (dsp.ISTFTANE(), (spec, torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, 1)),
                          ["out_spec", "mean_s", "std_s"], ["wav_blocks"]),
        "fused": lambda: (dsp.FusedPipelineANE(build_ane_model(source=source)[0]), (blocks, xt),
                          ["blocks", "x_time"], ["wav_blocks", "out_time"]),
        "pt_model": lambda: (Passthrough(2), (spec, xt), ["x_spec", "x_time"], ["out_spec", "out_time"]),
        "pt_stft": lambda: (Passthrough(1), (blocks,), ["blocks"], ["y"]),
    }
    for name, path in paths.items():
        if rebuild or not path.exists():
            print(f"[build] {path.name}")
            mod, ex, i, o = todo[name]()
            convert_fp16(mod, ex, i, o, path)
            if name in ("stft", "istft", "fused"):
                compute_plan_summary(str(path))
    return {k: _load(v) for k, v in paths.items()}


def timed(fn, store, key):
    t0 = time.perf_counter()
    out = fn()
    store.setdefault(key, []).append(time.perf_counter() - t0)
    return out


def run_cpu_dsp(m, mix, t):
    x_spec, x_time, stats = timed(lambda: cpu_preproc_numpy(mix), t, "CPU  numpy STFT + normalise")
    o = timed(lambda: m["model"].predict({"x_spec": np.ascontiguousarray(x_spec, np.float32), "x_time": fold_time(x_time)}),
              t, "ANE  separation model (call)")
    return timed(lambda: cpu_postproc_numpy(o["out_spec"], unfold_time(o["out_time"]), stats)[0], t,
                 "CPU  numpy iSTFT + combine")


def _cpu_time_branch(mix):
    x_time, mean_t, std_t = dsp.time_stats_numpy(mix)
    return dsp.reflect_pad_numpy(mix), fold_time(x_time), mean_t, std_t


def _combine(wav_blocks, out_time, mean_t, std_t):
    wav_time = unfold_time(out_time)[:, :, 0, :] * std_t + mean_t
    return (dsp.crop_istft(wav_blocks) + wav_time)[0]


def run_ane_dsp(m, mix, t):
    blocks, x_time, mean_t, std_t = timed(lambda: _cpu_time_branch(mix), t, "CPU  reflect pad + time-branch norm")
    s = timed(lambda: m["stft"].predict({"blocks": blocks}), t, "ANE  STFT model (call)")
    o = timed(lambda: m["model"].predict({"x_spec": s["x_spec"], "x_time": x_time}), t, "ANE  separation model (call)")
    w = timed(lambda: m["istft"].predict({"out_spec": o["out_spec"], "mean_s": s["mean_s"], "std_s": s["std_s"]}),
              t, "ANE  iSTFT model (call)")
    return timed(lambda: _combine(w["wav_blocks"], o["out_time"], mean_t, std_t), t, "CPU  crop + combine")


def run_fused(m, mix, t):
    blocks, x_time, mean_t, std_t = timed(lambda: _cpu_time_branch(mix), t, "CPU  reflect pad + time-branch norm")
    o = timed(lambda: m["fused"].predict({"blocks": blocks, "x_time": x_time}), t, "ANE  fused STFT+model+iSTFT (call)")
    return timed(lambda: _combine(o["wav_blocks"], o["out_time"], mean_t, std_t), t, "CPU  crop + combine")


def overheads(m, runs):
    """Median Core ML call cost with ~zero compute, per I/O signature."""
    rng = np.random.default_rng(0)
    spec, xt = rng.standard_normal(SPEC_SHAPE).astype(np.float32), rng.standard_normal(TIME_SHAPE).astype(np.float32)
    blocks = rng.standard_normal((1, 2, dsp.N_BLOCKS, dsp.HOP)).astype(np.float32)
    res = {}
    for key, fn in (("model", lambda: m["pt_model"].predict({"x_spec": spec, "x_time": xt})),
                    ("stft", lambda: m["pt_stft"].predict({"blocks": blocks}))):
        fn()
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        res[key] = float(np.median(ts))
    return res


def report(name, t, n_chunks):
    med = {k: 1000 * float(np.median(v)) for k, v in t.items()}
    total = sum(med.values())
    ane = sum(v for k, v in med.items() if k.startswith("ANE"))
    print(f"\n=== {name} (median per chunk over {n_chunks} chunks) ===")
    for k, v in med.items():
        print(f"  {k:40s} {v:7.1f} ms")
    print(f"  {'-' * 40} {'-' * 7}")
    print(f"  {'total':40s} {total:7.1f} ms  -> {CHUNK_SEC / (total / 1000):.1f}x real time (sequential)")
    print(f"  split: Core ML calls {ane:.0f} ms ({100 * ane / total:.0f}%) | CPU numpy {total - ane:.0f} ms "
          f"({100 * (total - ane) / total:.0f}%)")
    return total, med


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="drums")
    p.add_argument("--runs", type=int, default=10, help="Passes over the reference chunks")
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()

    ref = load_ref(ref_path(args.source))
    if ref is None:
        sys.exit(f"missing {ref_path(args.source)}: run export_htdemucs_ane.py --source {args.source} first")
    m = build_models(args.source, args.rebuild)
    m["model"] = _load(mlpackage_path(args.source))
    ovh = overheads(m, 20)
    print("\nCore ML call overhead with ~zero compute (passthrough models with the same I/O):")
    print(f"  separation-model I/O (x_spec + x_time in and out): {1000 * ovh['model']:.1f} ms")
    print(f"  STFT-block I/O ([1, 2, 343, 1024] in and out):      {1000 * ovh['stft']:.1f} ms")
    print("  This is CPU-side work inside every predict(): fp32 <-> fp16 conversion of the")
    print("  arrays and copying them into and out of the ANE's buffers.")

    pipelines = {"cpu-dsp": run_cpu_dsp, "ane-dsp": run_ane_dsp, "fused": run_fused}
    totals = {}
    for name, fn in pipelines.items():
        mixes = [ref["chunk"][i : i + 1] for i in range(len(ref["songs"]))]
        fn(m, mixes[0], {})  # warmup
        t, worst = {}, float("inf")
        for r in range(args.runs):
            for i, mix in enumerate(mixes):
                wav = fn(m, mix, t)
                if r == 0:
                    err = wav - ref["stock_wav"][i]
                    worst = min(worst, snr_db(ref["chunk"][i], ref["chunk"][i] + err))
        totals[name], _ = report(name, t, args.runs * len(mixes))
        print(f"  accuracy: worst error below mix {worst:.1f} dB (PASS >= 40)")

    base = totals["cpu-dsp"]
    print("\n=== Summary ===")
    for name, tot in totals.items():
        print(f"  {name:8s} {tot:6.1f} ms/chunk  ({base / tot:4.2f}x vs cpu-dsp)")


if __name__ == "__main__":
    sys.exit(main())
