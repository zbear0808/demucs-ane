"""
Verify where the model actually runs (CPU / GPU / ANE) and how long it takes.

Uses `macmon` (brew install macmon; no sudo needed) to sample the SoC power rails while
the model runs in a loop, once per Core ML compute-unit setting. If the ANE is doing the
work, CPU_AND_NE shows ANE power and no extra GPU power, and CPU_ONLY shows none.

Power is compared to an idle baseline taken first, so background load (e.g. a GPU render)
is subtracted out. Timing uses time.perf_counter around predict(), which blocks until the
outputs are ready.

  .venv/bin/python ane/measure_ane_usage.py
  .venv/bin/python ane/measure_ane_usage.py --units CPU_AND_NE CPU_ONLY --seconds 15
  .venv/bin/python ane/measure_ane_usage.py --onnx ane/models/htdemucs_drums_ane_fp16.onnx
"""
import argparse
import json
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

from ane_tools import (
    COMPUTE_UNITS, DEFAULT_MLPACKAGE, MODELS_DIR, SAMPLE_RATE, SEGMENT_LEN, SPEC_SHAPE, TIME_SHAPE,
    cpu_postproc_numpy, cpu_preproc_numpy, fold_time, unfold_time,
)
from parity_ane import CoreMLRunner, OnnxCoreMLRunner

FIELDS = ("ane_power", "gpu_power", "cpu_power", "gpu_active_ratio", "pcpu_active_ratio", "ecpu_active_ratio")


class PowerSampler:
    """Collects macmon samples in a background thread."""

    def __init__(self, interval_ms=250):
        self.proc = subprocess.Popen(["macmon", "pipe", "-i", str(interval_ms)], stdout=subprocess.PIPE, text=True)
        self.samples, self.recording = [], False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            if self.recording:
                try:
                    d = json.loads(line)
                    self.samples.append({k: d.get(k, 0.0) for k in FIELDS})
                except json.JSONDecodeError:
                    pass

    def start(self):
        self.samples, self.recording = [], True

    def stop(self) -> dict:
        self.recording = False
        if not self.samples:
            return {k: float("nan") for k in FIELDS}
        return {k: float(np.mean([s[k] for s in self.samples])) for k in FIELDS}

    def close(self):
        self.proc.terminate()


def run_loop(run, seconds):
    rng = np.random.default_rng(0)
    x_spec = rng.standard_normal(SPEC_SHAPE).astype(np.float32)
    x_time = rng.standard_normal(TIME_SHAPE).astype(np.float32)
    times, t_end = [], time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        run(x_spec, x_time)
        times.append(time.perf_counter() - t0)
    return times


def fmt_power(p, base):
    return (f"ANE {p['ane_power']:5.2f} W (+{p['ane_power'] - base['ane_power']:5.2f}) | "
            f"GPU {p['gpu_power']:5.2f} W (+{p['gpu_power'] - base['gpu_power']:5.2f}) | "
            f"CPU {p['cpu_power']:5.2f} W (+{p['cpu_power'] - base['cpu_power']:5.2f}) | "
            f"P-cores {100 * p['pcpu_active_ratio']:3.0f}% E-cores {100 * p['ecpu_active_ratio']:3.0f}%"
            f" GPU busy {100 * p['gpu_active_ratio']:3.0f}%")


def timing_breakdown(run, n=5):
    """Split one chunk into CPU preprocessing, model, CPU postprocessing."""
    rng = np.random.default_rng(1)
    chunk = (rng.standard_normal((1, 2, SEGMENT_LEN)) * 0.1).astype(np.float32)
    pre, mod, post = [], [], []
    for _ in range(n):
        t0 = time.perf_counter()
        x_spec, x_time, stats = cpu_preproc_numpy(chunk)
        x_spec, x_time = np.ascontiguousarray(x_spec, dtype=np.float32), fold_time(x_time)
        t1 = time.perf_counter()
        s, t = run(x_spec, x_time)
        t2 = time.perf_counter()
        cpu_postproc_numpy(s, unfold_time(t), stats)
        t3 = time.perf_counter()
        pre.append(t1 - t0), mod.append(t2 - t1), post.append(t3 - t2)
    med = lambda v: 1000 * float(np.median(v))  # noqa: E731
    total = med(pre) + med(mod) + med(post)
    print(f"  per chunk (median of {n}): preproc {med(pre):.0f} ms | model {med(mod):.0f} ms | "
          f"postproc {med(post):.0f} ms | sequential total {total:.0f} ms "
          f"({SEGMENT_LEN / SAMPLE_RATE / (total / 1000):.1f}x RT)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(DEFAULT_MLPACKAGE))
    p.add_argument("--onnx", default=None)
    p.add_argument("--units", nargs="+", choices=COMPUTE_UNITS, default=["CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"])
    p.add_argument("--seconds", type=float, default=10.0)
    args = p.parse_args()
    if not shutil.which("macmon"):
        sys.exit("macmon not found: brew install macmon")

    sampler = PowerSampler()
    time.sleep(1.0)
    print(f"Idle baseline ({args.seconds:.0f}s)...")
    sampler.start()
    time.sleep(args.seconds)
    base = sampler.stop()
    print(f"  {fmt_power(base, base)}")

    for units in args.units:
        print(f"\n=== {units} ===")
        t0 = time.perf_counter()
        if args.onnx:
            run = OnnxCoreMLRunner(args.onnx, units, cache_dir=MODELS_DIR / "ort_coreml_cache")
        else:
            run = CoreMLRunner(args.model, units)
        print(f"  load {time.perf_counter() - t0:.1f}s")
        run_loop(run, 2.0)  # warmup
        sampler.start()
        times = run_loop(run, args.seconds)
        power = sampler.stop()
        ms = 1000 * np.array(times)
        print(f"  model latency over {len(ms)} runs: median {np.median(ms):.0f} ms | min {ms.min():.0f} | "
              f"p90 {np.percentile(ms, 90):.0f} -> {SEGMENT_LEN / SAMPLE_RATE / (np.median(ms) / 1000):.1f}x RT")
        print(f"  {fmt_power(power, base)}")
        timing_breakdown(run)
        del run
    sampler.close()


if __name__ == "__main__":
    sys.exit(main())
