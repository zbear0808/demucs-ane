"""
Full comparison of the two ANE deployment styles (drums by default):

  split   separation model on the ANE, STFT/iSTFT in numpy on the CPU (the default runner)
  split3  separation model + separate STFT and iSTFT models, all on the ANE (`--dsp ane`)
  fused   one Core ML program: STFT -> separation -> iSTFT, all on the ANE

For each one, it reports:
  * size on disk (.mlpackage and the compiled .mlmodelc)
  * load time: cold (first launch; Core ML builds the ANE program) and warm (cached)
  * process memory footprint after load and after inference
  * per-chunk runtime broken into stages, and the ANE / CPU split
and, for the shared separation model, where the ANE time goes: per network stage
(measured with stage sub-models) and per op type (Core ML's cost estimate).

Load and memory are measured in fresh subprocesses so the numbers match a real launch.

  .venv/bin/python ane/profile_variants.py
  .venv/bin/python ane/profile_variants.py --source vocals --skip-stages
"""
import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ANE = Path(__file__).resolve().parent
PY = sys.executable

from ane_tools import MODELS_DIR, compiled_path, mlpackage_path  # noqa: E402

DSP_MODELS = ("htdemucs_stft_ane", "htdemucs_istft_ane")


def variant_packages(source):
    sep = mlpackage_path(source)
    dsp = [MODELS_DIR / f"{n}.mlpackage" for n in DSP_MODELS]
    return {
        "split": [sep],
        "split3": [sep, *dsp],
        "fused": [MODELS_DIR / f"htdemucs_{source}_fused_ane.mlpackage"],
    }


def dir_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**20


# ---------------------------------------------------------------------------
# Load time + memory, in a child process
# ---------------------------------------------------------------------------
CHILD = r"""
import json, os, re, resource, subprocess, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, coremltools as ct
def footprint():
    out = subprocess.run(["footprint", str(os.getpid())], capture_output=True, text=True).stdout
    m = re.search(r"Footprint:\s*([\d.]+)\s*(\w+)", out)
    v, unit = float(m.group(1)), m.group(2)
    return v * {"KB": 1 / 1024, "MB": 1, "GB": 1024}.get(unit, 1)
paths = json.loads(sys.argv[1])
base = footprint()
t0 = time.perf_counter()
models = [ct.models.CompiledMLModel(p, compute_units=ct.ComputeUnit.CPU_AND_NE) for p in paths]
load = time.perf_counter() - t0
after_load = footprint()
rng = np.random.default_rng(0)
peak = after_load
feeds = json.loads(sys.argv[2])
for m, feed in zip(models, feeds):
    x = {k: rng.standard_normal(v).astype(np.float32) for k, v in feed.items()}
    for _ in range(2):
        m.predict(x)
    peak = max(peak, footprint())
print(json.dumps({"load_s": load, "mem_base": base, "mem_loaded": after_load, "mem_infer": peak,
                  "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}))
"""

FEEDS = {
    "htdemucs_stft_ane": {"blocks": [1, 2, 343, 1024]},
    "htdemucs_istft_ane": {"out_spec": [1, 4, 2048, 336], "mean_s": [1, 1, 1, 1], "std_s": [1, 1, 1, 1]},
    "sep": {"x_spec": [1, 4, 2048, 336], "x_time": [1, 2, 32, 10752]},
    "fused": {"blocks": [1, 2, 343, 1024], "x_time": [1, 2, 32, 10752]},
}


def feed_for(pkg: Path):
    if pkg.stem in FEEDS:
        return FEEDS[pkg.stem]
    return FEEDS["fused"] if "fused" in pkg.stem else FEEDS["sep"]


def child(paths, feeds):
    out = subprocess.run([PY, "-c", CHILD, json.dumps([str(p) for p in paths]), json.dumps(feeds)],
                         capture_output=True, text=True, cwd=ANE)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    if not line:
        raise RuntimeError(out.stderr[-2000:])
    return json.loads(line[-1])


def load_and_memory(pkgs):
    compiled = [compiled_path(p) for p in pkgs]
    feeds = [feed_for(p) for p in pkgs]
    # Cold: copy the compiled models to fresh paths. The OS-side ANE cache is keyed by path,
    # so this reproduces a first launch (or the old MLModel(mlpackage) behaviour).
    tmp = Path(tempfile.mkdtemp(prefix="ane_cold_"))
    fresh = []
    for c in compiled:
        dst = tmp / f"{c.stem}_{time.time_ns()}.mlmodelc"
        shutil.copytree(c, dst)
        fresh.append(dst)
    cold = child(fresh, feeds)
    # Warm: prime the persistent paths once, then measure. The OS ANE cache is LRU with a
    # limited size, so an entry that has not been used lately may need rebuilding first.
    child(compiled, feeds)
    warm = child(compiled, feeds)
    shutil.rmtree(tmp, ignore_errors=True)
    return cold, warm


# ---------------------------------------------------------------------------
# Inside the separation model
# ---------------------------------------------------------------------------
def stage_profile(source):
    """
    Where the ANE time goes inside the separation model, per network stage.

    Skip tensors are up to 8M values. As model inputs or outputs they would cost more in
    array copies than the stage itself, so every program here has small I/O. The
    encoders and the bottleneck are timed directly. Each decoder is timed by subtraction:
    the whole model minus the whole model with that decoder's output dropped (the decoder
    is then dead code and removed).
    """
    import torch
    import torch.nn as nn

    from ane_modules import FR_BOTTOM, N_FRAMES, SPEC_SHAPE, T_BOTTOM, TIME_SHAPE, build_ane_model
    from ane_tools import convert_fp16, load_coreml

    model, _ = build_ane_model(source=source)

    class Stage(nn.Module):
        # Holds the model as a registered submodule so torch.export sees its weights.
        def __init__(self):
            super().__init__()
            self.model = model

    class FreqEnc(Stage):
        def forward(self, x):
            c = self.model.core
            for i in range(4):
                x = c.encoder[i](x)
                if i == 0:
                    x = x + c.baked_freq_emb
            return x

    class TimeEnc(Stage):
        def forward(self, xt):
            md = self.model
            for i in range(4):
                xt = md._fold(md.core.tencoder[i](md._fold(xt, md.t_lens[i])), md.t_lens[i + 1])
            return xt

    class Bottleneck(Stage):
        def forward(self, x, xt):
            c, n = self.model.core, FR_BOTTOM * N_FRAMES
            x = c.channel_upsampler(x.reshape(1, -1, 1, n)).reshape(1, -1, FR_BOTTOM, N_FRAMES)
            xt = c.channel_upsampler_t(xt)
            x, xt3 = c.crosstransformer(x, xt.reshape(1, -1, T_BOTTOM))
            xt = xt3.reshape(1, -1, 1, T_BOTTOM)
            x = c.channel_downsampler(x.reshape(1, -1, 1, n)).reshape(1, -1, FR_BOTTOM, N_FRAMES)
            return x, c.channel_downsampler_t(xt)

    class Full(Stage):
        def forward(self, x, xt):
            return self.model(x, xt)[:2]

    class NoTimeDec(Stage):
        def forward(self, x, xt):
            return self.model(x, xt)[0]

    class NoFreqDec(Stage):
        def forward(self, x, xt):
            return self.model(x, xt)[1]

    with torch.no_grad():
        xs, xt = torch.randn(SPEC_SHAPE), torch.randn(TIME_SHAPE)
        bx, bxt = FreqEnc()(xs), TimeEnc()(xt)
    progs = {
        "freq_enc": (FreqEnc(), (xs,)),
        "time_enc": (TimeEnc(), (xt,)),
        "bottleneck": (Bottleneck(), (bx, bxt)),
        "full": (Full(), (xs, xt)),
        "no_tdec": (NoTimeDec(), (xs, xt)),
        "no_fdec": (NoFreqDec(), (xs, xt)),
    }

    raw = {}
    tmp = Path(tempfile.mkdtemp(prefix="ane_stage_"))
    for name, (mod, ex) in progs.items():
        with torch.no_grad():
            out = mod(*ex)
        n_out = len(out) if isinstance(out, tuple) else 1
        ins = [f"i{k}" for k in range(len(ex))]
        path = tmp / f"stage_{name}.mlpackage"
        convert_fp16(mod, ex, ins, [f"o{k}" for k in range(n_out)], path)
        m = load_coreml(path)
        feed = {k: v.numpy() for k, v in zip(ins, ex)}
        for _ in range(2):
            m.predict(feed)
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            m.predict(feed)
            ts.append(time.perf_counter() - t0)
        raw[name] = 1000 * float(np.median(ts))
        print(f"  [program] {name:10s} {raw[name]:6.1f} ms", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    for f in (MODELS_DIR / "compiled").glob("stage_*"):
        shutil.rmtree(f, ignore_errors=True)
    res = {
        "freq encoder (4 layers)": raw["freq_enc"],
        "time encoder (4 layers)": raw["time_enc"],
        "bottleneck: cross-transformer": raw["bottleneck"],
        "freq decoder (4 layers)": raw["full"] - raw["no_fdec"],
        "time decoder (4 layers)": raw["full"] - raw["no_tdec"],
    }
    return res, raw["full"]


def op_cost(pkg):
    """Core ML's estimated relative cost per op type (MLComputePlan)."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    m = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)
    plan = MLComputePlan.load_from_path(m.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    cost, count = collections.Counter(), collections.Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        est = plan.get_estimated_cost_for_mlprogram_operation(op)
        if est is None:
            continue
        name = op.operator_name.split(".")[-1]
        cost[name] += est.weight
        count[name] += 1
    total = sum(cost.values())
    return [(k, 100 * v / total, count[k]) for k, v in cost.most_common()]


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="drums")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--skip-stages", action="store_true")
    args = p.parse_args()

    # Builds the STFT/iSTFT and fused packages if they are missing.
    import benchmark_pipeline as bp

    m = bp.build_models(args.source, rebuild=False)
    variants = variant_packages(args.source)
    report = {}

    print("\n=== Size on disk ===")
    for v, pkgs in variants.items():
        pk = sum(dir_mb(p) for p in pkgs)
        cm = sum(dir_mb(compiled_path(p)) for p in pkgs)
        report.setdefault(v, {}).update(mlpackage_mb=pk, mlmodelc_mb=cm)
        print(f"  {v:7s} .mlpackage {pk:6.1f} MB | compiled .mlmodelc {cm:6.1f} MB | {len(pkgs)} model(s): "
              + ", ".join(p.stem for p in pkgs))

    print("\n=== Load time and memory (fresh process each) ===")
    for v, pkgs in variants.items():
        cold, warm = load_and_memory(pkgs)
        report[v].update(cold=cold, warm=warm)
        print(f"  {v:7s} cold load {cold['load_s']:6.1f} s | warm load {warm['load_s']:5.2f} s | footprint: "
              f"python+libs {warm['mem_base']:.0f} MB, +{warm['mem_loaded'] - warm['mem_base']:.0f} MB loaded, "
              f"{warm['mem_infer']:.0f} MB peak while running")

    print("\n=== Runtime per 7.8 s chunk (real audio) ===")
    m["model"] = bp._load(mlpackage_path(args.source))
    ref = bp.load_ref(bp.ref_path(args.source))
    mixes = [ref["chunk"][i : i + 1] for i in range(len(ref["songs"]))]
    fns = {"split": bp.run_cpu_dsp, "split3": bp.run_ane_dsp, "fused": bp.run_fused}
    times = {v: {} for v in fns}
    for fn in fns.values():
        fn(m, mixes[0], {})
    # Round-robin over the variants so thermal / background drift hits all of them equally.
    for _ in range(args.runs):
        for mix in mixes:
            for v, fn in fns.items():
                fn(m, mix, times[v])
    for v in fns:
        total, med = bp.report(v, times[v], args.runs * len(mixes))
        report[v].update(stages=med, total_ms=total)
    ovh = bp.overheads(m, 20)
    print(f"\n  Core ML call overhead (CPU, array I/O): separation-model I/O {1000 * ovh['model']:.1f} ms, "
          f"STFT-block I/O {1000 * ovh['stft']:.1f} ms")

    print("\n=== Inside the separation model ===")
    print("  Estimated cost by op type (Core ML compute plan):")
    for name, pct, n in op_cost(mlpackage_path(args.source))[:10]:
        print(f"    {name:16s} {pct:5.1f}%  ({n} ops)")
    if not args.skip_stages:
        print("  Measured per stage (small-I/O sub-programs and subtraction; median of 10):")
        st, full = stage_profile(args.source)
        tot = sum(st.values())
        for k, v in st.items():
            print(f"    {k:32s} {v:6.1f} ms  ({100 * v / tot:4.1f}%)")
        print(f"    {'sum of stages':32s} {tot:6.1f} ms  (whole model as one program: {full:.1f} ms)")
        report["stages"] = st

    out = ANE / f"profile_{args.source}.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[OK] raw numbers -> {out}")


if __name__ == "__main__":
    sys.exit(main())
