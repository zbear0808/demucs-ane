"""
Fused model benchmark: the default split pipeline vs the fused model, Core ML and ONNX.

  split         Core ML separation model on the ANE + numpy STFT/iSTFT on the CPU (default runner)
  fused-coreml  one Core ML program: STFT -> separation -> iSTFT (htdemucs_<src>_fused_ane.mlpackage)
  fused-onnx    the same graph as fp16 ONNX through ONNX Runtime's CoreML EP (..._fused_ane_fp16.onnx)

Reports, per variant: size on disk, load time (cold and warm, fresh process each), process
memory footprint, per-chunk runtime on real audio (variants interleaved so drift hits all of them
equally), CPU seconds this process spends per chunk, and accuracy against stock Demucs.

  .venv/bin/python ane/profile_fused.py
  .venv/bin/python ane/profile_fused.py --runs 15 --skip-load
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import benchmark_pipeline as bp  # noqa: E402
from ane_tools import (  # noqa: E402
    FUSED_IO, MODELS_DIR, SAMPLE_RATE, SEGMENT_LEN, compiled_path, fused_mlpackage_path, fused_onnx_path,
    load_ref, mlpackage_path, ref_path, snr_db,
)
from parity_ane import OnnxCoreMLRunner  # noqa: E402
from profile_variants import PY, dir_mb, load_and_memory  # noqa: E402

ANE = Path(__file__).resolve().parent
ORT_CACHE = MODELS_DIR / "ort_coreml_cache"

CHILD_ORT = r"""
import json, os, re, resource, subprocess, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, onnxruntime as ort
def footprint():
    out = subprocess.run(["footprint", str(os.getpid())], capture_output=True, text=True).stdout
    m = re.search(r"Footprint:\s*([\d.]+)\s*(\w+)", out)
    return float(m.group(1)) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}.get(m.group(2), 1)
path, cache = sys.argv[1], sys.argv[2]
base = footprint()
opts = {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndNeuralEngine", "RequireStaticInputShapes": "1", "ModelCacheDirectory": cache}
t0 = time.perf_counter()
sess = ort.InferenceSession(path, providers=[("CoreMLExecutionProvider", opts), "CPUExecutionProvider"])
load = time.perf_counter() - t0
after_load = footprint()
rng = np.random.default_rng(0)
feed = {i.name: rng.standard_normal(i.shape).astype(np.float16) for i in sess.get_inputs()}
peak = after_load
for _ in range(3):
    sess.run(None, feed)
    peak = max(peak, footprint())
print(json.dumps({"load_s": load, "mem_base": base, "mem_loaded": after_load, "mem_infer": peak,
                  "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}))
"""


def child_ort(path, cache):
    out = subprocess.run([PY, "-c", CHILD_ORT, str(path), str(cache)], capture_output=True, text=True, cwd=ANE)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    if not line:
        raise RuntimeError(out.stderr[-2000:])
    return json.loads(line[-1])


class OrtPredict:
    """Gives the ONNX runner the .predict({...}) -> {...} interface that benchmark_pipeline.run_fused uses."""

    def __init__(self, runner):
        self.runner = runner

    def predict(self, feed):
        wav_blocks, out_time = self.runner(feed[FUSED_IO["inputs"][0]], feed[FUSED_IO["inputs"][1]])
        return {FUSED_IO["outputs"][0]: wav_blocks, FUSED_IO["outputs"][1]: out_time}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="drums")
    p.add_argument("--runs", type=int, default=10, help="Passes over the reference chunks")
    p.add_argument("--skip-load", action="store_true", help="Skip the slow cold-load measurements")
    p.add_argument("--only-load", action="store_true", help="Only size / load / memory; skip the runtime comparison")
    args = p.parse_args()

    split_pkg, fused_pkg = mlpackage_path(args.source), fused_mlpackage_path(args.source)
    onnx16 = fused_onnx_path(args.source, fp16=True)
    for need in (split_pkg, fused_pkg, onnx16):
        if not need.exists():
            sys.exit(f"missing {need}: run `export_htdemucs_ane.py --fused --onnx` first")
    report = {}

    print("=== Size on disk ===")
    sizes = {
        "split": (dir_mb(split_pkg), dir_mb(compiled_path(split_pkg))),
        "fused-coreml": (dir_mb(fused_pkg), dir_mb(compiled_path(fused_pkg))),
        "fused-onnx": (onnx16.stat().st_size / 2**20, None),
    }
    for v, (a, b) in sizes.items():
        print(f"  {v:13s} {a:6.1f} MB" + (f" (+ compiled .mlmodelc {b:.1f} MB)" if b else " (.onnx, fp16)"))
        report[v] = {"size_mb": a}

    if not args.skip_load:
        print("\n=== Load time and memory (fresh process each) ===")
        for v, pkgs in (("split", [split_pkg]), ("fused-coreml", [fused_pkg])):
            cold, warm = load_and_memory(pkgs)
            report[v].update(cold=cold, warm=warm)
            print(f"  {v:13s} cold {cold['load_s']:6.1f} s | warm {warm['load_s']:5.2f} s | peak RSS {warm['peak_rss']:5.0f} MB "
                  f"(footprint {warm['mem_infer']:.0f} MB, of which Python+libs {warm['mem_base']:.0f} MB)")
        tmp = Path(tempfile.mkdtemp(prefix="ort_cold_"))
        cold = child_ort(onnx16, tmp)
        ort_cache_mb = dir_mb(tmp)
        shutil.rmtree(tmp, ignore_errors=True)
        child_ort(onnx16, ORT_CACHE)  # make sure the persistent cache exists
        warm = child_ort(onnx16, ORT_CACHE)  # ORT reuses its ONNX -> Core ML conversion; the ANE build is still paid
        report["fused-onnx"].update(cold=cold, warm=warm, ort_cache_mb=ort_cache_mb)
        print(f"  {'fused-onnx':13s} cold {cold['load_s']:6.1f} s | cached conversion {warm['load_s']:5.1f} s | peak RSS "
              f"{warm['peak_rss']:5.0f} MB (footprint {warm['mem_infer']:.0f} MB; unreliable while the Mac swaps)  "
              f"+ ORT's converted-model cache on disk: {ort_cache_mb:.0f} MB")

    if args.only_load:
        out = ANE / f"profile_fused_{args.source}.json"
        out.write_text(json.dumps(report, indent=2, default=float))
        return 0

    print("\n=== Runtime per 7.8 s chunk, real audio, variants interleaved ===")
    ref = load_ref(ref_path(args.source))
    mixes = [ref["chunk"][i : i + 1] for i in range(len(ref["songs"]))]
    models = {"model": bp._load(split_pkg), "fused": bp._load(fused_pkg)}
    ort_models = {"fused": OrtPredict(OnnxCoreMLRunner(onnx16, "CPU_AND_NE", cache_dir=ORT_CACHE))}
    fns = {
        "split": lambda mix, t: bp.run_cpu_dsp(models, mix, t),
        "fused-coreml": lambda mix, t: bp.run_fused(models, mix, t),
        "fused-onnx": lambda mix, t: bp.run_fused(ort_models, mix, t),
    }
    times, cpu, acc = {v: {} for v in fns}, {v: [] for v in fns}, {v: [] for v in fns}
    for fn in fns.values():
        fn(mixes[0], {})
    for r in range(args.runs):
        for i, mix in enumerate(mixes):
            for v, fn in fns.items():
                c0 = time.process_time()
                wav = fn(mix, times[v])
                cpu[v].append(time.process_time() - c0)
                if r == 0:
                    acc[v].append(snr_db(ref["chunk"][i], ref["chunk"][i] + (wav - ref["stock_wav"][i])))

    chunk_sec = SEGMENT_LEN / SAMPLE_RATE
    print(f"  {'':13s} {'total':>9s} {'p90':>8s} {'model call':>11s} {'CPU numpy':>10s} {'process CPU':>12s} {'speed':>8s}  "
          f"accuracy (error below mix, worst)")
    for v in fns:
        med = {k: 1000 * float(np.median(x)) for k, x in times[v].items()}
        total = sum(med.values())
        call = sum(x for k, x in med.items() if k.startswith("ANE"))
        c = 1000 * float(np.median(cpu[v]))
        per_run = 1000 * np.sum([np.array(x) for x in times[v].values()], axis=0)  # total per chunk-run
        p90 = float(np.percentile(per_run, 90))
        report[v].update(total_ms=total, p90_ms=p90, call_ms=call, numpy_ms=total - call, process_cpu_ms=c,
                         accuracy_db=min(acc[v]), stages=med)
        print(f"  {v:13s} {total:7.1f}ms {p90:6.0f}ms {call:9.1f}ms {total - call:8.1f}ms {c:10.1f}ms "
              f"{chunk_sec / (total / 1000):6.1f}x   {min(acc[v]):.1f} dB ({'PASS' if min(acc[v]) >= 40 else 'FAIL'})")
        if p90 > 1.15 * total:
            print(f"    ^ p90 is {p90 / total:.2f}x the median: something else probably used the Neural Engine/CPU "
                  f"during this run. Rerun when the machine is quiet.")
    print("  (process CPU = CPU seconds this Python process spent per chunk, all threads; other apps do not affect it)")

    out = ANE / f"profile_fused_{args.source}.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[OK] raw numbers -> {out}")


if __name__ == "__main__":
    sys.exit(main())
