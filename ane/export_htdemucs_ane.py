"""
Build an HTDemucs single-source model (drums by default, or vocals/bass/other) for the Apple Neural Engine.

Pipeline (mirrors export_candidate_a_pure_aie.py for XDNA2):
  1. Load htdemucs (f7e0c4bc), prune to one source, apply the pure-AIE rewrites, then the
     ANE rewrites from ane_modules.py (static shapes, row-folded time branch, ...).
  2. PyTorch FP32 parity check against the pure-AIE core (= stock Demucs for that source).
  3. torch.export -> coremltools -> FP16 ML Program (.mlpackage).
  4. Core ML compute-plan audit: how many ops / how much estimated cost lands on the ANE.
  5. Optional ONNX export of the same graph (for ONNX Runtime's CoreML execution provider).
  6. Real-audio parity on the ANE (option_a_bfp16/parity_real_audio.npz).

  .venv/bin/python ane/export_htdemucs_ane.py
  .venv/bin/python ane/export_htdemucs_ane.py --onnx          # also write the ONNX model
  .venv/bin/python ane/export_htdemucs_ane.py --source vocals --onnx
  .venv/bin/python ane/export_htdemucs_ane.py --debug-taps    # per-layer SNR on the ANE
"""
import argparse
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

from ane_modules import SOURCES, SPEC_SHAPE, TAP_NAMES, TIME_SHAPE, build_ane_model  # noqa: E402
from ane_tools import (  # noqa: E402
    MODELS_DIR, compute_plan_summary, fold_time, load_ref, make_ref, mlpackage_path, onnx_path, ref_path, snr_db,
)


def check_torch_parity(model, base):
    """FP32 PyTorch: ANE graph vs the pure-AIE core (which matches stock Demucs to 1e-4)."""
    from option_a_bfp16.export_candidate_a_pure_aie import HTDemucsCandidateAPureAIECore

    ref_core = HTDemucsCandidateAPureAIECore(base).eval()
    torch.manual_seed(42)
    x_spec = torch.randn(SPEC_SHAPE)
    x_time = torch.randn(1, 2, 1, SPEC_SHAPE[-1] * 1024)
    with torch.no_grad():
        ref_spec, ref_time = ref_core(x_spec, x_time)
        out_spec, out_time = model(x_spec, x_time.reshape(TIME_SHAPE))[:2]
    s, t = snr_db(ref_spec.numpy(), out_spec.numpy()), snr_db(ref_time.numpy(), out_time.numpy())
    print(f"[PARITY fp32] ANE graph vs pure-AIE core: spec {s:.1f} dB | time {t:.1f} dB")
    # The only intended numerical change is tanh-GELU (~60 dB); anything lower is a bug.
    assert s > 50 and t > 50, "ANE rewrite changed the model maths"


def export_coreml(model, out_path: Path, example, source: str):
    import coremltools as ct

    t0 = time.perf_counter()
    with torch.no_grad():
        ep = torch.export.export(model, example).run_decompositions({})
    print(f"[OK] torch.export in {time.perf_counter() - t0:.1f}s")

    out_names = ["out_spec", "out_time"] + (TAP_NAMES if model.debug_taps else [])
    t0 = time.perf_counter()
    ml = ct.convert(
        ep,
        inputs=[ct.TensorType("x_spec", shape=SPEC_SHAPE), ct.TensorType("x_time", shape=TIME_SHAPE)],
        outputs=[ct.TensorType(n) for n in out_names],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,  # the ANE is fp16-only
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"[OK] coremltools convert in {time.perf_counter() - t0:.1f}s")
    ml.short_description = f"HTDemucs (f7e0c4bc) {source} stem, ANE-optimised core. STFT/iSTFT run on CPU."
    ml.input_description["x_spec"] = "Normalised complex-as-channels STFT [1, 4, 2048, 336]"
    ml.input_description["x_time"] = "Normalised waveform [1, 2, 344064] reshaped to [1, 2, 32, 10752]"
    ml.output_description["out_spec"] = f"{source.capitalize()} CaC spectrogram [1, 4, 2048, 336]"
    ml.output_description["out_time"] = f"{source.capitalize()} time-branch waveform, row-folded [1, 2, 32, 10752]"

    if out_path.exists():
        shutil.rmtree(out_path)
    ml.save(str(out_path))
    size = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 2**20
    print(f"[OK] Saved {out_path} ({size:.1f} MB)")
    return ml


def export_onnx(model, out_path: Path, example):
    import onnx
    from onnxsim import simplify

    raw = out_path.with_name(out_path.stem + "_raw.onnx")
    # dynamo=True goes through torch.export like the Core ML path; the legacy TorchScript
    # exporter cannot handle the static layer_norm shapes.
    torch.onnx.export(
        model, example, str(raw), input_names=["x_spec", "x_time"], output_names=["out_spec", "out_time"],
        opset_version=18, dynamo=True, external_data=False,
    )
    m, ok = simplify(onnx.load(str(raw)))
    assert ok, "onnxsim validation failed"
    onnx.save(m, str(out_path))
    raw.unlink()
    print(f"[OK] Saved {out_path} ({os.path.getsize(out_path) / 2**20:.1f} MB, {len(m.graph.node)} nodes)")

    # Core ML only schedules fp16 graphs on the ANE, and ORT's CoreML EP does not downcast
    # by itself, so ship an fp16 copy (fp16 inputs/outputs too) for the ANE path.
    from onnxconverter_common import float16

    fp16_path = out_path.with_name(out_path.stem + "_fp16.onnx")
    onnx.save(float16.convert_float_to_float16(m, keep_io_types=False), str(fp16_path))
    print(f"[OK] Saved {fp16_path} ({os.path.getsize(fp16_path) / 2**20:.1f} MB)")


def debug_taps(ml, model, ref):
    """Per-layer SNR of the ANE run vs PyTorch FP32 - shows where fp16 error accumulates."""
    x_spec, x_time = ref["x_spec"][:1], fold_time(ref["x_time"][:1])
    with torch.no_grad():
        want = [t.numpy() for t in model(torch.from_numpy(x_spec), torch.from_numpy(x_time))]
    got = ml.predict({"x_spec": x_spec, "x_time": x_time})
    print("\n=== Per-layer SNR on the ANE (song 1) ===")
    for name, w in zip(["out_spec", "out_time"] + TAP_NAMES, want):
        key = name if name in got else next(k for k in got if k.startswith(name))
        print(f"  {name:13s} {snr_db(w, got[key]):6.1f} dB")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=SOURCES, default="drums")
    p.add_argument("--out", default=None, help="Default: ane/models/htdemucs_<source>_ane.mlpackage")
    p.add_argument("--onnx", action="store_true", help="Also export ONNX (fp32 + fp16) next to the .mlpackage")
    p.add_argument("--debug-taps", action="store_true", help="Export with intermediate outputs and print per-layer SNR")
    p.add_argument("--ref", default=None, help="Real-audio reference (default: per source; built if missing)")
    p.add_argument("--skip-parity", action="store_true")
    args = p.parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    args.ref = args.ref or str(ref_path(args.source))
    print(f"Step 1: Building ANE {args.source} model from htdemucs f7e0c4bc...")
    model, base = build_ane_model(source=args.source)

    print("\nStep 2: PyTorch FP32 parity...")
    check_torch_parity(model, base)

    if args.source != "drums" and not Path(args.ref).exists():
        print(f"\n[ref] Building {args.source} reference from the drums reference chunks...")
        make_ref(args.source, out=args.ref)
    ref = load_ref(args.ref)
    if ref is not None:
        example = (torch.from_numpy(ref["x_spec"][:1]), torch.from_numpy(fold_time(ref["x_time"][:1])))
    else:
        example = (torch.randn(SPEC_SHAPE), torch.randn(TIME_SHAPE))

    out = Path(args.out or mlpackage_path(args.source))
    if args.debug_taps:
        model.debug_taps = True
        out = out.with_name(out.stem + "_taps.mlpackage")
    print(f"\nStep 3: Converting to Core ML (FP16 ML Program) -> {out}")
    ml = export_coreml(model, out, example, args.source)

    print("\nStep 4: Compute plan audit...")
    compute_plan_summary(str(out), "CPU_AND_NE")

    if args.onnx and not args.debug_taps:
        print(f"\nStep 5: ONNX export -> {onnx_path(args.source)}")
        export_onnx(model, onnx_path(args.source), example)

    if args.debug_taps:
        if ref is None:
            sys.exit("--debug-taps needs reference data")
        debug_taps(ml, model, ref)
        return 0

    if args.skip_parity:
        return 0
    print("\nStep 6: Real-audio parity on the Neural Engine...")
    import parity_ane

    if ref is None:
        print(f"[skip] {args.ref} not found")
        return 0
    run = parity_ane.CoreMLRunner(str(out), "CPU_AND_NE")
    return parity_ane.evaluate(run, ref, 40.0)


if __name__ == "__main__":
    sys.exit(main())
