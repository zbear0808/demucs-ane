"""Shared helpers for the Apple Neural Engine build: SNR metrics, Core ML compute-plan audit, model loading."""
import collections
import sys
from pathlib import Path

import numpy as np

ANE_DIR = Path(__file__).resolve().parent
ROOT = ANE_DIR.parent
MODELS_DIR = ANE_DIR / "models"
DEFAULT_MLPACKAGE = MODELS_DIR / "htdemucs_drums_ane.mlpackage"
DEFAULT_ONNX = MODELS_DIR / "htdemucs_drums_ane.onnx"
DEFAULT_REF = ROOT / "option_a_bfp16" / "parity_real_audio.npz"

sys.path.insert(0, str(ROOT / "option_a_bfp16"))
from htdemucs_dsp import SAMPLE_RATE, SEGMENT_LEN, cpu_postproc_numpy, cpu_preproc_numpy  # noqa: E402,F401

from ane_modules import SPEC_SHAPE, TIME_SHAPE  # noqa: E402

COMPUTE_UNITS = ("CPU_AND_NE", "ALL", "CPU_AND_GPU", "CPU_ONLY")


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref.astype(np.float64)
    err = ref - est.astype(np.float64).reshape(ref.shape)
    if not np.isfinite(err).all():
        return float("nan")
    return float(10 * np.log10(np.sum(ref**2) / (np.sum(err**2) + 1e-30)))


def fold_time(x_time: np.ndarray) -> np.ndarray:
    """[1, 2, 1, 344064] or [1, 2, 344064] -> [1, 2, 32, 10752] (free reshape)."""
    return np.ascontiguousarray(x_time, dtype=np.float32).reshape(TIME_SHAPE)


def unfold_time(out_time: np.ndarray) -> np.ndarray:
    """[1, 2, 32, 10752] -> [1, 2, 1, 344064], the layout cpu_postproc_numpy expects."""
    return out_time.reshape(1, 2, 1, SEGMENT_LEN)


def compute_unit(name: str):
    import coremltools as ct

    return getattr(ct.ComputeUnit, name)


def compute_plan_summary(mlpackage: str, units: str = "CPU_AND_NE", top: int = 15) -> dict:
    """
    Ask Core ML where each op will run (MLComputePlan, macOS 14.4+). This is the Apple
    equivalent of the VAI-ML partition summary: it lists the preferred device per op and
    an estimated relative cost, without needing to run the model.
    """
    import coremltools as ct
    from coremltools.models.compute_device import MLNeuralEngineComputeDevice
    from coremltools.models.compute_plan import MLComputePlan

    model = ct.models.MLModel(str(mlpackage), compute_units=compute_unit(units))
    plan = MLComputePlan.load_from_path(model.get_compiled_model_path(), compute_units=compute_unit(units))
    ops = plan.model_structure.program.functions["main"].block.operations

    count, cost, off_ane = collections.Counter(), collections.Counter(), collections.Counter()
    for op in ops:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:  # consts etc.
            continue
        dev = type(usage.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
        est = plan.get_estimated_cost_for_mlprogram_operation(op)
        w = est.weight if est is not None else 0.0
        count[dev] += 1
        cost[dev] += w
        if dev != "NeuralEngine":
            ane_ok = any(isinstance(d, MLNeuralEngineComputeDevice) for d in usage.supported_compute_devices)
            off_ane[(op.operator_name, dev, "ANE-capable" if ane_ok else "not ANE-capable")] += 1

    total_ops, total_cost = sum(count.values()), sum(cost.values()) or 1.0
    print(f"\n=== Core ML compute plan ({units}) ===")
    for dev in sorted(count, key=lambda d: -count[d]):
        print(f"  {dev:13s} {count[dev]:5d} ops ({100 * count[dev] / total_ops:5.1f}%)"
              f" | est. cost {100 * cost[dev] / total_cost:5.1f}%")
    if off_ane:
        print("  Ops not on the Neural Engine:")
        for (name, dev, why), n in off_ane.most_common(top):
            print(f"    {n:4d} x {name:28s} -> {dev} ({why})")
    return {"count": dict(count), "cost": dict(cost), "off_ane": off_ane}


def load_ref(path=DEFAULT_REF):
    if not Path(path).exists():
        return None
    return np.load(path)
