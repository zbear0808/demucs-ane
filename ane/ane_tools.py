"""Shared helpers for the Apple Neural Engine build: SNR metrics, Core ML compute-plan audit, model loading."""
import collections
import sys
from pathlib import Path

import numpy as np

ANE_DIR = Path(__file__).resolve().parent
ROOT = ANE_DIR.parent
MODELS_DIR = ANE_DIR / "models"
DRUMS_REF = ROOT / "option_a_bfp16" / "parity_real_audio.npz"


def mlpackage_path(source: str = "drums") -> Path:
    return MODELS_DIR / f"htdemucs_{source}_ane.mlpackage"


def onnx_path(source: str = "drums") -> Path:
    return MODELS_DIR / f"htdemucs_{source}_ane.onnx"


def fused_mlpackage_path(source: str = "drums") -> Path:
    """STFT + separation + iSTFT in one Core ML program."""
    return MODELS_DIR / f"htdemucs_{source}_fused_ane.mlpackage"


def fused_onnx_path(source: str = "drums", fp16: bool = True) -> Path:
    return MODELS_DIR / f"htdemucs_{source}_fused_ane{'_fp16' if fp16 else ''}.onnx"


# I/O names of the two model families (both take two inputs and return two outputs).
SPLIT_IO = {"inputs": ("x_spec", "x_time"), "outputs": ("out_spec", "out_time")}
FUSED_IO = {"inputs": ("blocks", "x_time"), "outputs": ("wav_blocks", "out_time")}


def ref_path(source: str = "drums") -> Path:
    """Drums reuses the XDNA2 reference; other sources get one built by make_ref()."""
    return DRUMS_REF if source == "drums" else ANE_DIR / f"parity_real_audio_{source}.npz"


DEFAULT_MLPACKAGE = mlpackage_path("drums")
DEFAULT_ONNX = onnx_path("drums")
DEFAULT_REF = DRUMS_REF

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


COMPILED_DIR = MODELS_DIR / "compiled"


def compiled_path(mlpackage) -> Path:
    """
    Compile `mlpackage` to a persistent .mlmodelc (once, ~0.3 s) and return its path.

    The ~25 s "load time" of these models is not the mlpackage compile. It is Core ML
    building the ANE program, and the OS caches that keyed on the compiled model's path.
    ct.models.MLModel(mlpackage) compiles into a new temp dir each time, so it pays the
    25 s on every launch. Loading a fixed .mlmodelc path pays it once, and later loads
    (including from new processes) take ~0.1 s.
    """
    import coremltools as ct

    src = Path(mlpackage)
    # Key on the files that hold the model, not on Manifest.json: coremltools rewrites that file (same content,
    # new mtime) every time it opens the package, which used to change the key and force a 25 s ANE rebuild.
    model_files = [f for f in src.rglob("*") if f.is_file() and f.name != "Manifest.json"]
    newest_src = int(max(f.stat().st_mtime for f in model_files))
    # The build time goes into the name, so a rebuilt package never reuses a stale OS-side ANE cache entry.
    dst = COMPILED_DIR / f"{src.stem}_{newest_src}.mlmodelc"
    if not dst.exists():
        import shutil

        COMPILED_DIR.mkdir(parents=True, exist_ok=True)
        for old in COMPILED_DIR.glob(f"{src.stem}_*.mlmodelc"):
            shutil.rmtree(old, ignore_errors=True)
        tmp = ct.utils.compile_model(str(src))
        shutil.move(tmp, dst)
    return dst


def load_coreml(mlpackage, units: str = "CPU_AND_NE"):
    """Load via a cached .mlmodelc (see compiled_path). Has the same .predict() as MLModel."""
    import coremltools as ct

    return ct.models.CompiledMLModel(str(compiled_path(mlpackage)), compute_units=compute_unit(units))


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


def convert_fp16(module, example: tuple, in_names: list, out_names: list, path) -> "ct.models.MLModel":
    """torch.export -> fp16 ML Program, saved to `path` (the same settings as the main model)."""
    import shutil

    import coremltools as ct
    import torch

    with torch.no_grad():
        ep = torch.export.export(module.eval(), example).run_decompositions({})
    ml = ct.convert(
        ep,
        inputs=[ct.TensorType(n, shape=tuple(e.shape)) for n, e in zip(in_names, example)],
        outputs=[ct.TensorType(n) for n in out_names],
        convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14, compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    ml.save(str(path))
    return ml


def load_ref(path=DEFAULT_REF):
    if not Path(path).exists():
        return None
    return np.load(path)


def make_ref(source: str, chunks_from=DRUMS_REF, out=None) -> Path:
    """
    Build a real-audio reference for `source` in the same format as parity_real_audio.npz,
    reusing the music chunks stored in the drums reference (so no audio files are needed).
    """
    import torch

    from ane_modules import build_ane_model, load_base_model
    from option_a_bfp16.export_candidate_a_pure_aie import HTDemucsCandidateAPureAIECore

    out = Path(out or ref_path(source))
    src = np.load(chunks_from)
    stock = load_base_model()
    stock.use_train_segment = False  # 344064 > training segment; let forward() accept it
    s_idx = list(stock.sources).index(source)
    _, single = build_ane_model(stock, source)
    core = HTDemucsCandidateAPureAIECore(single).eval()

    keys = ["chunk", "x_spec", "x_time", "ref_spec", "ref_time", "stock_wav", "mean_s", "std_s", "mean_t", "std_t"]
    data = {k: [] for k in keys}
    for i, song in enumerate(src["songs"]):
        chunk = src["chunk"][i][None]
        x_spec, x_time, stats = cpu_preproc_numpy(chunk)
        x_time = x_time.astype(np.float32)
        with torch.no_grad():
            ref_spec, ref_time = core(torch.from_numpy(x_spec), torch.from_numpy(x_time))
            stock_wav = stock(torch.from_numpy(chunk))[0, s_idx].numpy()
        wav = cpu_postproc_numpy(ref_spec.numpy(), ref_time.numpy(), stats)[0]
        print(f"  {song}: PyTorch {source} core + NumPy DSP vs stock Demucs {snr_db(stock_wav, wav):.1f} dB")
        for k, v in zip(keys, [chunk[0], x_spec[0], x_time[0], ref_spec.numpy()[0], ref_time.numpy()[0],
                               stock_wav, *stats]):
            data[k].append(np.asarray(v, dtype=np.float32))
    np.savez(out, songs=src["songs"], **{k: np.stack(v) for k, v in data.items()})
    print(f"[OK] Saved {out}")
    return out
