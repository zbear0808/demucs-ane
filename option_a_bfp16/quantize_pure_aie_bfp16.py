import os
import sys
import copy
import time
from pathlib import Path
import numpy as np
import onnx
from onnx import shape_inference

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# Setup MSVC & Ninja BEFORE quark.onnx imports
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from env_setup import ensure_build_environment
ensure_build_environment()

from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config


def infer_complete_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Temporarily aliases BFPQuantizeDequantize to Identity to allow ONNX's native
    C++ shape inference engine to infer complete shapes across all intermediate tensors.
    """
    bfp_node_indices = []
    original_attrs = {}

    for idx, n in enumerate(model.graph.node):
        if n.domain == "com.amd.quark" and n.op_type == "BFPQuantizeDequantize":
            bfp_node_indices.append(idx)
            original_attrs[idx] = list(n.attribute)
            n.domain = ""
            n.op_type = "Identity"
            del n.attribute[:]

    inferred = shape_inference.infer_shapes(model)

    # Restore BFP nodes on inferred model
    for idx in bfp_node_indices:
        n = inferred.graph.node[idx]
        n.domain = "com.amd.quark"
        n.op_type = "BFPQuantizeDequantize"
        n.attribute.extend(original_attrs[idx])

    return inferred


def repair_bfp16_graph(bfp_model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict]:
    stats = {
        "scalar_to_identity": 0,
        "axis_swapped": 0,
        "unaligned_to_identity": 0,
        "valid_kept": 0,
    }

    print("Inferring complete shapes across all graph tensors...")
    inferred = infer_complete_shapes(bfp_model)

    shapes = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        if vi.type.tensor_type.HasField("shape"):
            shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    for init in inferred.graph.initializer:
        shapes[init.name] = list(init.dims)

    print(f"Collected concrete shapes for {len(shapes)} tensors.")

    for n in inferred.graph.node:
        if n.domain == "com.amd.quark" and n.op_type == "BFPQuantizeDequantize":
            in_name = n.input[0] if n.input else ""
            shape = shapes.get(in_name)
            if shape is None:
                stats["valid_kept"] += 1
                continue

            rank = len(shape)
            axis = 0
            block_size = 8
            for attr in n.attribute:
                if attr.name == "axis":
                    axis = attr.i
                elif attr.name == "block_size":
                    block_size = attr.i

            norm_axis = axis if axis >= 0 else rank + axis

            if rank == 0 or norm_axis >= rank:
                n.domain = ""
                n.op_type = "Identity"
                del n.attribute[:]
                stats["scalar_to_identity"] += 1
            elif shape[norm_axis] > 0 and shape[norm_axis] % block_size == 0:
                stats["valid_kept"] += 1
            else:
                # Check for alternative axis divisible by block_size
                alt_axes = [i for i, d in enumerate(shape) if d > 0 and d % block_size == 0]
                if alt_axes:
                    for attr in n.attribute:
                        if attr.name == "axis":
                            attr.i = alt_axes[0]
                    stats["axis_swapped"] += 1
                else:
                    # No axis is divisible by block_size -> Bypass with Identity
                    n.domain = ""
                    n.op_type = "Identity"
                    del n.attribute[:]
                    stats["unaligned_to_identity"] += 1

    return inferred, stats


class CalibrationDataReader:
    """Feeds real-music chunks from parity_real_audio.npz; falls back to random noise."""

    def __init__(self, num_samples=2):
        self.idx = 0
        ref_path = SCRIPT_DIR / "parity_real_audio.npz"
        if ref_path.exists():
            ref = np.load(ref_path)
            self.samples = [
                {"x_spec": ref["x_spec"][i : i + 1], "x_time": ref["x_time"][i : i + 1]} for i in range(len(ref["songs"]))
            ]
            print(f"  [Calib] Using {len(self.samples)} real-music chunks from {ref_path.name}")
        else:
            rng = np.random.default_rng(42)
            self.samples = [
                {
                    "x_spec": rng.standard_normal((1, 4, 2048, 336)).astype(np.float32),
                    "x_time": rng.standard_normal((1, 2, 1, 344064)).astype(np.float32),
                }
                for _ in range(num_samples)
            ]
            print(f"  [Calib] {ref_path.name} not found, using {num_samples} random-noise samples")

    def get_next(self):
        if self.idx >= len(self.samples):
            return None
        self.idx += 1
        print(f"  [Calib] Sample {self.idx}/{len(self.samples)}...", flush=True)
        return self.samples[self.idx - 1]


def run_quantization(tag: str = "_v2", op_types: list[str] | None = None, include_cle: bool = False, suffix: str = "bfp16"):
    models_dir = SCRIPT_DIR / "models"
    input_path = str(models_dir / f"htdemucs_candidate_a_pure_aie{tag}_simp.onnx")
    raw_bfp_path = str(models_dir / f"htdemucs_candidate_a_pure_aie{tag}_{suffix}_raw.onnx")
    final_bfp_path = str(models_dir / f"htdemucs_candidate_a_pure_aie{tag}_{suffix}.onnx")

    assert os.path.exists(input_path), f"Input model not found: {input_path}"

    print("=" * 76)
    print("Step 1: Quantizing Pure AIE Model to AMD Quark BFP16")
    print(f"Input Model:  {input_path} ({os.path.getsize(input_path)/(1024*1024):.2f} MB)")
    print(f"Raw Output:   {raw_bfp_path}")
    print(f"Final Output: {final_bfp_path}")
    print(f"Op types:     {op_types or 'ALL (Quark default)'} | CLE: {include_cle}")
    print("=" * 76)

    t0 = time.perf_counter()
    quant_config = copy.deepcopy(get_default_config("BFP16"))
    # CLE rescales weights across activations; that is only exact for ReLU-like
    # activations, not the GELU/GLU used throughout Demucs.
    quant_config.include_cle = include_cle
    if op_types:
        quant_config.op_types_to_quantize = list(op_types)
    config = Config(global_quant_config=quant_config)
    quantizer = ModelQuantizer(config)

    calib_reader = CalibrationDataReader(num_samples=2)
    quantizer.quantize_model(
        model_input=input_path,
        model_output=raw_bfp_path,
        calibration_data_reader=calib_reader,
    )
    print(f"[OK] Raw BFP16 quantization completed in {time.perf_counter() - t0:.1f}s")

    print("\n" + "=" * 76)
    print("Step 2: Repairing BFP16 Attributes and Resolving Complete Shapes")
    print("=" * 76)
    raw_model = onnx.load(raw_bfp_path)
    repaired_model, stats = repair_bfp16_graph(raw_model)

    from export_candidate_a_pure_aie import split_multi_axis_reduce_means

    repaired_model, n_split = split_multi_axis_reduce_means(repaired_model)
    print(f"[Precision Guard] Split {n_split} multi-axis ReduceMean nodes re-fused during quantization.")
    onnx.save(repaired_model, final_bfp_path)

    print("\nRepair Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"[OK] Saved repaired BFP16 model: {final_bfp_path} ({os.path.getsize(final_bfp_path)/(1024*1024):.2f} MB)")

    # Final verification of scissors
    scissor_ops = ["InstanceNormalization", "ConvTranspose"]
    found_scissors = {}
    asym_convs = 0
    for n in repaired_model.graph.node:
        if n.op_type in scissor_ops:
            found_scissors[n.op_type] = found_scissors.get(n.op_type, 0) + 1
        elif n.op_type == "Conv":
            for a in n.attribute:
                if a.name == "kernel_shape" and list(a.ints) in [[8, 1], [1, 8]]:
                    asym_convs += 1

    print("\n=== Pure AIE Final Scissor Verification ===")
    for s in scissor_ops:
        print(f"  {s}: {found_scissors.get(s, 0)}")
    print(f"  Asymmetric Convs [8, 1]/[1, 8]: {asym_convs}")
    assert len(found_scissors) == 0 and asym_convs == 0, "Scissor operators detected!"
    print("[SUCCESS] ZERO scissor operators remain in the quantized graph!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize the Pure AIE HTDemucs model to Quark BFP16")
    parser.add_argument("--tag", default="_v2", help="Model tag used by export_candidate_a_pure_aie.py (default: _v2)")
    parser.add_argument(
        "--op-types", nargs="*", default=["Conv", "MatMul", "Gemm"],
        help="Op types to quantize (default: Conv MatMul Gemm). Pass with no values to quantize everything.",
    )
    parser.add_argument("--cle", action="store_true", help="Enable Quark cross-layer equalization")
    parser.add_argument("--suffix", default=None, help="Output name suffix (default: bfp16_gemm, or bfp16_all for all ops)")
    a = parser.parse_args()
    suffix = a.suffix or ("bfp16_gemm" if a.op_types else "bfp16_all")
    run_quantization(a.tag, a.op_types, a.cle, suffix)

