import os
import sys
import copy
import time
import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference, TensorProto
from pathlib import Path

# Setup MSVC & Ninja BEFORE quark.onnx imports
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from env_setup import ensure_build_environment
ensure_build_environment()

from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config


def decompose_instance_norms(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """
    Decomposes InstanceNormalization nodes (where input is [1, 1, L]) into elementary
    ONNX operators (ReduceMean, Sub, Mul, Add, Sqrt, Div) natively supported by VAI-ML on AIE.
    """
    graph = model.graph
    new_nodes = []
    new_initializers = []
    decomp_count = 0

    # Shared axis initializer [2]
    axis_name = "shared_inst_norm_axis_2"
    axis_init = helper.make_tensor(axis_name, TensorProto.INT64, [1], [2])
    new_initializers.append(axis_init)

    for node in graph.node:
        if node.op_type == "InstanceNormalization":
            x_in = node.input[0]
            scale_in = node.input[1]
            bias_in = node.input[2]
            y_out = node.output[0]

            eps = 1e-5
            for attr in node.attribute:
                if attr.name == "epsilon":
                    eps = attr.f

            eps_name = f"{node.name}_eps"
            eps_init = helper.make_tensor(eps_name, TensorProto.FLOAT, [1], [eps])
            new_initializers.append(eps_init)

            mean_out = f"{node.name}_mean"
            diff_out = f"{node.name}_diff"
            sq_diff_out = f"{node.name}_sq_diff"
            var_out = f"{node.name}_var"
            var_eps_out = f"{node.name}_var_eps"
            std_out = f"{node.name}_std"
            inv_std_out = f"{node.name}_inv_std"
            scaled_out = f"{node.name}_scaled"

            n_mean = helper.make_node("ReduceMean", [x_in, axis_name], [mean_out], name=f"{node.name}_mean", keepdims=1)
            n_diff = helper.make_node("Sub", [x_in, mean_out], [diff_out], name=f"{node.name}_diff")
            n_sq = helper.make_node("Mul", [diff_out, diff_out], [sq_diff_out], name=f"{node.name}_sq")
            n_var = helper.make_node("ReduceMean", [sq_diff_out, axis_name], [var_out], name=f"{node.name}_var", keepdims=1)
            n_add_eps = helper.make_node("Add", [var_out, eps_name], [var_eps_out], name=f"{node.name}_add_eps")
            n_sqrt = helper.make_node("Sqrt", [var_eps_out], [std_out], name=f"{node.name}_sqrt")
            n_div = helper.make_node("Div", [diff_out, std_out], [inv_std_out], name=f"{node.name}_div")
            n_scale = helper.make_node("Mul", [inv_std_out, scale_in], [scaled_out], name=f"{node.name}_scale")
            n_bias = helper.make_node("Add", [scaled_out, bias_in], [y_out], name=f"{node.name}_bias")

            new_nodes.extend([n_mean, n_diff, n_sq, n_var, n_add_eps, n_sqrt, n_div, n_scale, n_bias])
            decomp_count += 1
        else:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_initializers)
    return model, decomp_count


def infer_complete_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Temporarily aliases BFPQuantizeDequantize to Identity to allow ONNX's native
    C++ shape inference engine to infer complete shapes across all 4,600+ intermediate tensors.
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

    # First infer complete shapes
    inferred = infer_complete_shapes(bfp_model)

    shapes = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        if vi.type.tensor_type.HasField("shape"):
            shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    for init in inferred.graph.initializer:
        shapes[init.name] = list(init.dims)

    for n in inferred.graph.node:
        if n.domain == "com.amd.quark" and n.op_type == "BFPQuantizeDequantize":
            in_name = n.input[0] if n.input else ""
            shape = shapes.get(in_name)
            if shape is None:
                # Fallback: keep valid or convert to identity
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

            if rank == 0:
                n.domain = ""
                n.op_type = "Identity"
                del n.attribute[:]
                stats["scalar_to_identity"] += 1
            elif norm_axis >= rank:
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


if __name__ == "__main__":
    fp32_path = str(WORKSPACE_ROOT / "attempt2_explicit_padding" / "models" / "htdemucs_candidate_a_simp.onnx")
    out_dir = SCRIPT_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    decomp_fp32_path = str(out_dir / "htdemucs_candidate_a_decomp.onnx")

    print("Step 1: Decomposing InstanceNorm in FP32 model...")
    m = onnx.load(fp32_path)
    m_decomp, decomp_count = decompose_instance_norms(m)
    onnx.save(m_decomp, decomp_fp32_path)
    print(f"Decomposed {decomp_count} InstanceNorm nodes -> {decomp_fp32_path}")

    raw_bfp_path = str(out_dir / "htdemucs_candidate_a_bfp16_v2_raw.onnx")
    final_bfp_path = str(out_dir / "htdemucs_candidate_a_bfp16_v2.onnx")

    print("\nStep 2: Running BFP16 Quantization on decomposed model...")
    class SimpleCalibReader:
        def __init__(self, num_samples=2):
            self.num_samples = num_samples
            self.idx = 0
            np.random.seed(42)

        def get_next(self):
            if self.idx >= self.num_samples:
                return None
            self.idx += 1
            print(f"  [Calib] Generating sample {self.idx}/{self.num_samples}...", flush=True)
            return {
                "x_spec": np.random.randn(1, 4, 2048, 336).astype(np.float32),
                "x_time": np.random.randn(1, 2, 1, 344064).astype(np.float32),
            }

    if not os.path.exists(raw_bfp_path):
        quant_config = copy.deepcopy(get_default_config("BFP16"))
        config = Config(global_quant_config=quant_config)
        quantizer = ModelQuantizer(config)
        t0 = time.perf_counter()
        quantizer.quantize_model(
            model_input=decomp_fp32_path,
            model_output=raw_bfp_path,
            calibration_data_reader=SimpleCalibReader(num_samples=2),
        )
        print(f"Raw BFP16 quantized in {time.perf_counter() - t0:.1f}s -> {raw_bfp_path}")
    else:
        print(f"Using existing raw BFP16 model: {raw_bfp_path}")

    print("\nStep 3: Repairing BFP16 attributes and populating complete shapes...")
    raw_m = onnx.load(raw_bfp_path)
    repaired_m, stats = repair_bfp16_graph(raw_m)
    onnx.save(repaired_m, final_bfp_path)
    print("Repair Stats:", stats)
    print(f"Final Model Saved: {final_bfp_path} ({os.path.getsize(final_bfp_path)/(1024*1024):.2f} MB)")
