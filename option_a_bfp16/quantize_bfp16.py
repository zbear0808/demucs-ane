import os
import sys
import copy
import time
import argparse
from pathlib import Path

# Setup MSVC & Ninja BEFORE quark.onnx imports
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
from env_setup import ensure_build_environment
ensure_build_environment()

import numpy as np
import onnx
from onnx import helper, shape_inference
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config


class DemucsBFP16CalibrationReader:
    """Calibration data reader supporting both 2-channel (original) and 4-channel audio."""
    def __init__(self, num_samples=2, time_channels=2):
        self.num_samples = num_samples
        self.time_channels = time_channels
        self.idx = 0
        np.random.seed(42)

    def get_next(self):
        if self.idx >= self.num_samples:
            return None
        self.idx += 1
        print(f"  [Calibration] Generating sample {self.idx}/{self.num_samples} (time_channels={self.time_channels})...", flush=True)

        if self.time_channels == 4:
            # 4-channel: first 2 are stereo signal, remaining 2 are padded channels
            x_time = np.zeros((1, 4, 1, 344064), dtype=np.float32)
            x_time[:, :2, :, :] = np.random.randn(1, 2, 1, 344064).astype(np.float32)
        else:
            # 2-channel: stereo signal
            x_time = np.random.randn(1, 2, 1, 344064).astype(np.float32)

        return {
            "x_spec": np.random.randn(1, 4, 2048, 336).astype(np.float32),
            "x_time": x_time,
        }


def detect_time_channels(model_path: str) -> int:
    """Inspect model graph inputs to determine whether x_time expects 2 or 4 channels."""
    m = onnx.load(model_path, load_external_data=False)
    for inp in m.graph.input:
        if inp.name == "x_time":
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            if len(dims) > 1 and dims[1] > 0:
                return dims[1]
    return 2


def repair_bfp16_axis_attributes(model_proto: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """
    Repairs Quark BFPQuantizeDequantize nodes on rank-0 scalar tensors where
    axis=0 triggers VAI-ML compiler failure:
    "op axis attribute value 0 is out of range [-0, 0)"
    """
    try:
        inferred = shape_inference.infer_shapes(model_proto)
    except Exception:
        inferred = model_proto

    value_info_ranks = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        if vi.type.tensor_type.HasField("shape"):
            value_info_ranks[vi.name] = len(vi.type.tensor_type.shape.dim)

    for init in inferred.graph.initializer:
        value_info_ranks[init.name] = len(init.dims)

    repaired_count = 0

    for node in inferred.graph.node:
        if node.domain == "com.amd.quark" and node.op_type == "BFPQuantizeDequantize":
            in_name = node.input[0] if node.input else ""
            rank = value_info_ranks.get(in_name, None)

            if rank == 0:
                node.domain = ""
                node.op_type = "Identity"
                del node.attribute[:]
                repaired_count += 1
            elif rank is not None and rank > 0:
                for attr in node.attribute:
                    if attr.name == "axis":
                        if attr.i >= rank or attr.i < -rank:
                            attr.i = 0
                            repaired_count += 1

    return inferred, repaired_count


def quantize_bfp16_model(input_model: str, output_model: str, num_calib_samples: int = 2):
    print("\n" + "=" * 76, flush=True)
    print(f"Quantizing to AMD Quark BFP16 with MLIR Axis Repair", flush=True)
    print(f"Input:  {input_model}", flush=True)
    print(f"Output: {output_model}", flush=True)
    print(f"Samples: {num_calib_samples}", flush=True)
    print("=" * 76, flush=True)

    assert os.path.exists(input_model), f"Input model not found: {input_model}"
    os.makedirs(os.path.dirname(os.path.abspath(output_model)), exist_ok=True)

    time_channels = detect_time_channels(input_model)
    print(f"Detected x_time channels: {time_channels}", flush=True)

    quant_config = copy.deepcopy(get_default_config("BFP16"))
    config = Config(global_quant_config=quant_config)

    temp_raw_bfp = output_model + ".raw.onnx"
    t0 = time.perf_counter()
    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(
        model_input=input_model,
        model_output=temp_raw_bfp,
        calibration_data_reader=DemucsBFP16CalibrationReader(
            num_samples=num_calib_samples,
            time_channels=time_channels,
        ),
    )
    quant_time = time.perf_counter() - t0
    print(f"[OK] Raw BFP16 model generated in {quant_time:.1f}s -> {temp_raw_bfp}", flush=True)

    print("Post-processing graph to repair rank-0 scalar axis attributes...", flush=True)
    model = onnx.load(temp_raw_bfp)
    repaired_model, count = repair_bfp16_axis_attributes(model)
    onnx.save(repaired_model, output_model)
    print(f"[OK] Repaired {count} scalar BFP nodes. Saved final model: {output_model}", flush=True)

    if os.path.exists(temp_raw_bfp):
        try:
            os.remove(temp_raw_bfp)
        except Exception:
            pass

    out_size_mb = os.path.getsize(output_model) / (1024 * 1024)
    print(f"[SUMMARY] Output size: {out_size_mb:.2f} MB | Repaired nodes: {count}", flush=True)
    return output_model, count, out_size_mb


def main():
    parser = argparse.ArgumentParser(description="Quantize Candidate A models to Quark BFP16 with axis repair.")
    parser.add_argument("--target", choices=["all", "candidate_a", "candidate_a_4ch"], default="all",
                        help="Which candidate model(s) to quantize")
    parser.add_argument("--samples", type=int, default=2, help="Number of calibration samples")
    args = parser.parse_args()

    workspace_root = script_dir.parent
    models_dir = script_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    if args.target in ["all", "candidate_a"]:
        targets.append({
            "name": "Candidate A (2-channel audio)",
            "in": str(workspace_root / "attempt2_explicit_padding" / "models" / "htdemucs_candidate_a_simp.onnx"),
            "out": str(models_dir / "htdemucs_candidate_a_bfp16.onnx"),
        })
    if args.target in ["all", "candidate_a_4ch"]:
        targets.append({
            "name": "Candidate A 4-Channel (4-channel audio)",
            "in": str(workspace_root / "attempt2_explicit_padding" / "models" / "htdemucs_candidate_a_4ch_simp.onnx"),
            "out": str(models_dir / "htdemucs_candidate_a_4ch_bfp16.onnx"),
        })

    for t in targets:
        print(f"\nProcessing {t['name']}...")
        quantize_bfp16_model(t["in"], t["out"], num_calib_samples=args.samples)

    print("\n[ALL COMPLETE] Quantization completed successfully.")


if __name__ == "__main__":
    main()
