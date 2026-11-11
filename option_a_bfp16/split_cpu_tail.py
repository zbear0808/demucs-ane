"""
Split the FP32 HTDemucs drums ONNX graph into an NPU "body" and a CPU "tail".

The last frequency decoder layer (decoder.3: rewrite conv, GLU, DConv, final
polyphase ConvTranspose) is where bf16 precision hurts most: running only that
layer in bf16 drops the frequency branch to ~28 dB SNR, while every other
layer in bf16 together still gives ~41-47 dB. It only costs ~0.2 s per 7.8 s
chunk on the CPU, so keep it in FP32 there.

  body: x_spec, x_time -> <cut tensor>, out_time   (run on VitisAIExecutionProvider)
  tail: <cut tensor>   -> out_spec                 (run on CPUExecutionProvider)

Usage (.venv):
  python option_a_bfp16\\split_cpu_tail.py --model option_a_bfp16\\models\\htdemucs_candidate_a_pure_aie_v2_simp.onnx
"""
import argparse
import os
import sys
from pathlib import Path

import onnx
from onnx.utils import Extractor

SCRIPT_DIR = Path(__file__).resolve().parent
# Input of decoder.3 (y = x + skip) in the v2 export; see find_cut_tensor() to locate it again.
DEFAULT_CUT = "add_190"


def find_cut_tensor(raw_model_path: str, scope: str = "decoder.3") -> str:
    """Finds the first input tensor of `scope` using the exporter's name_scopes metadata (raw export only)."""
    m = onnx.load(raw_model_path, load_external_data=False)
    for n in m.graph.node:
        md = {p.key: p.value for p in n.metadata_props}
        scopes = md.get("pkg.torch.onnx.name_scopes", "")
        if f"'{scope}'" in scopes:
            return n.output[0] if n.op_type == "Add" else n.input[0]
    raise ValueError(f"No nodes found for scope {scope}")


def split(model_path: str, cut: str, body_path: str, tail_path: str):
    model = onnx.load(model_path)
    produced = {o for n in model.graph.node for o in n.output}
    assert cut in produced, f"Cut tensor {cut!r} not found in {model_path}"

    ex = Extractor(model)
    body = ex.extract_model(["x_spec", "x_time"], [cut, "out_time"])
    tail = ex.extract_model([cut], ["out_spec"])
    for part, path in ((body, body_path), (tail, tail_path)):
        onnx.checker.check_model(part)
        onnx.save(part, path)
        ops = {}
        for n in part.graph.node:
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
        print(f"[OK] {path}: {len(part.graph.node)} nodes, {os.path.getsize(path) / 2**20:.1f} MB")
        print(f"     inputs {[i.name for i in part.graph.input]} -> outputs {[o.name for o in part.graph.output]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(SCRIPT_DIR / "models" / "htdemucs_candidate_a_pure_aie_v2_simp.onnx"))
    p.add_argument("--cut", default=DEFAULT_CUT, help="Tensor to cut at (default: decoder.3 input)")
    p.add_argument("--find-cut-in", default=None, help="Raw export to locate the decoder.3 input tensor from")
    args = p.parse_args()

    cut = find_cut_tensor(args.find_cut_in) if args.find_cut_in else args.cut
    base = args.model[: -len(".onnx")]
    if base.endswith("_simp"):
        base = base[: -len("_simp")]
    split(args.model, cut, f"{base}_npu_body.onnx", f"{base}_cpu_tail.onnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
