import copy
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import onnx

def ensure_build_environment():
    """Ensure cl.exe (MSVC) and ninja.exe are in PATH so Quark custom ops compile / load cleanly."""
    workspace_root = Path(__file__).resolve().parent
    venv_scripts = workspace_root / ".venv" / "Scripts"
    if venv_scripts.exists() and str(venv_scripts) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{venv_scripts};{os.environ.get('PATH', '')}"

    if not shutil.which("cl.exe"):
        vcvars_candidates = [
            r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        ]
        for vcvars in vcvars_candidates:
            if os.path.exists(vcvars):
                try:
                    cmd = f'cmd.exe /c "call \"{vcvars}\" && set"'
                    output = subprocess.check_output(cmd, shell=True, text=True, errors="ignore")
                    for line in output.splitlines():
                        if "=" in line:
                            k, v = line.split("=", 1)
                            os.environ[k] = v
                    break
                except Exception as e:
                    print(f"[WARN] Failed to load {vcvars}: {e}")

    # Ensure .pyd to .dll alias exists in torch_extensions cache for Windows
    try:
        from quark.common.torch_cpp_build_specs import ONNX_CUSTOM_OPS_JIT_BASENAME
        from torch.utils.cpp_extension import _get_build_directory

        jit_dir = _get_build_directory(ONNX_CUSTOM_OPS_JIT_BASENAME, verbose=False)
        pyd_path = os.path.join(jit_dir, f"{ONNX_CUSTOM_OPS_JIT_BASENAME}.pyd")
        dll_path = os.path.join(jit_dir, f"{ONNX_CUSTOM_OPS_JIT_BASENAME}.dll")
        if os.path.exists(pyd_path) and not os.path.exists(dll_path):
            shutil.copy2(pyd_path, dll_path)
    except Exception:
        pass

ensure_build_environment()

from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config
try:
    from onnxruntime.quantization import QuantFormat, QuantType
except ImportError:
    from quark.onnx.quantization.quant_utils import VitisQuantFormat as QuantFormat, VitisQuantType as QuantType


class DemucsCalibrationReader:
    def __init__(self, num_samples=4):
        self.num_samples = num_samples
        self.idx = 0
        np.random.seed(42)

    def get_next(self):
        if self.idx >= self.num_samples:
            return None
        self.idx += 1
        return {
            "x_spec": np.random.randn(1, 4, 2048, 336).astype(np.float32),
            "x_time": np.random.randn(1, 2, 1, 344064).astype(np.float32),
        }


def convert_quark_bfp_to_vai_bfp(in_path: str, out_path: str):
    """
    Translates com.amd.quark:BFPQuantizeDequantize -> com.vai.quantize:BFPFixNeuron
    so Vitis AI passes looking for com.vai.quantize can match the nodes.
    """
    if not os.path.exists(in_path):
        print(f"[SKIP] Input model not found: {in_path}")
        return

    try:
        from quark.onnx.tools.convert_custom_ops import convert_custom_ops
        convert_custom_ops(in_path, out_path)
        print(f"[OK] Converted via quark.onnx.tools.convert_custom_ops -> {out_path}")
        return
    except Exception as e:
        print(f"[INFO] Direct API call fallback ({e}), rewriting ONNX proto domains directly...")

    model = onnx.load(in_path)
    converted = 0
    for node in model.graph.node:
        if node.domain == "com.amd.quark" and node.op_type == "BFPQuantizeDequantize":
            node.domain = "com.vai.quantize"
            node.op_type = "BFPFixNeuron"
            converted += 1

    # Ensure com.vai.quantize opset is declared in the model header
    has_vai_domain = any(op.domain == "com.vai.quantize" for op in model.opset_import)
    if not has_vai_domain:
        vai_opset = model.opset_import.add()
        vai_opset.domain = "com.vai.quantize"
        vai_opset.version = 1

    onnx.save(model, out_path)
    print(f"[OK] Converted {converted} nodes to com.vai.quantize:BFPFixNeuron -> {out_path}")


def audit_domains(onnx_path: str):
    if not os.path.exists(onnx_path):
        print(f"[WARN] Cannot audit non-existent file: {onnx_path}")
        return
    m = onnx.load(onnx_path, load_external_data=False)
    domains = {}
    for n in m.graph.node:
        key = f"{n.domain or 'ai.onnx'}:{n.op_type}"
        domains[key] = domains.get(key, 0) + 1
    custom = {k: v for k, v in domains.items() if not k.startswith("ai.onnx:")}
    print(f"  [{onnx_path}] Total nodes: {len(m.graph.node)} | Custom ops: {custom or 'NONE (100% Standard ONNX!)'}")


if __name__ == "__main__":
    src_model = "htdemucs_drums_xdna2_simp.onnx"

    if not os.path.exists(src_model):
        print(f"[ERROR] Source model not found: {src_model}")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Candidate 1: Convert existing BFP16 model to com.vai.quantize
    # -----------------------------------------------------------------
    print("\n--- Building Candidate 1: BFPFixNeuron (com.vai.quantize) ---")
    convert_quark_bfp_to_vai_bfp(
        "htdemucs_drums_xdna2_bfp16.onnx",
        "htdemucs_drums_xdna2_bfp16_vai.onnx",
    )
    audit_domains("htdemucs_drums_xdna2_bfp16_vai.onnx")

    # -----------------------------------------------------------------
    # Candidate 2: Standard ONNX QDQ A16W8 (Int16 Act / Int8 Weight)
    # Zero custom ops! High audio SDR + native XDNA 2 WAIC support
    # -----------------------------------------------------------------
    print("\n--- Building Candidate 2: Standard ONNX QDQ A16W8 ---")
    a16w8_cfg = copy.deepcopy(get_default_config("A16W8"))
    a16w8_cfg.quant_format = QuantFormat.QDQ
    a16w8_cfg.use_external_data_format = False
    if hasattr(a16w8_cfg, "extra_options"):
        a16w8_cfg.extra_options["UseRandomData"] = True
    quantizer_a16w8 = ModelQuantizer(Config(global_quant_config=a16w8_cfg))
    quantizer_a16w8.quantize_model(
        model_input=src_model,
        model_output="htdemucs_drums_xdna2_a16w8_qdq.onnx",
        calibration_data_reader=DemucsCalibrationReader(num_samples=4),
    )
    audit_domains("htdemucs_drums_xdna2_a16w8_qdq.onnx")

    # -----------------------------------------------------------------
    # Candidate 3: Standard ONNX QDQ XINT8 (Power-of-2 Int8 QDQ)
    # -----------------------------------------------------------------
    print("\n--- Building Candidate 3: Standard ONNX QDQ XINT8 ---")
    xint8_cfg = copy.deepcopy(get_default_config("XINT8"))
    xint8_cfg.quant_format = QuantFormat.QDQ
    xint8_cfg.use_external_data_format = False
    quantizer_xint8 = ModelQuantizer(Config(global_quant_config=xint8_cfg))
    quantizer_xint8.quantize_model(
        model_input=src_model,
        model_output="htdemucs_drums_xdna2_xint8_qdq.onnx",
        calibration_data_reader=DemucsCalibrationReader(num_samples=4),
    )
    audit_domains("htdemucs_drums_xdna2_xint8_qdq.onnx")
