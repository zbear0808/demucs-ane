import os
import sys
import shutil
import subprocess
from pathlib import Path

def ensure_build_environment():
    """Ensure cl.exe (MSVC) and ninja.exe are in PATH so Quark custom ops compile / load cleanly."""
    if sys.platform == "win32":
        os.environ["PYTHONIOENCODING"] = "utf-8"

    workspace_root = Path(__file__).resolve().parent.parent
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

