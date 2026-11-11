# setup_quark_custom_ops.ps1
# Sets up MSVC x64 build environment + Ninja, then triggers Quark custom ops compilation.

$ErrorActionPreference = "Stop"

$workspaceRoot = $PSScriptRoot
$venvScripts = Join-Path $workspaceRoot ".venv\Scripts"
$pythonExe = Join-Path $venvScripts "python.exe"

# 1. Put .venv\Scripts at the front of PATH for ninja.exe
$env:PATH = "$venvScripts;$env:PATH"

# 2. Locate vcvars64.bat (check VS 18 first, fallback to VS 2022 if needed)
$vcvarsCandidates = @(
    "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)

$vcvars = $null
foreach ($cand in $vcvarsCandidates) {
    if (Test-Path $cand) {
        $vcvars = $cand
        break
    }
}

if (-not $vcvars) {
    Write-Error "Could not find vcvars64.bat. Please ensure Visual Studio C++ build tools are installed."
}

Write-Host "Configuring MSVC environment from: $vcvars"
cmd.exe /c "call `"$vcvars`" && set" | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
        Set-Item -Path "env:\$($matches[1])" -Value $matches[2]
    }
}

Write-Host "Verifying compiler and build tools:"
$clPath = (Get-Command cl.exe -ErrorAction SilentlyContinue).Source
$ninjaPath = (Get-Command ninja.exe -ErrorAction SilentlyContinue).Source
Write-Host "  cl.exe:    $clPath"
Write-Host "  ninja.exe: $ninjaPath"

if (-not $clPath -or -not $ninjaPath) {
    Write-Error "cl.exe or ninja.exe not found in PATH after environment setup!"
}

Write-Host "Compiling / verifying Quark custom ops library..."
& $pythonExe -c @'
import os
import shutil
import glob
from quark.common.torch_cpp_build_specs import ONNX_CUSTOM_OPS_JIT_BASENAME
from torch.utils.cpp_extension import _get_build_directory

# Trigger JIT compile if not yet built
try:
    import quark.onnx
except Exception as e:
    print(f"Note during import: {e}")

jit_dir = _get_build_directory(ONNX_CUSTOM_OPS_JIT_BASENAME, verbose=False)
pyd_path = os.path.join(jit_dir, f"{ONNX_CUSTOM_OPS_JIT_BASENAME}.pyd")
dll_path = os.path.join(jit_dir, f"{ONNX_CUSTOM_OPS_JIT_BASENAME}.dll")

# If PyTorch built .pyd on Windows, ensure .dll alias exists for Quark custom-op loader
if os.path.exists(pyd_path) and not os.path.exists(dll_path):
    shutil.copy2(pyd_path, dll_path)
    print(f"[OK] Copied {pyd_path} -> {dll_path}")

from quark.onnx.operators.custom_ops import get_library_path
lib = get_library_path()
print(f"[SUCCESS] Quark custom ops library loaded: {lib}")
'@

