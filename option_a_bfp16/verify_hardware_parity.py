import os
import sys
import torch
import numpy as np
import onnxruntime as ort
from demucs.pretrained import get_model

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

torch_lib = r"C:\Users\zubair\Documents\GitHub\demucs-npu\.venv\Lib\site-packages\torch\lib"
os.add_dll_directory(torch_lib)
os.environ["PATH"] = f"{torch_lib};{os.environ.get('PATH', '')}"

dll = r"C:\Users\zubair\AppData\Local\torch_extensions\torch_extensions\Cache\py312_cpu\quark_custom_ops_stable_abi\quark_custom_ops_stable_abi.dll"
so = ort.SessionOptions()
so.register_custom_ops_library(dll)
so.intra_op_num_threads = 8

cfg = r"C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0-demucs\Lib\site-packages\misc\vaip_config.json"
model_path = r"option_a_bfp16\models\htdemucs_candidate_a_pure_aie_bfp16_clean_cpu.onnx"
cache_dir = r"option_a_bfp16\xdna2_cache_pure_aie"
cache_key = "candidate_a_pure_aie"

print("=" * 76)
print("Verifying NPU Hardware Execution Parity against Stock Demucs (f7e0c4bc)")
print("=" * 76)

# Load stock PyTorch model
print("Loading stock f7e0c4bc specialist model...")
stock_model = get_model("f7e0c4bc")
if hasattr(stock_model, "models"):
    stock_model = stock_model.models[0]
stock_model.eval()

# Load NPU session
print("Loading NPU hardware session from cache...")
sess = ort.InferenceSession(
    model_path,
    sess_options=so,
    providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
    provider_options=[{"config_file": cfg, "cacheDir": cache_dir, "cacheKey": cache_key}, {}]
)
print("Session loaded successfully!")

# Deterministic input audio
torch.manual_seed(42)
np.random.seed(42)

# Input waveform (4 seconds at 44.1kHz = 176400 samples)
# Candidate A core takes dummy_spec [1, 4, 2048, 336] and dummy_time [1, 2, 1, 344064]
x_spec_t = torch.randn(1, 4, 2048, 336, dtype=torch.float32)
x_time_t = torch.randn(1, 2, 1, 344064, dtype=torch.float32)

x_spec_np = x_spec_t.numpy()
x_time_np = x_time_t.numpy()

# Run NPU inference
print("Running NPU inference pass...")
npu_outs = sess.run(None, {"x_spec": x_spec_np, "x_time": x_time_np})
npu_spec = npu_outs[0]
npu_time = npu_outs[1]

# Run PyTorch reference
from option_a_bfp16.export_candidate_a_pure_aie import HTDemucsCandidateAPureAIECore
ref_core = HTDemucsCandidateAPureAIECore(stock_model).eval()

with torch.no_grad():
    ref_spec_t, ref_time_t = ref_core(x_spec_t, x_time_t)
ref_spec = ref_spec_t.numpy()
ref_time = ref_time_t.numpy()

# Calculate parity metrics
spec_diff = np.abs(npu_spec - ref_spec)
time_diff = np.abs(npu_time - ref_time)

spec_max_err = np.max(spec_diff)
spec_mean_err = np.mean(spec_diff)
spec_snr = 10 * np.log10(np.sum(ref_spec**2) / np.sum((ref_spec - npu_spec)**2))

time_max_err = np.max(time_diff)
time_mean_err = np.mean(time_diff)
time_snr = 10 * np.log10(np.sum(ref_time**2) / np.sum((ref_time - npu_time)**2))

print("\n" + "=" * 76)
print("PARITY VERIFICATION METRICS (NPU Hardware BFP16 vs FP32 Reference)")
print("=" * 76)
print(f"Spectrogram Max Absolute Error:  {spec_max_err:.6e}")
print(f"Spectrogram Mean Absolute Error: {spec_mean_err:.6e}")
print(f"Spectrogram SNR:                 {spec_snr:.2f} dB")
print("-" * 76)
print(f"Time-Domain Max Absolute Error:  {time_max_err:.6e}")
print(f"Time-Domain Mean Absolute Error: {time_mean_err:.6e}")
print(f"Time-Domain SNR:                 {time_snr:.2f} dB")
print("=" * 76)

