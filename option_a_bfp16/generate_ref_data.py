import os
import sys

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from demucs.pretrained import get_model
from option_a_bfp16.export_candidate_a_pure_aie import HTDemucsCandidateAPureAIECore

def main():
    print("=" * 76)
    print("Generating PyTorch FP32 Reference Data for Drums Specialist Parity Check")
    print("=" * 76)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Loading stock f7e0c4bc specialist model...")
    stock_model = get_model("f7e0c4bc")
    if hasattr(stock_model, "models"):
        stock_model = stock_model.models[0]
    stock_model.eval()
    
    print("Building Candidate A Pure AIE Core...")
    ref_core = HTDemucsCandidateAPureAIECore(stock_model).eval()
    
    # 4-second audio chunk at 44.1kHz: STFT [1, 4, 2048, 336], Waveform [1, 2, 1, 344064]
    x_spec_t = torch.randn(1, 4, 2048, 336, dtype=torch.float32)
    x_time_t = torch.randn(1, 2, 1, 344064, dtype=torch.float32)
    
    print("Computing FP32 reference forward pass...")
    with torch.no_grad():
        ref_spec_t, ref_time_t = ref_core(x_spec_t, x_time_t)
        
    out_file = r"option_a_bfp16\parity_data.npz"
    np.savez(
        out_file,
        x_spec=x_spec_t.numpy(),
        x_time=x_time_t.numpy(),
        ref_spec=ref_spec_t.numpy(),
        ref_time=ref_time_t.numpy()
    )
    print(f"Saved reference data to {out_file} ({os.path.getsize(out_file)/(1024*1024):.2f} MB)")
    print("=" * 76)

if __name__ == "__main__":
    main()
