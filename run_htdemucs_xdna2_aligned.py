import math
import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from demucs.spec import spectro, ispectro


class HTDemucsXDNA2AlignedRunner:
    def __init__(self, onnx_path: str, vaip_config_path: str):
        # Initialize ONNX Runtime with AMD Vitis AI Execution Provider (XDNA 2 NPU)
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            provider_options=[
                {
                    "config_file": vaip_config_path,
                    "cacheDir": "./xdna2_cache",
                    "cacheKey": "htdemucs_drums_aligned_bfp16",
                },
                {},
            ],
        )
        self.nfft = 4096
        self.hop_length = 1024
        self.segment_len = 344064  # 336 * 1024 aligned samples @ 44.1 kHz

    def _cpu_preproc(self, mix: torch.Tensor):
        """Runs fast AVX-512 FFT & segment normalization on Zen 5 CPU (< 2.5 ms)."""
        B, C, L = mix.shape
        hl = self.hop_length
        le = int(math.ceil(L / hl))
        pad = (hl // 2) * 3  # 1536

        # Pad and compute complex STFT on CPU
        x_pad = F.pad(mix, (pad, pad + le * hl - L), mode="reflect")
        z = spectro(x_pad, self.nfft, hl)[..., :-1, :]  # Drop Nyquist -> 2048 bins
        z = z[..., 2 : 2 + le]                          # Slice to 336 frames

        # Unpack Complex-as-Channels (cac): (1, 4, 2048, 336)
        x_spec = torch.view_as_real(z).permute(0, 1, 4, 2, 3).reshape(B, C * 2, 2048, le)

        # Normalize frequency & time branches on CPU
        mean_s = x_spec.mean(dim=(1, 2, 3), keepdim=True)
        std_s = x_spec.std(dim=(1, 2, 3), keepdim=True)
        norm_spec = (x_spec - mean_s) / (1e-5 + std_s)

        mean_t = mix.mean(dim=(1, 2), keepdim=True)
        std_t = mix.std(dim=(1, 2), keepdim=True)
        norm_time = ((mix - mean_t) / (1e-5 + std_t)).unsqueeze(2)  # 4D (1, 2, 1, 344064)

        return norm_spec.numpy(), norm_time.numpy(), (mean_s, std_s, mean_t, std_t)

    def _cpu_postproc(self, out_spec: np.ndarray, out_time: np.ndarray, stats, length: int) -> torch.Tensor:
        """Denormalizes NPU outputs and runs inverse FFT on Zen 5 CPU."""
        mean_s, std_s, mean_t, std_t = stats
        hl = self.hop_length
        pad = (hl // 2) * 3

        # Denormalize frequency branch & pack back to complex64
        x_s = torch.from_numpy(out_spec) * std_s + mean_s  # (1, 4, 2048, 336)
        z_out = x_s.view(1, 2, 2, 2048, 336).permute(0, 1, 3, 4, 2).contiguous()
        z_complex = torch.view_as_complex(z_out)

        # Pad Nyquist bin and boundary frames, then run iSTFT
        z_padded = F.pad(z_complex, (0, 0, 0, 1))
        z_padded = F.pad(z_padded, (2, 2))
        padded_len = hl * int(math.ceil(length / hl)) + 2 * pad
        wav_spec = ispectro(z_padded, hl, length=padded_len)[..., pad : pad + length]

        # Denormalize time branch and combine with frequency branch
        wav_time = torch.from_numpy(out_time).squeeze(2) * std_t + mean_t
        return (wav_spec + wav_time)[..., :length]

    def extract_drums(self, audio: torch.Tensor, overlap: float = 0.25) -> torch.Tensor:
        """Full-song overlap-add runner dispatching aligned windows to the XDNA 2 NPU."""
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        _, C, total_length = audio.shape
        stride = int((1.0 - overlap) * self.segment_len)

        # Global track normalization
        ref = audio.mean(dim=1)
        ref_mean = ref.mean(dim=-1, keepdim=True).unsqueeze(1)
        ref_std = ref.std(dim=-1, keepdim=True).unsqueeze(1) + 1e-8
        norm_audio = (audio - ref_mean) / ref_std

        num_chunks = 1 if total_length <= self.segment_len else math.ceil((total_length - self.segment_len) / stride) + 1
        pad_len = (num_chunks - 1) * stride + self.segment_len - total_length
        padded_audio = F.pad(norm_audio, (0, pad_len))

        # Triangular overlap-add window
        half_seg = self.segment_len // 2
        ramp = torch.arange(1, half_seg + 1, dtype=torch.float32)
        weight = torch.cat([ramp, torch.flip(ramp, dims=[0])]).view(1, 1, self.segment_len)
        weight = weight / weight.max()

        out_drums = torch.zeros_like(padded_audio)
        sum_weight = torch.zeros((1, 1, padded_audio.shape[-1]), dtype=torch.float32)

        for chunk_idx in range(num_chunks):
            offset = chunk_idx * stride
            chunk = padded_audio[:, :, offset : offset + self.segment_len]

            # 1. CPU Pre-processing
            norm_spec, norm_time, stats = self._cpu_preproc(chunk)

            # 2. XDNA 2 NPU Inference
            out_spec, out_time = self.session.run(
                None, {"x_spec": norm_spec, "x_time": norm_time}
            )

            # 3. CPU Post-processing & Overlap-Add
            pred_chunk = self._cpu_postproc(out_spec, out_time, stats, self.segment_len)
            out_drums[:, :, offset : offset + self.segment_len] += pred_chunk * weight
            sum_weight[:, :, offset : offset + self.segment_len] += weight

        drums = out_drums / sum_weight.clamp(min=1e-8)
        return (drums[..., :total_length] * ref_std + ref_mean).squeeze(0)
