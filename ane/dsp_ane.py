"""
STFT / iSTFT for the ANE.

The ANE has no FFT, but at HTDemucs' sizes the DFT is cheap as a matmul. With 336 frames
x 4096 samples x 4096 outputs per channel, that is ~11 GMAC, a few percent of the model.
So the transforms are written as framing (reshapes) plus one dense matmul. That maps onto
the same conv/matmul hardware as the network.

What stays on the CPU, and why:
  - Reflect padding: data-dependent edge copies on a 344064-sample vector. The ANE
    rejects W > 16384, and the copies are ~1 us of numpy anyway.
  - The final crop: the output starts 3584 samples into the overlap-add buffer, which is
    not a multiple of the 1024-sample block, so it would need an unaligned slice of a
    344064-wide tensor. On the CPU it is a free view.
  - Chunk overlap-add across chunks: it spans model calls.

Layout: a padded signal of 343 * 1024 samples is passed as [1, 2, 343, 1024] blocks. Frame
f is blocks f..f+3, so framing is 4 slices plus a concat, and nothing exceeds 16384.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NFFT, HOP = 4096, 1024
N_FRAMES = 336
FIRST_FRAME = 2  # cpu_preproc_numpy keeps frames 2 .. 2+336 of the centred STFT
PAD = (HOP // 2) * 3  # 1536, demucs' extra reflect pad on each side
N_BINS = NFFT // 2  # 2048 (the Nyquist bin is dropped)
SEGMENT_LEN = N_FRAMES * HOP  # 344064
PADDED_LEN = SEGMENT_LEN + 2 * PAD + NFFT  # 351232 = 343 blocks
N_BLOCKS = PADDED_LEN // HOP  # 343
OLA_BLOCKS = N_FRAMES + 2 * FIRST_FRAME + 3  # 343: iSTFT output blocks (340 frames, each 4 blocks wide)

_WINDOW = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(NFFT) / NFFT))


def reflect_pad_numpy(mix: np.ndarray) -> np.ndarray:
    """[1, 2, 344064] -> [1, 2, 343, 1024]: the two reflect pads of cpu_preproc_numpy, as blocks."""
    x = np.pad(mix, ((0, 0), (0, 0), (PAD, PAD)), mode="reflect")
    x = np.pad(x, ((0, 0), (0, 0), (NFFT // 2, NFFT // 2)), mode="reflect")
    return np.ascontiguousarray(x, dtype=np.float32).reshape(1, 2, N_BLOCKS, HOP)


def _dft_matrix() -> torch.Tensor:
    """[4096, 2*2048]: windowed real DFT scaled by 1/64; columns [re_0..re_2047, im_0..im_2047]."""
    n = np.arange(NFFT)[:, None]
    k = np.arange(N_BINS)[None, :]
    ang = 2.0 * np.pi * n * k / NFFT
    w = _WINDOW[:, None] / 64.0
    return torch.from_numpy(np.concatenate([w * np.cos(ang), -w * np.sin(ang)], axis=1).astype(np.float32))


def _idft_matrix() -> torch.Tensor:
    """[2*2048, 4096]: irfft(n=4096) * 64 * window for bins 0..2047 (Nyquist is zero)."""
    k = np.arange(N_BINS)[:, None]
    n = np.arange(NFFT)[None, :]
    ang = 2.0 * np.pi * k * n / NFFT
    c = np.where(k == 0, 1.0, 2.0) / NFFT * 64.0  # DC counted once, other bins twice (Hermitian)
    re = c * np.cos(ang)
    im = -c * np.sin(ang)
    im[0] = 0.0  # irfft ignores the imaginary part of DC
    return torch.from_numpy((np.concatenate([re, im], axis=0) * _WINDOW[None, :]).astype(np.float32))


def _ola_inv_weight() -> torch.Tensor:
    """1 / (sum of squared windows) per output sample, laid out as [343, 1024] blocks."""
    n_frames = OLA_BLOCKS - 3
    wsum = np.zeros((n_frames - 1) * HOP + NFFT)
    for i in range(n_frames):
        wsum[i * HOP : i * HOP + NFFT] += _WINDOW**2
    inv = np.where(wsum > 1e-8, 1.0 / np.maximum(wsum, 1e-8), 0.0)
    # Near the buffer edges wsum -> 0 and 1/wsum reaches ~1e8. Those samples are cropped
    # away anyway, but a constant that large does not fit in fp16, so coremltools would
    # keep this multiply in fp32 and that forces it onto the CPU. Inside the crop the value is ~0.67.
    inv = np.minimum(inv, 100.0)
    return torch.from_numpy(inv.reshape(OLA_BLOCKS, HOP).astype(np.float32))


class STFTANE(nn.Module):
    """
    blocks [1, 2, 343, 1024] (reflect-padded mix) ->
      x_spec [1, 4, 2048, 336] normalised CaC spectrogram, plus mean_s / std_s [1, 1, 1, 1].
    Matches cpu_preproc_numpy's spectral half.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("dft", _dft_matrix())

    def forward(self, blocks):
        # frames f = blocks f..f+3, only the 336 frames the model uses
        b = blocks[:, :, FIRST_FRAME : FIRST_FRAME + N_FRAMES + 3, :]  # [1, 2, 339, 1024]
        frames = torch.cat([b[:, :, j : j + N_FRAMES, :] for j in range(4)], dim=3)  # [1, 2, 336, 4096]
        z = torch.matmul(frames, self.dft)  # [1, 2, 336, 4096] = [re | im] per channel
        z = z.permute(0, 1, 3, 2)  # [1, 2, 4096, 336]
        re0, im0 = z[:, 0:1, :N_BINS], z[:, 0:1, N_BINS:]
        re1, im1 = z[:, 1:2, :N_BINS], z[:, 1:2, N_BINS:]
        x = torch.cat([re0, im0, re1, im1], dim=1)  # [1, 4, 2048, 336]
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        n = x.numel()
        var = ((x - mean) ** 2).mean(dim=(1, 2, 3), keepdim=True) * (n / (n - 1))  # ddof=1
        std = torch.sqrt(var)
        return (x - mean) / (std + 1e-5), mean, std


class ISTFTANE(nn.Module):
    """
    out_spec [1, 4, 2048, 336] + mean_s / std_s -> overlap-added blocks [1, 2, 343, 1024].
    The caller crops samples [3584, 3584 + 344064) of the flattened blocks (see crop_istft).
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("idft", _idft_matrix())
        self.register_buffer("inv_w", _ola_inv_weight())

    def forward(self, out_spec, mean_s, std_s):
        x = out_spec * std_s + mean_s  # [1, 4, 2048, 336]
        ch = [torch.cat([x[:, 2 * c : 2 * c + 1], x[:, 2 * c + 1 : 2 * c + 2]], dim=2) for c in range(2)]
        z = torch.cat(ch, dim=1).permute(0, 1, 3, 2)  # [1, 2, 336, 4096] = [re | im]
        frames = torch.matmul(z, self.idft)  # [1, 2, 336, 4096], windowed
        # demucs pads 2 empty frames on each side; then frame f covers blocks f..f+3
        frames = F.pad(frames, (0, 0, FIRST_FRAME, FIRST_FRAME))  # [1, 2, 340, 4096]
        out = 0
        for j in range(4):
            part = frames[..., j * HOP : (j + 1) * HOP]  # [1, 2, 340, 1024] -> lands on block f + j
            out = out + F.pad(part, (0, 0, j, 3 - j))
        return out * self.inv_w  # [1, 2, 343, 1024]


class FusedPipelineANE(nn.Module):
    """
    STFT -> separation core -> iSTFT in a single Core ML program, so the spectrogram
    never leaves the ANE.
      in:  blocks [1, 2, 343, 1024] (reflect-padded mix), x_time [1, 2, 32, 10752]
      out: wav_blocks [1, 2, 343, 1024] (spectral branch, crop with crop_istft),
           out_time [1, 2, 32, 10752]
    """

    def __init__(self, core: nn.Module):
        super().__init__()
        self.stft, self.core, self.istft = STFTANE(), core, ISTFTANE()

    def forward(self, blocks, x_time):
        x_spec, mean_s, std_s = self.stft(blocks)
        out_spec, out_time = self.core(x_spec, x_time)[:2]
        return self.istft(out_spec, mean_s, std_s), out_time


def crop_istft(blocks: np.ndarray) -> np.ndarray:
    """[1, 2, 343, 1024] -> [1, 2, 344064] (the unaligned crop stays on the CPU)."""
    flat = blocks.reshape(1, 2, -1)
    start = NFFT // 2 + PAD
    return flat[:, :, start : start + SEGMENT_LEN]


def time_stats_numpy(mix: np.ndarray):
    """Time-branch normalisation from cpu_preproc_numpy (two scalars + a scale; trivially cheap)."""
    mean_t = mix.mean(axis=(1, 2), keepdims=True)
    std_t = mix.std(axis=(1, 2), keepdims=True, ddof=1)
    return ((mix - mean_t) / (1e-5 + std_t))[:, :, np.newaxis, :], mean_t, std_t


assert OLA_BLOCKS == N_BLOCKS and math.isclose(PADDED_LEN / HOP, N_BLOCKS)
