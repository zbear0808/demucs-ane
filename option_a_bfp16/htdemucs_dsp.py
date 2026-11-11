import numpy as np

SAMPLE_RATE = 44100
NFFT = 4096
HOP_LENGTH = 1024
SEGMENT_LEN = 344064  # 336 * 1024 aligned samples (7.8019 seconds)

# Precompute Hann window for STFT/iSTFT
HANN_WINDOW = (0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(NFFT) / NFFT))).astype(np.float32)
SQ_WINDOW = HANN_WINDOW**2


def cpu_preproc_numpy(mix: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple]:
    """
    Bit-exact NumPy HTDemucs CPU preprocessing:
    Complex STFT, cac packing, and standard deviation normalization.
    mix: [1, 2, 344064]
    """
    hl = HOP_LENGTH
    le = 336
    pad = (hl // 2) * 3
    total_pad_len = pad + le * hl - SEGMENT_LEN

    x_pad = np.pad(mix, ((0, 0), (0, 0), (pad, total_pad_len)), mode="reflect")
    n_pad = NFFT // 2
    x_stft_padded = np.pad(x_pad, ((0, 0), (0, 0), (n_pad, n_pad)), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(x_stft_padded, NFFT, axis=-1)[:, :, ::hl, :]

    z = np.fft.rfft(frames * HANN_WINDOW, axis=-1) / 64.0
    z = np.swapaxes(z, 2, 3)[..., :-1, :]
    z = z[..., 2 : 2 + le]

    real = np.real(z)
    imag = np.imag(z)
    x_spec = np.concatenate(
        [
            np.stack([real[:, 0], imag[:, 0]], axis=1),
            np.stack([real[:, 1], imag[:, 1]], axis=1),
        ],
        axis=1,
    ).astype(np.float32)

    mean_s = x_spec.mean(axis=(1, 2, 3), keepdims=True)
    std_s = x_spec.std(axis=(1, 2, 3), keepdims=True, ddof=1)
    norm_spec = (x_spec - mean_s) / (1e-5 + std_s)

    mean_t = mix.mean(axis=(1, 2), keepdims=True)
    std_t = mix.std(axis=(1, 2), keepdims=True, ddof=1)
    norm_time = ((mix - mean_t) / (1e-5 + std_t))[:, :, np.newaxis, :]

    return norm_spec, norm_time, (mean_s, std_s, mean_t, std_t)


def cpu_postproc_numpy(
    out_spec: np.ndarray,
    out_time: np.ndarray,
    stats: tuple,
) -> np.ndarray:
    """
    Bit-exact NumPy HTDemucs CPU postprocessing:
    Inverse STFT synthesis, denormalization, and dual-domain combination.
    """
    mean_s, std_s, mean_t, std_t = stats
    hl = HOP_LENGTH
    pad = (hl // 2) * 3
    padded_len = hl * 336 + 2 * pad

    x_s = out_spec * std_s + mean_s
    ch0 = x_s[:, 0] + 1j * x_s[:, 1]
    ch1 = x_s[:, 2] + 1j * x_s[:, 3]
    z_complex = np.stack([ch0, ch1], axis=1)

    z_padded = np.pad(z_complex, ((0, 0), (0, 0), (0, 1), (2, 2)))
    z_padded_t = np.swapaxes(z_padded, 2, 3)
    inv_frames = np.fft.irfft(z_padded_t, n=NFFT, axis=-1) * 64.0
    windowed = inv_frames * HANN_WINDOW

    n_frames = z_padded_t.shape[2]
    total_len = (n_frames - 1) * hl + NFFT
    out_buf = np.zeros((1, 2, total_len), dtype=np.float32)
    weight_buf = np.zeros(total_len, dtype=np.float32)

    for i in range(n_frames):
        st = i * hl
        out_buf[:, :, st : st + NFFT] += windowed[:, :, i, :]
        weight_buf[st : st + NFFT] += SQ_WINDOW

    mask = weight_buf > 1e-8
    out_buf[:, :, mask] /= weight_buf[mask]
    n_pad = NFFT // 2
    wav_spec = out_buf[:, :, n_pad : n_pad + padded_len][:, :, pad : pad + SEGMENT_LEN]

    wav_time = out_time[:, :, 0, :] * std_t + mean_t
    return (wav_spec + wav_time)[:, :, :SEGMENT_LEN]
