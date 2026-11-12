"""
ANE-friendly HTDemucs single-source core (drums, vocals, bass or other).

Starts from the XDNA2 "pure AIE" rewrite (option_a_bfp16/export_candidate_a_pure_aie.py),
which already removes most of what the Apple Neural Engine dislikes:
  - dilated convs           -> dense kernels with the holes filled with zeros
  - stride-4 K=8 convs      -> unfold + 1x1 conv
  - ConvTranspose           -> polyphase 1x1 conv
  - Instance/GroupNorm      -> explicit normalisation
  - 1D tensors              -> 4D (B, C, H, W)

On top of that, for the ANE:
  1. Static shapes. forward() does no shape arithmetic, so torch.export / coremltools
     produce a fully static graph (no einops, no `H // 4` int casts).
  2. Row-folded time branch. The ANE rejects any tensor dimension > 16384, and the
     time branch runs at 344064 / 86016 / 21504 samples. [1, C, 1, L] is folded into
     [1, C, R, L/R] (a free reshape) and every conv along W takes a "halo" from the
     neighbouring rows, so the maths is identical to the unfolded conv.
  3. Transformer layers rewritten as plain rank-3 matmuls with pre-baked positional
     embeddings and the softmax scale folded into the q projection.
  4. Activations written as `x / (1 + exp(-z))` instead of sigmoid/GELU ops. When the
     ANE compiler fuses sigmoid/GELU into the preceding conv it uses a low-precision
     approximation (~45-52 dB per op vs ~66 dB for the exp form), which compounded to
     ~10 dB of extra error over the whole network.
"""
import copy
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from demucs.pretrained import get_model  # noqa: E402
from demucs.transformer import CrossTransformerEncoder, create_2d_sin_embedding, create_sin_embedding  # noqa: E402

from option_a_bfp16.export_candidate_a_pure_aie import (  # noqa: E402
    DConvGroupNorm4D,
    DecomposedGroupNorm1,
    HTDemucsCandidateAPureAIECore,
    PolyphaseConvTrFreq,
    PolyphaseConvTrTime,
    UnfoldedConvFreq,
    UnfoldedConvTime,
)

SEGMENT_LEN = 344064  # 336 * 1024 samples (7.8 s @ 44.1 kHz)
N_FRAMES = 336  # STFT frames per segment; the freq branch never strides along time
FR_BOTTOM = 8  # 2048 freq bins / 4^4
T_BOTTOM = SEGMENT_LEN // 4**4  # 1344 time-branch steps at the bottleneck
ANE_MAX_DIM = 16384


def rows_for(length: int) -> int:
    """Smallest power-of-two row count that keeps the row width <= ANE_MAX_DIM."""
    r = 1
    while length // r > ANE_MAX_DIM or length % r:
        r *= 2
    return r


TIME_ROWS = rows_for(SEGMENT_LEN)  # 32 rows x 10752 samples
SPEC_SHAPE = (1, 4, 2048, N_FRAMES)
TIME_SHAPE = (1, 2, TIME_ROWS, SEGMENT_LEN // TIME_ROWS)


# ---------------------------------------------------------------------------
# ANE-precise activations
# ---------------------------------------------------------------------------
def ane_gelu(x):
    # tanh-approximation GELU, written as x * sigmoid(2z) = x / (1 + exp(-2z)).
    # Differs from erf-GELU by ~61 dB SNR at the model output, far below fp16 noise.
    return x / (1.0 + torch.exp(-1.5957691216057308 * (x + 0.044715 * x * x * x)))


def ane_glu(x, dim=1):
    a, b = x.chunk(2, dim=dim)
    return a / (1.0 + torch.exp(b * -1.0))  # `* -1` rather than unary neg: ORT's CoreML EP has no Neg


class GELUANE(nn.Module):
    def forward(self, x):
        return ane_gelu(x)


class GLUANE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return ane_glu(x, self.dim)


# ---------------------------------------------------------------------------
# Normalisation (lowered to the native layer_norm op, which the ANE runs directly)
# ---------------------------------------------------------------------------
def _norm(x, dims, eps):
    """Zero-mean / unit-variance over `dims`, no affine."""
    if sorted(dims) == list(range(1, x.dim())):
        return F.layer_norm(x, x.shape[1:], eps=eps)
    assert x.dim() == 4 and sorted(dims) == [1, 3]
    x = x.transpose(1, 2)
    return F.layer_norm(x, x.shape[2:], eps=eps).transpose(1, 2)


def _gn_freq_forward(self, x):
    # DConv on the freq branch treats H (frequency) as batch: reduce over C and W only.
    return _norm(x, (1, 3), self.eps) * self.weight + self.bias


def _gn_time_forward(self, x):
    # Time-branch GroupNorm(1) on a row-folded tensor: reduce over C, rows and W.
    return _norm(x, (1, 2, 3), self.eps) * self.weight + self.bias


def _gn_tokens_forward(self, x):
    # Transformer norm_out: GroupNorm(1) over [1, T, C] = normalise over T and C.
    return _norm(x, (1, 2), self.eps) * self.weight + self.bias


class GroupNorm1ANE(nn.Module):
    """nn.GroupNorm(1, C) for 4D input."""

    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        assert gn.num_groups == 1
        self.eps = gn.eps
        self.weight = nn.Parameter(gn.weight.data.view(1, -1, 1, 1).clone())
        self.bias = nn.Parameter(gn.bias.data.view(1, -1, 1, 1).clone())

    def forward(self, x):
        return _norm(x, (1, 2, 3), self.eps) * self.weight + self.bias


# ---------------------------------------------------------------------------
# Frequency branch: static-shape versions of the pure-AIE blocks
# ---------------------------------------------------------------------------
def _unfold_freq_forward(self, x):
    # x: [1, C, H, W] -> [1, 8C, H/4, W]   (Conv K=[8,1] S=[4,1] P=[2,0])
    c, w = self.conv1x1.in_channels // 8, N_FRAMES
    x = F.pad(x, (0, 0, 2, 2)).reshape(1, c, -1, 4 * w)  # [1, C, N+1, 4W]
    a, b = x[:, :, :-1, :], x[:, :, 1:, :]
    taps = [a[..., k * w:(k + 1) * w] for k in range(4)] + [b[..., k * w:(k + 1) * w] for k in range(4)]
    return self.conv1x1(torch.cat(taps, dim=1))


def _poly_freq_forward(self, x):
    # x: [1, C, H, W] -> [1, out, 4H, W]   (ConvTranspose K=[8,1] S=[4,1] + trim)
    x_prev = F.pad(x, (0, 0, 1, 0))[:, :, :-1, :]
    x_next = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    p = self.conv1x1(torch.cat([x_prev, x, x_next], dim=1))  # [1, 4*out, H, W]
    out, w = self.conv1x1.out_channels // 4, N_FRAMES
    # Interleave the 4 phases along H: concat along W, then fold back into H.
    phases = [p[:, k * out:(k + 1) * out] for k in range(4)]
    return torch.cat(phases, dim=3).reshape(1, out, -1, w)


def _freq_enc_forward(self, x):
    x = self.norm1(self.conv(x))
    if self.empty:
        return x
    x = ane_gelu(x)
    if self.dconv is not None:
        x = self.dconv(x)
    if self.rewrite is not None:
        x = ane_glu(self.norm2(self.rewrite(x)), dim=1)
    return x


def _freq_dec_forward(self, x, skip):
    if not self.empty:
        y = x + skip
        if self.rewrite is not None:
            y = ane_glu(self.norm1(self.rewrite(y)), dim=1)
        if self.dconv is not None:
            y = self.dconv(y)
    else:
        y = x
    z = self.norm2(self.conv_tr(y))
    if not self.last:
        z = ane_gelu(z)
    return z


# ---------------------------------------------------------------------------
# Time branch: row-folded [1, C, R, W] tensors with halo exchange between rows
# ---------------------------------------------------------------------------
def _halo(x, p):
    """Prepend the previous row's last p columns and append the next row's first p."""
    tail_prev = F.pad(x[..., -p:], (0, 0, 1, 0))[:, :, :-1, :]
    head_next = F.pad(x[..., :p], (0, 0, 0, 1))[:, :, 1:, :]
    return torch.cat([tail_prev, x, head_next], dim=3)


class HaloConv(nn.Module):
    """Conv2d with kernel (1, k) / padding (0, p), exact on row-folded input."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        assert conv.kernel_size[0] == 1 and conv.stride == (1, 1) and conv.padding[0] == 0
        self.p = conv.padding[1]
        self.conv = copy.deepcopy(conv)
        self.conv.padding = (0, 0)

    def forward(self, x):
        return self.conv(_halo(x, self.p) if self.p else x)


def _unfold_time_folded(self, x):
    # x: [1, C, R, W] -> [1, 8C, R, W/4]   (Conv K=8 S=4 P=2)
    c, r = self.conv1x1.in_channels // 8, self.rows
    x = _halo(x, 2).reshape(1, c * r, -1, 4)  # [1, CR, N+1, 4]
    x = torch.cat([x[:, :, :-1, :], x[:, :, 1:, :]], dim=3)  # [1, CR, N, 8] = taps 0..7
    x = x.reshape(c, r, -1, 8).permute(3, 0, 1, 2).reshape(1, 8 * c, r, -1)  # tap-major channels
    return self.conv1x1(x)


def _poly_time_folded(self, x):
    # x: [1, C, R, N] -> [1, out, 4R, N]   (ConvTranspose K=8 S=4 + trim)
    # The output keeps row width N (4x more rows) so W never exceeds the ANE limit.
    h = _halo(x, 1)
    x_prev, x_next = h[..., :-2], h[..., 2:]
    p = self.conv1x1(torch.cat([x_prev, x, x_next], dim=1))  # [1, 4*out, R, N]
    out, r = self.conv1x1.out_channels // 4, self.rows
    p = p.reshape(4, out, r, -1).permute(1, 2, 3, 0)  # [out, R, N, 4]
    return p.reshape(1, out, 4 * r, -1)


def _fold_time_layer(layer: nn.Module, rows: int):
    for m in layer.modules():
        if isinstance(m, UnfoldedConvTime):
            m.rows = rows
            m.forward = types.MethodType(_unfold_time_folded, m)
        elif isinstance(m, PolyphaseConvTrTime):
            m.rows = rows
            m.forward = types.MethodType(_poly_time_folded, m)
        elif isinstance(m, DConvGroupNorm4D):
            m.forward = types.MethodType(_gn_time_forward, m)
    _wrap_halo_convs(layer)


def _wrap_halo_convs(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.kernel_size[0] == 1 and child.kernel_size[1] > 1:
            setattr(module, name, HaloConv(child))
        elif not isinstance(child, (UnfoldedConvTime, PolyphaseConvTrTime, HaloConv)):
            _wrap_halo_convs(child)


def _time_enc_forward(self, xt):
    xt = self.norm1(self.conv(xt))
    if self.empty:
        return xt
    xt = ane_gelu(xt)
    if self.dconv is not None:
        xt = self.dconv(xt)
    if self.rewrite is not None:
        xt = ane_glu(self.norm2(self.rewrite(xt)), dim=1)
    return xt


def _time_dec_forward(self, xt, skip):
    if not self.empty:
        y = xt + skip
        if self.rewrite is not None:
            y = ane_glu(self.norm1(self.rewrite(y)), dim=1)
        if self.dconv is not None:
            y = self.dconv(y)
    else:
        y = xt
    z = self.norm2(self.conv_tr(y))
    if not self.last:
        z = ane_gelu(z)
    return z


# ---------------------------------------------------------------------------
# Cross transformer
# ---------------------------------------------------------------------------
def _attention(mha: nn.MultiheadAttention, query, kv):
    """Batch-1 multi-head attention on [1, T, E] tensors using only rank-3 matmuls."""
    e, h = mha.embed_dim, mha.num_heads
    hd = e // h
    w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = mha.in_proj_bias.chunk(3, dim=0)
    scale = 1.0 / math.sqrt(hd)
    # Fold the softmax scale into the q projection (keeps fp16 logits small).
    q = F.linear(query.reshape(-1, e), w_q * scale, b_q * scale)  # [T, E]
    k = F.linear(kv.reshape(-1, e), w_k, b_k)  # [S, E]
    v = F.linear(kv.reshape(-1, e), w_v, b_v)
    q = q.reshape(-1, h, hd).transpose(0, 1)  # [h, T, hd]
    k = k.reshape(-1, h, hd).permute(1, 2, 0)  # [h, hd, S]
    v = v.reshape(-1, h, hd).transpose(0, 1)  # [h, S, hd]
    attn = torch.softmax(torch.matmul(q, k), dim=-1)  # [h, T, S]
    out = torch.matmul(attn, v).transpose(0, 1).reshape(1, -1, e)
    return mha.out_proj(out)


def _ff(layer, x):
    return layer.linear2(layer.activation(layer.linear1(x)))


def _self_layer_forward(self, x):
    assert self.norm_first
    y = self.norm1(x)
    x = x + self.gamma_1(_attention(self.self_attn, y, y))
    x = x + self.gamma_2(_ff(self, self.norm2(x)))
    return self.norm_out(x) if self.norm_out else x


def _cross_layer_forward(self, q, k):
    assert self.norm_first
    x = q + self.gamma_1(_attention(self.cross_attn, self.norm1(q), self.norm2(k)))
    x = x + self.gamma_2(_ff(self, self.norm3(x)))
    return self.norm_out(x) if self.norm_out else x


def _cross_transformer_forward(self, x, xt):
    # x: [1, C, Fr, T1] -> [1, T1*Fr, C]; xt: [1, C, T2] -> [1, T2, C]
    c = self.norm_in.normalized_shape[0]
    x = x.permute(0, 3, 2, 1).reshape(1, N_FRAMES * FR_BOTTOM, c)
    x = self.norm_in(x) + self.baked_pos_2d
    xt = self.norm_in_t(xt.permute(0, 2, 1)) + self.baked_pos_1d
    for idx in range(self.num_layers):
        if idx % 2 == self.classic_parity:
            x = self.layers[idx](x)
            xt = self.layers_t[idx](xt)
        else:
            old_x = x
            x = self.layers[idx](x, xt)
            xt = self.layers_t[idx](xt, old_x)
    x = x.reshape(1, N_FRAMES, FR_BOTTOM, c).permute(0, 3, 2, 1)
    return x, xt.permute(0, 2, 1)


def _patch_cross_transformer(ct: CrossTransformerEncoder):
    """Pre-bake positional embeddings and swap in the static forwards."""
    c = ct.norm_in.normalized_shape[0]
    with torch.no_grad():
        pe2d = create_2d_sin_embedding(c, FR_BOTTOM, N_FRAMES, "cpu", ct.max_period)  # [1, C, Fr, T1]
        pe2d = pe2d.permute(0, 3, 2, 1).reshape(1, N_FRAMES * FR_BOTTOM, c)
        pe1d = create_sin_embedding(T_BOTTOM, c, shift=0, max_period=ct.max_period).permute(1, 0, 2)  # [1, T2, C]
    ct.register_buffer("baked_pos_2d", (ct.weight_pos_embed * pe2d).clone())
    ct.register_buffer("baked_pos_1d", (ct.weight_pos_embed * pe1d).clone())
    ct.forward = types.MethodType(_cross_transformer_forward, ct)
    for layer in list(ct.layers) + list(ct.layers_t):
        fwd = _cross_layer_forward if hasattr(layer, "cross_attn") else _self_layer_forward
        layer.forward = types.MethodType(fwd, layer)
        for m in layer.modules():
            if isinstance(m, DecomposedGroupNorm1):
                m.forward = types.MethodType(_gn_tokens_forward, m)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
def _replace_modules(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.GroupNorm) and child.num_groups == 1:
            setattr(module, name, GroupNorm1ANE(child))
        elif isinstance(child, nn.GELU):
            setattr(module, name, GELUANE())
        elif isinstance(child, nn.GLU):
            setattr(module, name, GLUANE(child.dim))
        else:
            _replace_modules(child)
    for m in module.modules():
        if getattr(m, "activation", None) is F.gelu:
            m.activation = ane_gelu


class HTDemucsSourceANE(nn.Module):
    """
    Inputs:  x_spec   [1, 4, 2048, 336]  normalised CaC spectrogram
             x_time   [1, 2, 32, 10752]  normalised waveform, row-folded (= reshape of [1, 2, 344064])
    Outputs: out_spec [1, 4, 2048, 336]  CaC spectrogram of the selected source (drums, vocals, ...)
             out_time [1, 2, 32, 10752]  waveform of the selected source, row-folded
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        core = HTDemucsCandidateAPureAIECore(base_model.eval()).eval()
        for m in core.modules():
            if isinstance(m, UnfoldedConvFreq):
                m.forward = types.MethodType(_unfold_freq_forward, m)
            elif isinstance(m, PolyphaseConvTrFreq):
                m.forward = types.MethodType(_poly_freq_forward, m)
            elif isinstance(m, DConvGroupNorm4D):
                m.forward = types.MethodType(_gn_freq_forward, m)
            elif isinstance(m, CrossTransformerEncoder):
                _patch_cross_transformer(m)
        for layer in core.encoder:
            layer.forward = types.MethodType(_freq_enc_forward, layer)
        for layer in core.decoder:
            layer.forward = types.MethodType(_freq_dec_forward, layer)

        # Time-branch length at each depth: 344064, 86016, 21504, 5376, 1344
        self.t_lens = [SEGMENT_LEN // 4**i for i in range(len(core.tencoder) + 1)]
        depth = len(core.tencoder)
        for i, layer in enumerate(core.tencoder):
            _fold_time_layer(layer, rows_for(self.t_lens[i]))
            layer.forward = types.MethodType(_time_enc_forward, layer)
        for i, layer in enumerate(core.tdecoder):
            _fold_time_layer(layer, rows_for(self.t_lens[depth - i]))
            layer.forward = types.MethodType(_time_dec_forward, layer)

        _replace_modules(core)
        self.core = core
        self.debug_taps = False  # True: also return every intermediate encoder/bottleneck/decoder tensor

    def _fold(self, xt, length):
        r = rows_for(length)
        return xt.reshape(1, -1, r, length // r)

    def forward(self, x_spec, x_time):
        c = self.core
        x, xt = x_spec, x_time
        saved_s, saved_t = [], []
        for i in range(len(c.encoder)):
            xt = self._fold(c.tencoder[i](self._fold(xt, self.t_lens[i])), self.t_lens[i + 1])
            saved_t.append(xt)
            x = c.encoder[i](x)
            if i == 0 and c.baked_freq_emb is not None:
                x = x + c.baked_freq_emb
            saved_s.append(x)

        # Bottleneck: x [1, 384, 8, 336], xt [1, 384, 1, 1344]
        n_bot = FR_BOTTOM * N_FRAMES
        if c.bottom_channels:
            x = c.channel_upsampler(x.reshape(1, -1, 1, n_bot)).reshape(1, -1, FR_BOTTOM, N_FRAMES)
            xt = c.channel_upsampler_t(xt)
        x, xt3 = c.crosstransformer(x, xt.reshape(1, -1, T_BOTTOM))
        xt = xt3.reshape(1, -1, 1, T_BOTTOM)
        if c.bottom_channels:
            x = c.channel_downsampler(x.reshape(1, -1, 1, n_bot)).reshape(1, -1, FR_BOTTOM, N_FRAMES)
            xt = c.channel_downsampler_t(xt)

        taps = list(saved_s) + list(saved_t) + [x, xt] if self.debug_taps else []
        for i in range(len(c.decoder)):
            x = c.decoder[i](x, saved_s.pop(-1))
            depth = len(c.decoder) - i
            xt = c.tdecoder[i](self._fold(xt, self.t_lens[depth]), saved_t.pop(-1))
            if self.debug_taps and i < len(c.decoder) - 1:
                taps += [x, xt]
        return (x, self._fold(xt, SEGMENT_LEN), *taps)


TAP_NAMES = (
    [f"enc{i}" for i in range(4)] + [f"tenc{i}" for i in range(4)] + ["bottleneck", "bottleneck_t"]
    + [n for i in range(3) for n in (f"dec{i}", f"tdec{i}")]
)


def load_base_model() -> nn.Module:
    base = get_model("f7e0c4bc")
    base = base.models[0] if hasattr(base, "models") else base
    return base.eval()


SOURCES = ("drums", "bass", "other", "vocals")


def select_source(base: nn.Module, source: str) -> nn.Module:
    """
    Copy of `base` whose final decoder layers only produce `source`. The output channels
    are source-major (freq: 4 CaC channels per source, time: 2 per source), so this is a
    plain slice. The pure-AIE core then keeps the first source, which is now `source`.
    """
    s = list(base.sources).index(source)
    base = copy.deepcopy(base)
    c = base.audio_channels
    for layer, n in ((base.decoder[-1], 2 * c), (base.tdecoder[-1], c)):
        tr = layer.conv_tr
        tr.weight = nn.Parameter(tr.weight[:, s * n:(s + 1) * n].clone())
        if tr.bias is not None:
            tr.bias = nn.Parameter(tr.bias[s * n:(s + 1) * n].clone())
        tr.out_channels = n
        layer.chout = n
    return base


def build_ane_model(base: nn.Module = None, source: str = "drums") -> tuple[HTDemucsSourceANE, nn.Module]:
    """Returns (ANE model for `source`, single-source base model it was built from)."""
    base = select_source(load_base_model() if base is None else base, source)
    return HTDemucsSourceANE(copy.deepcopy(base)).eval(), base
