import copy
import math
import os
import sys
import types
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from demucs.pretrained import get_model
from demucs.transformer import CrossTransformerEncoder, create_sin_embedding
from demucs.hdemucs import HEncLayer, HDecLayer, DConv
import onnx
from onnx import helper, TensorProto
from onnxsim import simplify

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


# =====================================================================
# 1. 1D -> 2D Convolutions with Dilation Flattening
# =====================================================================
def conv1d_to_conv2d_nondilated(conv1d: nn.Conv1d) -> nn.Conv2d:
    out_ch, in_ch_per_group, k = conv1d.weight.shape
    in_ch = in_ch_per_group * conv1d.groups
    d = conv1d.dilation[0]
    p = conv1d.padding[0]
    s = conv1d.stride[0]

    if d > 1:
        k_eff = 1 + d * (k - 1)
        conv2d = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, k_eff),
            stride=(1, s),
            padding=(0, p),
            dilation=(1, 1),
            groups=conv1d.groups,
            bias=(conv1d.bias is not None),
        )
        with torch.no_grad():
            conv2d.weight.zero_()
            for idx in range(k):
                conv2d.weight[:, :, 0, idx * d] = conv1d.weight[:, :, idx]
            if conv1d.bias is not None:
                conv2d.bias.copy_(conv1d.bias)
        return conv2d
    else:
        conv2d = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, k),
            stride=(1, s),
            padding=(0, p),
            dilation=(1, 1),
            groups=conv1d.groups,
            bias=(conv1d.bias is not None),
        )
        with torch.no_grad():
            conv2d.weight.copy_(conv1d.weight.unsqueeze(2))
            if conv1d.bias is not None:
                conv2d.bias.copy_(conv1d.bias)
        return conv2d


def convtr1d_to_convtr2d(convtr1d: nn.ConvTranspose1d) -> nn.ConvTranspose2d:
    in_ch, out_ch_per_group, k = convtr1d.weight.shape
    out_ch = out_ch_per_group * convtr1d.groups
    convtr2d = nn.ConvTranspose2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=(1, k),
        stride=(1, convtr1d.stride[0]),
        padding=(0, convtr1d.padding[0]),
        output_padding=(0, convtr1d.output_padding[0]),
        groups=convtr1d.groups,
        dilation=(1, convtr1d.dilation[0]),
        bias=(convtr1d.bias is not None),
    )
    with torch.no_grad():
        convtr2d.weight.copy_(convtr1d.weight.unsqueeze(2))
        if convtr1d.bias is not None:
            convtr2d.bias.copy_(convtr1d.bias)
    return convtr2d


# =====================================================================
# 2. Pure AIE Unfolded Downsampling Convolutions (Replaces K=8, S=4, P=2)
# =====================================================================
class UnfoldedConvTime(nn.Module):
    """
    Decomposes Conv1d(K=8, S=4, P=2) into:
    Pad(2, 2) -> Reshape -> Slices -> Concat -> Conv1x1.
    100% native AIE lowering; eliminates asymmetric Conv on NPU.
    """
    def __init__(self, conv2d: nn.Conv2d):
        super().__init__()
        out_ch, in_ch, kh, kw = conv2d.weight.shape
        assert kh == 1 and kw == 8, f"Expected [1, 8] kernel, got {conv2d.weight.shape}"
        self.conv1x1 = nn.Conv2d(8 * in_ch, out_ch, kernel_size=1, bias=(conv2d.bias is not None))
        w_new = torch.cat([conv2d.weight[:, :, 0, k:k+1].squeeze(-1) for k in range(8)], dim=1)
        self.conv1x1.weight.data = w_new.view(out_ch, 8 * in_ch, 1, 1)
        if conv2d.bias is not None:
            self.conv1x1.bias.data = conv2d.bias.data.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, 1, L]
        B, C, _, L = x.shape
        N = L // 4
        x_pad = F.pad(x, (2, 2, 0, 0))  # pad 2 left, 2 right along dim 3
        x_reshaped = x_pad.view(B, C, N + 1, 4)
        c0, c1, c2, c3 = x_reshaped[:, :, :, 0], x_reshaped[:, :, :, 1], x_reshaped[:, :, :, 2], x_reshaped[:, :, :, 3]
        s0, s1, s2, s3 = c0[:, :, :N], c1[:, :, :N], c2[:, :, :N], c3[:, :, :N]
        s4, s5, s6, s7 = c0[:, :, 1:N+1], c1[:, :, 1:N+1], c2[:, :, 1:N+1], c3[:, :, 1:N+1]
        x_unfolded = torch.cat([s0, s1, s2, s3, s4, s5, s6, s7], dim=1).unsqueeze(2)  # [B, 8*C, 1, N]
        return self.conv1x1(x_unfolded)


class UnfoldedConvFreq(nn.Module):
    """
    Decomposes Conv2d(K=[8, 1], S=[4, 1], P=[2, 0]) into:
    Pad(top=2, bot=2) -> Reshape -> Slices -> Concat -> Conv1x1.
    100% native AIE lowering; eliminates asymmetric Conv on NPU.
    """
    def __init__(self, conv2d: nn.Conv2d):
        super().__init__()
        out_ch, in_ch, kh, kw = conv2d.weight.shape
        assert kh == 8 and kw == 1, f"Expected [8, 1] kernel, got {conv2d.weight.shape}"
        self.conv1x1 = nn.Conv2d(8 * in_ch, out_ch, kernel_size=1, bias=(conv2d.bias is not None))
        w_new = torch.cat([conv2d.weight[:, :, k:k+1, 0].squeeze(-1) for k in range(8)], dim=1)
        self.conv1x1.weight.data = w_new.view(out_ch, 8 * in_ch, 1, 1)
        if conv2d.bias is not None:
            self.conv1x1.bias.data = conv2d.bias.data.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, H, W]
        B, C, H, W = x.shape
        N = H // 4
        x_pad = F.pad(x, (0, 0, 2, 2))  # pad 2 top, 2 bottom along dim 2
        x_reshaped = x_pad.view(B, C, N + 1, 4, W)
        c0, c1, c2, c3 = x_reshaped[:, :, :, 0, :], x_reshaped[:, :, :, 1, :], x_reshaped[:, :, :, 2, :], x_reshaped[:, :, :, 3, :]
        s0, s1, s2, s3 = c0[:, :, :N, :], c1[:, :, :N, :], c2[:, :, :N, :], c3[:, :, :N, :]
        s4, s5, s6, s7 = c0[:, :, 1:N+1, :], c1[:, :, 1:N+1, :], c2[:, :, 1:N+1, :], c3[:, :, 1:N+1, :]
        x_unfolded = torch.cat([s0, s1, s2, s3, s4, s5, s6, s7], dim=1)  # [B, 8*C, N, W]
        return self.conv1x1(x_unfolded)


# =====================================================================
# 3. Pure AIE Polyphase Upsampling Convolutions (Replaces ConvTranspose)
# =====================================================================
class PolyphaseConvTrTime(nn.Module):
    """
    Decomposes ConvTranspose1d(K=8, S=4) + trim[..., 2:-2] into:
    Pad -> Neighbor Concat -> Conv1x1(4*out_ch, 3*in_ch) -> Reshape -> Permute -> Reshape.
    100% native AIE lowering; eliminates ConvTranspose and trailing Slice on NPU.
    """
    def __init__(self, convtr2d: nn.ConvTranspose2d):
        super().__init__()
        in_ch, out_ch, kh, kw = convtr2d.weight.shape
        assert kh == 1 and kw == 8, f"Expected [1, 8] kernel, got {convtr2d.weight.shape}"
        self.conv1x1 = nn.Conv2d(3 * in_ch, 4 * out_ch, kernel_size=1, bias=(convtr2d.bias is not None))

        W = convtr2d.weight[:, :, 0, :]  # [in_ch, out_ch, 8]
        zeros = torch.zeros(out_ch, in_ch)
        w_p0 = torch.cat([W[:, :, 6].T, W[:, :, 2].T, zeros], dim=1)
        w_p1 = torch.cat([W[:, :, 7].T, W[:, :, 3].T, zeros], dim=1)
        w_p2 = torch.cat([zeros, W[:, :, 4].T, W[:, :, 0].T], dim=1)
        w_p3 = torch.cat([zeros, W[:, :, 5].T, W[:, :, 1].T], dim=1)
        W_poly = torch.cat([w_p0, w_p1, w_p2, w_p3], dim=0).view(4 * out_ch, 3 * in_ch, 1, 1)
        self.conv1x1.weight.data = W_poly
        if convtr2d.bias is not None:
            b = convtr2d.bias.data
            self.conv1x1.bias.data = torch.cat([b, b, b, b], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, 1, N]
        B, C, _, N = x.shape
        x_prev = F.pad(x, (1, 0, 0, 0))[:, :, :, :N]
        x_curr = x
        x_next = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        x_cat = torch.cat([x_prev, x_curr, x_next], dim=1)  # [B, 3*C, 1, N]
        P = self.conv1x1(x_cat)  # [B, 4*out_ch, 1, N]
        out_ch = P.shape[1] // 4
        P_reshaped = P.view(B, 4, out_ch, 1, N)
        P_perm = P_reshaped.permute(0, 2, 3, 4, 1)  # [B, out_ch, 1, N, 4]
        return P_perm.reshape(B, out_ch, 1, 4 * N)


class PolyphaseConvTrFreq(nn.Module):
    """
    Decomposes ConvTranspose2d(K=[8, 1], S=[4, 1]) + trim[..., 2:-2, :] into:
    Pad -> Neighbor Concat -> Conv1x1(4*out_ch, 3*in_ch) -> Reshape -> Permute -> Reshape.
    100% native AIE lowering; eliminates ConvTranspose and trailing Slice on NPU.
    """
    def __init__(self, convtr2d: nn.ConvTranspose2d):
        super().__init__()
        in_ch, out_ch, kh, kw = convtr2d.weight.shape
        assert kh == 8 and kw == 1, f"Expected [8, 1] kernel, got {convtr2d.weight.shape}"
        self.conv1x1 = nn.Conv2d(3 * in_ch, 4 * out_ch, kernel_size=1, bias=(convtr2d.bias is not None))

        W = convtr2d.weight[:, :, :, 0]  # [in_ch, out_ch, 8]
        zeros = torch.zeros(out_ch, in_ch)
        w_p0 = torch.cat([W[:, :, 6].T, W[:, :, 2].T, zeros], dim=1)
        w_p1 = torch.cat([W[:, :, 7].T, W[:, :, 3].T, zeros], dim=1)
        w_p2 = torch.cat([zeros, W[:, :, 4].T, W[:, :, 0].T], dim=1)
        w_p3 = torch.cat([zeros, W[:, :, 5].T, W[:, :, 1].T], dim=1)
        W_poly = torch.cat([w_p0, w_p1, w_p2, w_p3], dim=0).view(4 * out_ch, 3 * in_ch, 1, 1)
        self.conv1x1.weight.data = W_poly
        if convtr2d.bias is not None:
            b = convtr2d.bias.data
            self.conv1x1.bias.data = torch.cat([b, b, b, b], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, H, W]
        B, C, H, W = x.shape
        x_prev = F.pad(x, (0, 0, 1, 0))[:, :, :H, :]
        x_curr = x
        x_next = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        x_cat = torch.cat([x_prev, x_curr, x_next], dim=1)  # [B, 3*C, H, W]
        P = self.conv1x1(x_cat)  # [B, 4*out_ch, H, W]
        out_ch = P.shape[1] // 4
        P_reshaped = P.view(B, 4, out_ch, H, W)
        P_perm = P_reshaped.permute(0, 2, 3, 1, 4)  # [B, out_ch, H, 4, W]
        return P_perm.reshape(B, out_ch, 4 * H, W)


# =====================================================================
# 4. Decomposed GroupNorm (Replaces InstanceNormalization in Transformer)
# =====================================================================
class DecomposedGroupNorm1(nn.Module):
    """
    Decomposes GroupNorm(num_groups=1) on (B, T, C) into:
    ReduceMean -> Sub -> Mul -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> Add.
    100% native AIE lowering; completely prevents InstanceNormalization generation.
    """
    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        assert gn.num_groups == 1, f"Expected num_groups=1, got {gn.num_groups}"
        self.eps = gn.eps
        self.weight = nn.Parameter(gn.weight.data.clone())
        self.bias = nn.Parameter(gn.bias.data.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, T, C]. Reduce C then T in two ReduceMean ops: a single
        # ReduceMean over ~1M elements loses precision in ORT (and far more in bf16).
        mean = x.mean(dim=2, keepdim=True).mean(dim=1, keepdim=True)
        diff = x - mean
        var = (diff * diff).mean(dim=2, keepdim=True).mean(dim=1, keepdim=True)
        return (diff / torch.sqrt(var + self.eps)) * self.weight + self.bias


class DConvGroupNorm4D(nn.Module):
    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        assert gn.num_groups == 1, f"Expected num_groups=1, got {gn.num_groups}"
        self.eps = gn.eps
        self.weight = nn.Parameter(gn.weight.data.view(1, -1, 1, 1).clone())
        self.bias = nn.Parameter(gn.bias.data.view(1, -1, 1, 1).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reduce W (time) then C in two ReduceMean ops. Time-branch tensors reach
        # 96 x 86016 = 8.3M elements; a single fused ReduceMean over that many
        # values costs ~40 dB of SNR in ONNX Runtime FP32.
        mean = x.mean(dim=3, keepdim=True).mean(dim=1, keepdim=True)
        diff = x - mean
        var = (diff * diff).mean(dim=3, keepdim=True).mean(dim=1, keepdim=True)
        return (diff / torch.sqrt(var + self.eps)) * self.weight + self.bias


class LayerScale4D(nn.Module):
    def __init__(self, ls_module: nn.Module):
        super().__init__()
        self.scale = nn.Parameter(ls_module.scale.data.view(1, -1, 1, 1).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class DConv4DNonDilated(nn.Module):
    def __init__(self, dconv: DConv):
        super().__init__()
        new_layers = nn.ModuleList()
        for seq in dconv.layers:
            new_seq = nn.Sequential()
            for name, layer in seq.named_children():
                if isinstance(layer, nn.Conv1d):
                    new_seq.add_module(name, conv1d_to_conv2d_nondilated(layer))
                elif isinstance(layer, nn.GroupNorm):
                    new_seq.add_module(name, DConvGroupNorm4D(layer))
                elif hasattr(layer, "scale"):
                    new_seq.add_module(name, LayerScale4D(layer))
                else:
                    new_seq.add_module(name, copy.deepcopy(layer))
            new_layers.append(new_seq)
        self.layers = new_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"DConv4D expects 4D (B, C, H, W), got {x.shape}"
        for layer in self.layers:
            x = x + layer(x)
        return x


# =====================================================================
# 5. Dedicated Pure-AIE Encoder & Decoder Layers
# =====================================================================
class FreqEncLayer4DPure(nn.Module):
    def __init__(self, enc: HEncLayer):
        super().__init__()
        assert enc.freq, "FreqEncLayer4DPure requires a frequency-branch HEncLayer"
        self.empty = enc.empty
        self.conv = UnfoldedConvFreq(enc.conv)
        self.norm1 = copy.deepcopy(enc.norm1)
        self.dconv = DConv4DNonDilated(enc.dconv) if (not enc.empty and enc.dconv is not None) else None
        self.rewrite = copy.deepcopy(enc.rewrite) if (not enc.empty and enc.rewrite is not None) else None
        self.norm2 = copy.deepcopy(enc.norm2) if not enc.empty else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(self.conv(x))
        if self.empty:
            return x
        x = F.gelu(x)
        if self.dconv is not None:
            x = self.dconv(x)
        if self.rewrite is not None:
            x = F.glu(self.norm2(self.rewrite(x)), dim=1)
        return x


class FreqDecLayer4DPure(nn.Module):
    def __init__(self, dec: HDecLayer):
        super().__init__()
        assert dec.freq, "FreqDecLayer4DPure requires a frequency-branch HDecLayer"
        self.last = dec.last
        self.empty = dec.empty
        self.rewrite = copy.deepcopy(dec.rewrite) if (not dec.empty and dec.rewrite is not None) else None
        self.norm1 = copy.deepcopy(dec.norm1) if not dec.empty else None
        self.dconv = DConv4DNonDilated(dec.dconv) if (not dec.empty and dec.dconv is not None) else None
        self.conv_tr = PolyphaseConvTrFreq(dec.conv_tr)
        self.norm2 = copy.deepcopy(dec.norm2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if not self.empty:
            y = x + skip
            if self.rewrite is not None:
                y = F.glu(self.norm1(self.rewrite(y)), dim=1)
            if self.dconv is not None:
                y = self.dconv(y)
        else:
            y = x
        z = self.norm2(self.conv_tr(y))
        if not self.last:
            z = F.gelu(z)
        return z


class TimeEncLayer4DPure(nn.Module):
    def __init__(self, enc: HEncLayer):
        super().__init__()
        assert not enc.freq, "TimeEncLayer4DPure requires a time-branch HEncLayer"
        self.stride = enc.stride
        self.empty = enc.empty
        conv2d_tmp = conv1d_to_conv2d_nondilated(enc.conv)
        self.conv = UnfoldedConvTime(conv2d_tmp)
        self.norm1 = copy.deepcopy(enc.norm1)
        self.dconv = DConv4DNonDilated(enc.dconv) if (not enc.empty and enc.dconv is not None) else None
        self.rewrite = conv1d_to_conv2d_nondilated(enc.rewrite) if (not enc.empty and enc.rewrite is not None) else None
        self.norm2 = copy.deepcopy(enc.norm2) if not enc.empty else None

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        assert xt.dim() == 4 and xt.shape[2] == 1, f"Expected (B, C, 1, L), got {xt.shape}"
        xt = self.norm1(self.conv(xt))
        if self.empty:
            return xt
        xt = F.gelu(xt)
        if self.dconv is not None:
            xt = self.dconv(xt)
        if self.rewrite is not None:
            xt = F.glu(self.norm2(self.rewrite(xt)), dim=1)
        return xt


class TimeDecLayer4DPure(nn.Module):
    def __init__(self, dec: HDecLayer):
        super().__init__()
        assert not dec.freq, "TimeDecLayer4DPure requires a time-branch HDecLayer"
        self.last = dec.last
        self.empty = dec.empty
        self.rewrite = conv1d_to_conv2d_nondilated(dec.rewrite) if (not dec.empty and dec.rewrite is not None) else None
        self.norm1 = copy.deepcopy(dec.norm1) if not dec.empty else None
        self.dconv = DConv4DNonDilated(dec.dconv) if (not dec.empty and dec.dconv is not None) else None
        convtr2d_tmp = convtr1d_to_convtr2d(dec.conv_tr)
        self.conv_tr = PolyphaseConvTrTime(convtr2d_tmp)
        self.norm2 = copy.deepcopy(dec.norm2)

    def forward(self, xt: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        assert xt.dim() == 4 and xt.shape[2] == 1, f"Expected (B, C, 1, L), got {xt.shape}"
        if not self.empty:
            y = xt + skip
            if self.rewrite is not None:
                y = F.glu(self.norm1(self.rewrite(y)), dim=1)
            if self.dconv is not None:
                y = self.dconv(y)
        else:
            y = xt
        z = self.norm2(self.conv_tr(y))
        if not self.last:
            z = F.gelu(z)
        return z


# =====================================================================
# 6. Streamlined Batch-First Decomposed MHA for Vitis AI EP
# =====================================================================
def _npu_friendly_mha_forward(
    self_, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    key_padding_mask=None, need_weights: bool = False, attn_mask=None,
    average_attn_weights: bool = True, is_causal: bool = False
):
    if not self_.batch_first:
        query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

    bsz, tgt_len, embed_dim = query.shape
    src_len = key.shape[1]
    num_heads = self_.num_heads
    head_dim = embed_dim // num_heads

    w_q, w_k, w_v = self_.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = (
        self_.in_proj_bias.chunk(3, dim=0) if self_.in_proj_bias is not None else (None, None, None)
    )
    q = F.linear(query, w_q, b_q).view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
    k = F.linear(key, w_k, b_k).view(bsz, src_len, num_heads, head_dim).transpose(1, 2)
    v = F.linear(value, w_v, b_v).view(bsz, src_len, num_heads, head_dim).transpose(1, 2)

    scale = 1.0 / math.sqrt(head_dim)
    attn_weights = torch.matmul(q * scale, k.transpose(-2, -1))
    attn_probs = F.softmax(attn_weights, dim=-1)
    out = torch.matmul(attn_probs, v)

    out = out.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
    out = self_.out_proj(out)

    if not self_.batch_first:
        out = out.transpose(0, 1)
    return out, None


# =====================================================================
# 7. Complete Pure-AIE Candidate A Core Module
# =====================================================================
class HTDemucsCandidateAPureAIECore(nn.Module):
    def __init__(self, base_model: nn.Module = None):
        super().__init__()
        if base_model is None:
            base = get_model("f7e0c4bc")
            if hasattr(base, "models"):
                base = base.models[0]
        else:
            base = copy.deepcopy(base_model)
        base.eval()

        # 1. Prune final output heads to Drums-Only (Index 0)
        C = base.audio_channels  # 2
        freq_final = base.decoder[-1]
        freq_out_ch = C * 2  # 4
        freq_final.conv_tr.weight = nn.Parameter(freq_final.conv_tr.weight[:, :freq_out_ch, ...].clone())
        if freq_final.conv_tr.bias is not None:
            freq_final.conv_tr.bias = nn.Parameter(freq_final.conv_tr.bias[:freq_out_ch].clone())
        freq_final.conv_tr.out_channels = freq_out_ch
        freq_final.chout = freq_out_ch

        time_final = base.tdecoder[-1]
        time_out_ch = C  # 2
        time_final.conv_tr.weight = nn.Parameter(time_final.conv_tr.weight[:, :time_out_ch, ...].clone())
        if time_final.conv_tr.bias is not None:
            time_final.conv_tr.bias = nn.Parameter(time_final.conv_tr.bias[:time_out_ch].clone())
        time_final.conv_tr.out_channels = time_out_ch
        time_final.chout = time_out_ch

        # 2. Build 4D Stacks with Pure AIE modules
        self.encoder = nn.ModuleList([FreqEncLayer4DPure(enc) for enc in base.encoder])
        self.decoder = nn.ModuleList([FreqDecLayer4DPure(dec) for dec in base.decoder])
        self.tencoder = nn.ModuleList([TimeEncLayer4DPure(enc) for enc in base.tencoder])
        self.tdecoder = nn.ModuleList([TimeDecLayer4DPure(dec) for dec in base.tdecoder])

        # 3. Pre-bake static frequency embedding
        if base.freq_emb is not None:
            with torch.no_grad():
                frs = torch.arange(512)
                baked_emb = base.freq_emb(frs).t()[None, :, :, None] * base.freq_emb_scale
                self.register_buffer("baked_freq_emb", baked_emb.clone())
        else:
            self.baked_freq_emb = None

        # 4. Patch CrossTransformer: static embeddings, decomposed MHA, and decomposed GroupNorm
        self.crosstransformer = copy.deepcopy(base.crosstransformer)
        for m in self.crosstransformer.modules():
            if isinstance(m, CrossTransformerEncoder):
                def _static_pos_emb(self_, T, B, C_emb, device):
                    return create_sin_embedding(T, C_emb, shift=0, device=device, max_period=self_.max_period)
                m._get_pos_embedding = types.MethodType(_static_pos_emb, m)
            elif isinstance(m, nn.MultiheadAttention):
                m.forward = types.MethodType(_npu_friendly_mha_forward, m)

        # Decompose norm_out in all CrossTransformer encoder layers
        for layer in self.crosstransformer.layers:
            if hasattr(layer, "norm_out") and isinstance(layer.norm_out, nn.GroupNorm):
                layer.norm_out = DecomposedGroupNorm1(layer.norm_out)
        for layer in self.crosstransformer.layers_t:
            if hasattr(layer, "norm_out") and isinstance(layer.norm_out, nn.GroupNorm):
                layer.norm_out = DecomposedGroupNorm1(layer.norm_out)

        self.bottom_channels = base.bottom_channels
        if self.bottom_channels:
            self.channel_upsampler = conv1d_to_conv2d_nondilated(base.channel_upsampler)
            self.channel_downsampler = conv1d_to_conv2d_nondilated(base.channel_downsampler)
            self.channel_upsampler_t = conv1d_to_conv2d_nondilated(base.channel_upsampler_t)
            self.channel_downsampler_t = conv1d_to_conv2d_nondilated(base.channel_downsampler_t)

    def forward(self, x_spec: torch.Tensor, x_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x_spec
        xt = x_time
        saved_spec, saved_time = [], []

        for idx in range(len(self.encoder)):
            xt = self.tencoder[idx](xt)
            saved_time.append(xt)

            x = self.encoder[idx](x)
            if idx == 0 and self.baked_freq_emb is not None:
                x = x + self.baked_freq_emb
            saved_spec.append(x)

        if self.crosstransformer is not None:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = x.flatten(2).unsqueeze(2)
                x = self.channel_upsampler(x)
                x = x.squeeze(2).view(b, -1, f, t)
                xt = self.channel_upsampler_t(xt)

            xt_3d = xt.squeeze(2)
            x, xt_3d = self.crosstransformer(x, xt_3d)
            xt = xt_3d.unsqueeze(2)

            if self.bottom_channels:
                x = x.flatten(2).unsqueeze(2)
                x = self.channel_downsampler(x)
                x = x.squeeze(2).view(b, c, f, t)
                xt = self.channel_downsampler_t(xt)

        for idx in range(len(self.decoder)):
            skip_s = saved_spec.pop(-1)
            x = self.decoder[idx](x, skip_s)

            skip_t = saved_time.pop(-1)
            xt = self.tdecoder[idx](xt, skip_t)

        return x, xt


# =====================================================================
# 8. Parity Verification Against Stock Demucs
# =====================================================================
def verify_parity_with_stock(core: HTDemucsCandidateAPureAIECore, base: nn.Module):
    torch.manual_seed(42)
    x_spec = torch.randn(1, 4, 2048, 336, dtype=torch.float32)
    x_time_3d = torch.randn(1, 2, 344064, dtype=torch.float32)
    x_time_4d = x_time_3d.unsqueeze(2)

    base_det = copy.deepcopy(base).eval()
    for m in base_det.modules():
        if isinstance(m, CrossTransformerEncoder):
            def _static_pos_emb(self_, T, B, C_emb, device):
                return create_sin_embedding(T, C_emb, shift=0, device=device, max_period=self_.max_period)
            m._get_pos_embedding = types.MethodType(_static_pos_emb, m)

    print("Running forward pass on Pure AIE Core...")
    with torch.no_grad():
        out_spec_pure, out_time_pure = core(x_spec, x_time_4d)

        # Stock Demucs forward pass
        x = x_spec.clone()
        xt = x_time_3d.clone()
        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(base_det.encoder):
            lengths.append(x.shape[-1])
            lengths_t.append(xt.shape[-1])
            xt = base_det.tencoder[idx](xt)
            saved_t.append(xt)
            x = encode(x, None)
            if idx == 0 and base_det.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = base_det.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + base_det.freq_emb_scale * emb
            saved.append(x)

        if base_det.crosstransformer:
            if base_det.bottom_channels:
                b, c, f, t = x.shape
                x = base_det.channel_upsampler(x.flatten(2)).view(b, -1, f, t)
                xt = base_det.channel_upsampler_t(xt)
            x, xt = base_det.crosstransformer(x, xt)
            if base_det.bottom_channels:
                x = base_det.channel_downsampler(x.flatten(2)).view(b, c, f, t)
                xt = base_det.channel_downsampler_t(xt)

        for idx, decode in enumerate(base_det.decoder):
            x, _ = decode(x, saved.pop(-1), lengths.pop(-1))
            xt, _ = base_det.tdecoder[idx](xt, saved_t.pop(-1), lengths_t.pop(-1))

        stock_drums_spec = x[:, :4, :, :]
        stock_drums_time = xt[:, :2, :].unsqueeze(2)

        diff_spec = (out_spec_pure - stock_drums_spec).abs().max().item()
        diff_time = (out_time_pure - stock_drums_time).abs().max().item()
        print(f"[PARITY CHECK] Max Abs Error -> Spec: {diff_spec:.2e}, Time: {diff_time:.2e}")
        assert diff_spec < 1e-4 and diff_time < 1e-4, f"Parity check failed: spec={diff_spec}, time={diff_time}"
        print("[SUCCESS] Pure AIE model matches stock Demucs within 1e-4 tolerance!")


# =====================================================================
# 9. Safety Net: ONNX InstanceNormalization Decomposer
# =====================================================================
def decompose_instance_norms_safety(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    graph = model.graph
    new_nodes = []
    new_initializers = []
    decomp_count = 0

    axis_name = "shared_inst_norm_axis_2"
    axis_init = helper.make_tensor(axis_name, TensorProto.INT64, [1], [2])
    new_initializers.append(axis_init)

    for node in graph.node:
        if node.op_type == "InstanceNormalization":
            x_in = node.input[0]
            scale_in = node.input[1]
            bias_in = node.input[2]
            y_out = node.output[0]

            eps = 1e-5
            for attr in node.attribute:
                if attr.name == "epsilon":
                    eps = attr.f

            eps_name = f"{node.name}_eps"
            eps_init = helper.make_tensor(eps_name, TensorProto.FLOAT, [1], [eps])
            new_initializers.append(eps_init)

            mean_out = f"{node.name}_mean"
            diff_out = f"{node.name}_diff"
            sq_diff_out = f"{node.name}_sq_diff"
            var_out = f"{node.name}_var"
            var_eps_out = f"{node.name}_var_eps"
            std_out = f"{node.name}_std"
            inv_std_out = f"{node.name}_inv_std"
            scaled_out = f"{node.name}_scaled"

            n_mean = helper.make_node("ReduceMean", [x_in, axis_name], [mean_out], name=f"{node.name}_mean", keepdims=1)
            n_diff = helper.make_node("Sub", [x_in, mean_out], [diff_out], name=f"{node.name}_diff")
            n_sq = helper.make_node("Mul", [diff_out, diff_out], [sq_diff_out], name=f"{node.name}_sq")
            n_var = helper.make_node("ReduceMean", [sq_diff_out, axis_name], [var_out], name=f"{node.name}_var", keepdims=1)
            n_add_eps = helper.make_node("Add", [var_out, eps_name], [var_eps_out], name=f"{node.name}_add_eps")
            n_sqrt = helper.make_node("Sqrt", [var_eps_out], [std_out], name=f"{node.name}_sqrt")
            n_div = helper.make_node("Div", [diff_out, std_out], [inv_std_out], name=f"{node.name}_div")
            n_scale = helper.make_node("Mul", [inv_std_out, scale_in], [scaled_out], name=f"{node.name}_scale")
            n_bias = helper.make_node("Add", [scaled_out, bias_in], [y_out], name=f"{node.name}_bias")

            new_nodes.extend([n_mean, n_diff, n_sq, n_var, n_add_eps, n_sqrt, n_div, n_scale, n_bias])
            decomp_count += 1
        else:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_initializers)
    return model, decomp_count


def verify_onnx_parity(onnx_path: str, ref_path: Path, min_snr: float = 50.0):
    """Checks the exported ONNX graph (ORT CPU) against PyTorch on real-music chunks."""
    if not ref_path.exists():
        print(f"[ONNX PARITY] Skipped: {ref_path} not found (run parity_real_audio.py make-ref first).")
        return
    import numpy as np
    import onnxruntime as ort

    def snr(r, e):
        r = r.astype(np.float64)
        return 10 * np.log10(np.sum(r**2) / (np.sum((r - e) ** 2) + 1e-30))

    ref = np.load(ref_path)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    worst = float("inf")
    for i, song in enumerate(ref["songs"]):
        out_spec, out_time = sess.run(None, {"x_spec": ref["x_spec"][i : i + 1], "x_time": ref["x_time"][i : i + 1]})
        s, t = snr(ref["ref_spec"][i : i + 1], out_spec), snr(ref["ref_time"][i : i + 1], out_time)
        worst = min(worst, s, t)
        print(f"[ONNX PARITY] {song}: spec {s:.2f} dB | time {t:.2f} dB")
    assert worst >= min_snr, f"ONNX vs PyTorch parity too low on real audio: {worst:.2f} dB < {min_snr} dB"
    print(f"[SUCCESS] ONNX matches PyTorch on real audio (worst {worst:.2f} dB >= {min_snr} dB)")


# =====================================================================
# 9b. Precision Guard: Split Multi-Axis ReduceMean
# =====================================================================
def split_multi_axis_reduce_means(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """
    onnxsim fuses the two-stage means in DConvGroupNorm4D / DecomposedGroupNorm1 back
    into a single ReduceMean(axes=[1, 3]). Over 8.3M time-branch elements that costs
    ~40 dB of SNR in ORT FP32, so re-split every keepdims=1 multi-axis ReduceMean into a
    chain of single-axis ReduceMeans, innermost (longest) axis first.
    """
    from onnx import numpy_helper

    graph = model.graph
    consts = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for n in graph.node:
        if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value":
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)

    new_nodes, new_inits, split_count = [], [], 0
    axis_inits = {}
    for node in graph.node:
        keepdims = next((a.i for a in node.attribute if a.name == "keepdims"), 1)
        axes = consts.get(node.input[1]) if node.op_type == "ReduceMean" and len(node.input) > 1 else None
        if axes is None or keepdims != 1 or axes.size < 2:
            new_nodes.append(node)
            continue
        cur = node.input[0]
        ordered = sorted(axes.tolist(), reverse=True)
        for k, ax in enumerate(ordered):
            ax_name = f"split_reduce_axis_{ax}"
            if ax_name not in axis_inits:
                axis_inits[ax_name] = helper.make_tensor(ax_name, TensorProto.INT64, [1], [ax])
            out = node.output[0] if k == len(ordered) - 1 else f"{node.name}_split{k}"
            new_nodes.append(helper.make_node("ReduceMean", [cur, ax_name], [out], name=f"{node.name}_split{k}", keepdims=1))
            cur = out
        split_count += 1

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(axis_inits.values())
    return model, split_count


# =====================================================================
# 10. Main Export Pipeline
# =====================================================================
def export_pure_aie(tag: str = "_v2"):
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading f7e0c4bc base model...")
    raw_model = get_model("f7e0c4bc")
    if hasattr(raw_model, "models"):
        raw_model = raw_model.models[0]
    raw_model.eval()

    print("\nStep 2: Constructing HTDemucsCandidateAPureAIECore...")
    core = HTDemucsCandidateAPureAIECore(raw_model).eval()

    print("\nStep 3: Verifying parity against stock Demucs...")
    verify_parity_with_stock(core, raw_model)

    raw_onnx_path = str(models_dir / f"htdemucs_candidate_a_pure_aie{tag}_raw.onnx")
    simp_onnx_path = str(models_dir / f"htdemucs_candidate_a_pure_aie{tag}_simp.onnx")

    dummy_spec = torch.randn(1, 4, 2048, 336, dtype=torch.float32)
    dummy_time = torch.randn(1, 2, 1, 344064, dtype=torch.float32)

    print(f"\nStep 4: Exporting FP32 ONNX model to {raw_onnx_path}...")
    torch.onnx.export(
        core,
        (dummy_spec, dummy_time),
        raw_onnx_path,
        input_names=["x_spec", "x_time"],
        output_names=["out_spec", "out_time"],
        opset_version=18,
        do_constant_folding=True,
    )
    print(f"[OK] Exported raw model ({os.path.getsize(raw_onnx_path)/(1024*1024):.2f} MB)")

    print(f"\nStep 5: Simplifying ONNX model via onnxsim -> {simp_onnx_path}...")
    m = onnx.load(raw_onnx_path)
    model_simp, check = simplify(m)
    assert check, "ONNX Simplifier validation failed"

    # Safety check for InstanceNormalization
    model_simp, decomp_count = decompose_instance_norms_safety(model_simp)
    if decomp_count > 0:
        print(f"[Safety Net] Decomposed {decomp_count} InstanceNormalization nodes created during export/simplification.")
    else:
        print("[Safety Net] Confirmed 0 InstanceNormalization nodes in simplified model!")

    model_simp, split_count = split_multi_axis_reduce_means(model_simp)
    print(f"[Precision Guard] Split {split_count} multi-axis ReduceMean nodes into single-axis chains.")

    onnx.save(model_simp, simp_onnx_path)
    print(f"[OK] Successfully saved simplified model: {simp_onnx_path} ({os.path.getsize(simp_onnx_path)/(1024*1024):.2f} MB)")

    # Audit operator counts
    op_counts = {}
    for n in model_simp.graph.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1

    print("\n=== Pure AIE Simplified Model Operator Summary ===")
    print(f"Total Nodes: {len(model_simp.graph.node)}")
    for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {op}: {cnt}")

    print("\n=== Scissor Operator Audit ===")
    print(f"  InstanceNormalization: {op_counts.get('InstanceNormalization', 0)}")
    print(f"  ConvTranspose:         {op_counts.get('ConvTranspose', 0)}")

    # Check for asymmetric Convs (kernel != [1, 1], [3, 3], etc. or stride > 1 with asymmetric shape)
    asym_convs = 0
    for n in model_simp.graph.node:
        if n.op_type == "Conv":
            for a in n.attribute:
                if a.name == "kernel_shape" and list(a.ints) in [[8, 1], [1, 8]]:
                    asym_convs += 1
    print(f"  Asymmetric Convs [8, 1]/[1, 8]: {asym_convs}")

    print("\nStep 6: Verifying ONNX parity against PyTorch on real audio...")
    verify_onnx_parity(simp_onnx_path, script_dir / "parity_real_audio.npz")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export the Pure AIE HTDemucs drums core to ONNX")
    parser.add_argument("--tag", default="_v2", help="Suffix for output model names (default: _v2)")
    export_pure_aie(parser.parse_args().tag)

