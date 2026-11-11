import copy
import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from demucs.pretrained import get_model
from demucs.transformer import CrossTransformerEncoder, create_sin_embedding
from demucs.hdemucs import HEncLayer, HDecLayer, DConv


# =====================================================================
# 1. Safe 1D -> 2D (1, K) Convolution Converters
# =====================================================================
def conv1d_to_conv2d(conv1d: nn.Conv1d) -> nn.Conv2d:
    """Converts Conv1d(C_in, C_out, K) -> Conv2d(C_in, C_out, (1, K)) reading shapes from weight."""
    out_ch, in_ch_per_group, k = conv1d.weight.shape
    in_ch = in_ch_per_group * conv1d.groups
    conv2d = nn.Conv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=(1, k),
        stride=(1, conv1d.stride[0]),
        padding=(0, conv1d.padding[0]),
        dilation=(1, conv1d.dilation[0]),
        groups=conv1d.groups,
        bias=(conv1d.bias is not None),
    )
    with torch.no_grad():
        conv2d.weight.copy_(conv1d.weight.unsqueeze(2))
        if conv1d.bias is not None:
            conv2d.bias.copy_(conv1d.bias)
    return conv2d


def convtr1d_to_convtr2d(convtr1d: nn.ConvTranspose1d) -> nn.ConvTranspose2d:
    """
    Converts ConvTranspose1d -> ConvTranspose2d((1, K)).
    Derives in_channels and out_channels directly from convtr1d.weight.shape
    so pruned output heads (8 -> 2 channels) never mismatch module metadata.
    """
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
# 2. Pure 4D DConv & Per-Frequency GroupNorm
# =====================================================================
class DConvGroupNorm4D(nn.Module):
    """
    Replaces GroupNorm(1, C) inside DConv only.
    Normalizes over dim=(1, 3) on 4D (B, C, H, T) to match stock Demucs
    (B * Fr, C, T) flattened GroupNorm(1, C) for both Frequency (H=Fr)
    and Time (H=1) branches without leaving 4D.
    """
    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        assert gn.num_groups == 1, f"Expected num_groups=1 in DConv, got {gn.num_groups}"
        self.eps = gn.eps
        self.weight = nn.Parameter(gn.weight.data.view(1, -1, 1, 1).clone())
        self.bias = nn.Parameter(gn.bias.data.view(1, -1, 1, 1).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 3), keepdim=True)
        var = x.var(dim=(1, 3), unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class LayerScale4D(nn.Module):
    """4D replacement for DConv's LayerScale so (C, 1) broadcasts over (B, C, H, W)."""
    def __init__(self, ls_module: nn.Module):
        super().__init__()
        self.scale = nn.Parameter(ls_module.scale.data.view(1, -1, 1, 1).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class DConv4D(nn.Module):
    """Strictly 4D (B, C, H, W) replacement for DConv."""
    def __init__(self, dconv: DConv):
        super().__init__()
        new_layers = nn.ModuleList()
        for seq in dconv.layers:
            new_seq = nn.Sequential()
            for name, layer in seq.named_children():
                if isinstance(layer, nn.Conv1d):
                    new_seq.add_module(name, conv1d_to_conv2d(layer))
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
# 3. Dedicated 4D Encoder & Decoder Layers (Zero Double-Padding!)
# =====================================================================
class FreqEncLayer4D(nn.Module):
    """
    4D Frequency Encoder Layer: operates on (B, C, Fr, T).
    Note: self.conv already contains padding=(self.pad, 0) from stock HEncLayer.
    """
    def __init__(self, enc: HEncLayer):
        super().__init__()
        assert enc.freq, "FreqEncLayer4D requires a frequency-branch HEncLayer"
        self.pad = enc.pad
        self.empty = enc.empty
        self.conv = copy.deepcopy(enc.conv)
        self.norm1 = copy.deepcopy(enc.norm1)
        self.dconv = DConv4D(enc.dconv) if (not enc.empty and enc.dconv is not None) else None
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


class FreqDecLayer4D(nn.Module):
    """
    4D Frequency Decoder Layer: operates on (B, C, Fr, T).
    Note: self.conv_tr has padding=(0, 0), so Height (dim 2) is cropped by [pad : -pad].
    """
    def __init__(self, dec: HDecLayer):
        super().__init__()
        assert dec.freq, "FreqDecLayer4D requires a frequency-branch HDecLayer"
        self.pad = dec.pad
        self.last = dec.last
        self.empty = dec.empty
        self.rewrite = copy.deepcopy(dec.rewrite) if (not dec.empty and dec.rewrite is not None) else None
        self.norm1 = copy.deepcopy(dec.norm1) if not dec.empty else None
        self.dconv = DConv4D(dec.dconv) if (not dec.empty and dec.dconv is not None) else None
        self.conv_tr = copy.deepcopy(dec.conv_tr)
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
        if self.pad > 0:
            z = z[:, :, self.pad : -self.pad, :]  # Crop Height (Fr, dim 2)
        if not self.last:
            z = F.gelu(z)
        return z


class TimeEncLayer4D(nn.Module):
    """
    Dedicated 4D Time Encoder Layer: operates on (B, C, 1, L).
    Note: self.conv converted via conv1d_to_conv2d ALREADY has padding=(0, self.pad)!
    We only pad the right edge when L is not divisible by self.stride.
    """
    def __init__(self, enc: HEncLayer):
        super().__init__()
        assert not enc.freq, "TimeEncLayer4D requires a time-branch HEncLayer"
        self.pad = enc.pad
        self.stride = enc.stride
        self.empty = enc.empty
        self.conv = conv1d_to_conv2d(enc.conv)
        self.norm1 = copy.deepcopy(enc.norm1)
        self.dconv = DConv4D(enc.dconv) if (not enc.empty and enc.dconv is not None) else None
        self.rewrite = conv1d_to_conv2d(enc.rewrite) if (not enc.empty and enc.rewrite is not None) else None
        self.norm2 = copy.deepcopy(enc.norm2) if not enc.empty else None

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        assert xt.dim() == 4 and xt.shape[2] == 1, f"Expected (B, C, 1, L), got {xt.shape}"
        le = int(xt.shape[-1])
        if le % self.stride != 0:
            xt = F.pad(xt, (0, self.stride - (le % self.stride)))
        # DO NOT manually F.pad by self.pad here: self.conv already has padding=(0, self.pad)!
        xt = self.norm1(self.conv(xt))
        if self.empty:
            return xt
        xt = F.gelu(xt)
        if self.dconv is not None:
            xt = self.dconv(xt)
        if self.rewrite is not None:
            xt = F.glu(self.norm2(self.rewrite(xt)), dim=1)
        assert xt.shape[2] == 1, f"TimeEncLayer4D corrupted height: {xt.shape}"
        return xt


class TimeDecLayer4D(nn.Module):
    """
    Dedicated 4D Time Decoder Layer: operates on (B, C, 1, L).
    Note: self.conv_tr has padding=(0, 0), so we crop Width (dim 3) to [pad : pad + target_len].
    """
    def __init__(self, dec: HDecLayer):
        super().__init__()
        assert not dec.freq, "TimeDecLayer4D requires a time-branch HDecLayer"
        self.pad = int(dec.pad)
        self.last = dec.last
        self.empty = dec.empty
        self.rewrite = conv1d_to_conv2d(dec.rewrite) if (not dec.empty and dec.rewrite is not None) else None
        self.norm1 = copy.deepcopy(dec.norm1) if not dec.empty else None
        self.dconv = DConv4D(dec.dconv) if (not dec.empty and dec.dconv is not None) else None
        self.conv_tr = convtr1d_to_convtr2d(dec.conv_tr)
        self.norm2 = copy.deepcopy(dec.norm2)

    def forward(self, xt: torch.Tensor, skip: torch.Tensor, target_len: int) -> torch.Tensor:
        assert xt.dim() == 4 and xt.shape[2] == 1, f"Expected (B, C, 1, L), got {xt.shape}"
        if not self.empty:
            assert xt.shape == skip.shape, f"Skip mismatch: xt={xt.shape} vs skip={skip.shape}"
            y = xt + skip
            if self.rewrite is not None:
                y = F.glu(self.norm1(self.rewrite(y)), dim=1)
            if self.dconv is not None:
                y = self.dconv(y)
        else:
            y = xt
        z = self.norm2(self.conv_tr(y))
        start = self.pad
        end = self.pad + int(target_len)
        z = z[:, :, :, start:end]
        assert z.shape[2] == 1 and z.shape[3] == target_len, (
            f"TimeDecLayer4D output shape mismatch: got {z.shape}, expected H=1, W={target_len}"
        )
        if not self.last:
            z = F.gelu(z)
        return z


# =====================================================================
# 4. Streamlined Batch-First Decomposed MHA for Vitis AI EP
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
# 5. Complete XDNA 2 4D Core Module (Drums-Only)
# =====================================================================
class HTDemucsXDNA2Core(nn.Module):
    def __init__(self, base_model: nn.Module = None):
        super().__init__()
        if base_model is None:
            base = get_model("f7e0c4bc")
            if hasattr(base, "models"):
                base = base.models[0]
        else:
            base = copy.deepcopy(base_model)
        base.eval()

        # 1. Surgically prune final output heads to Drums-Only (Index 0)
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

        # 2. Build explicit 4D Encoder and Decoder stacks
        self.encoder = nn.ModuleList([FreqEncLayer4D(enc) for enc in base.encoder])
        self.decoder = nn.ModuleList([FreqDecLayer4D(dec) for dec in base.decoder])
        self.tencoder = nn.ModuleList([TimeEncLayer4D(enc) for enc in base.tencoder])
        self.tdecoder = nn.ModuleList([TimeDecLayer4D(dec) for dec in base.tdecoder])

        # 3. Pre-bake static frequency embedding (512 bins after encoder[0])
        if base.freq_emb is not None:
            with torch.no_grad():
                frs = torch.arange(512)
                baked_emb = base.freq_emb(frs).t()[None, :, :, None] * base.freq_emb_scale
                self.register_buffer("baked_freq_emb", baked_emb.clone())
        else:
            self.baked_freq_emb = None

        # 4. Patch CrossTransformer static positional embedding & decomposed MHA
        self.crosstransformer = copy.deepcopy(base.crosstransformer)
        for m in self.crosstransformer.modules():
            if isinstance(m, CrossTransformerEncoder):
                def _static_pos_emb(self_, T, B, C_emb, device):
                    return create_sin_embedding(T, C_emb, shift=0, device=device, max_period=self_.max_period)
                m._get_pos_embedding = types.MethodType(_static_pos_emb, m)
            elif isinstance(m, nn.MultiheadAttention):
                m.forward = types.MethodType(_npu_friendly_mha_forward, m)

        self.bottom_channels = base.bottom_channels
        if self.bottom_channels:
            self.channel_upsampler = conv1d_to_conv2d(base.channel_upsampler)
            self.channel_downsampler = conv1d_to_conv2d(base.channel_downsampler)
            self.channel_upsampler_t = conv1d_to_conv2d(base.channel_upsampler_t)
            self.channel_downsampler_t = conv1d_to_conv2d(base.channel_downsampler_t)

    def forward(self, x_spec: torch.Tensor, x_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x_spec
        xt = x_time
        saved_spec = []
        saved_time = []
        lengths_time = []

        # 1. Dual 4D Encoder Pass
        for idx in range(len(self.encoder)):
            lengths_time.append(int(xt.shape[-1]))
            xt = self.tencoder[idx](xt)
            saved_time.append(xt)

            x = self.encoder[idx](x)
            if idx == 0 and self.baked_freq_emb is not None:
                x = x + self.baked_freq_emb
            saved_spec.append(x)

        # 2. Cross-Transformer Bottleneck
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

        # 3. Dual 4D Decoder Pass
        for idx in range(len(self.decoder)):
            skip_s = saved_spec.pop(-1)
            x = self.decoder[idx](x, skip_s)

            skip_t = saved_time.pop(-1)
            target_len_t = lengths_time.pop(-1)
            xt = self.tdecoder[idx](xt, skip_t, target_len_t)

        return x, xt


# =====================================================================
# 6. Automated Parity Verification & ONNX Exporter
# =====================================================================
def verify_against_stock_demucs(core: HTDemucsXDNA2Core, base: nn.Module, time_samples: int, spec_frames: int):
    torch.manual_seed(42)
    x_spec = torch.randn(1, 4, 2048, spec_frames, dtype=torch.float32)
    x_time_3d = torch.randn(1, 2, time_samples, dtype=torch.float32)
    x_time_4d = x_time_3d.unsqueeze(2)

    base_det = copy.deepcopy(base).eval()
    for m in base_det.modules():
        if isinstance(m, CrossTransformerEncoder):
            def _static_pos_emb(self_, T, B, C_emb, device):
                return create_sin_embedding(T, C_emb, shift=0, device=device, max_period=self_.max_period)
            m._get_pos_embedding = types.MethodType(_static_pos_emb, m)

    with torch.no_grad():
        out_spec_4d, out_time_4d = core(x_spec, x_time_4d)

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

        diff_spec = (out_spec_4d - stock_drums_spec).abs().max().item()
        diff_time = (out_time_4d - stock_drums_time).abs().max().item()
        print(f"[Parity Check @ L={time_samples}] Max Abs Error -> Spec: {diff_spec:.2e}, Time: {diff_time:.2e}")
        assert diff_spec < 1e-4 and diff_time < 1e-4, "Parity check failed!"


if __name__ == "__main__":
    print("Loading f7e0c4bc drums specialist checkpoint...")
    raw_model = get_model("f7e0c4bc")
    if hasattr(raw_model, "models"):
        raw_model = raw_model.models[0]
    raw_model.eval()

    core = HTDemucsXDNA2Core(raw_model).eval()

    verify_against_stock_demucs(core, raw_model, time_samples=343980, spec_frames=336)
    verify_against_stock_demucs(core, raw_model, time_samples=344064, spec_frames=336)

    TARGET_SAMPLES = 344064
    dummy_spec = torch.randn(1, 4, 2048, 336, dtype=torch.float32)
    dummy_time = torch.randn(1, 2, 1, TARGET_SAMPLES, dtype=torch.float32)

    onnx_filename = "htdemucs_drums_xdna2_fp32.onnx"
    torch.onnx.export(
        core,
        (dummy_spec, dummy_time),
        onnx_filename,
        input_names=["x_spec", "x_time"],
        output_names=["out_spec", "out_time"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Successfully exported verified 4D ONNX graph to: {onnx_filename}")