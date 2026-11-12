"""
How accurate is one op, or a short chain, when Core ML runs it on the ANE in fp16?

Each case is a 1x1 conv (weights exactly representable in fp16) followed by an activation. It is
converted to an fp16 ML Program, run on the chosen compute units, and compared with the same
module in fp64 on the fp16-rounded input. On the ANE the first row is the floor: plain fp16
rounding of the input gives ~73-75 dB, so anything near that is "free".

This is how the exp/divide activations in ane_modules.py were found. The finding (M1, macOS 27.2):
    conv -> sigmoid           51 dB on the ANE
    conv -> 1 / (1 + exp(-y)) 68 dB on the ANE
    conv -> GELU / GLU        similar loss to sigmoid, and the erf/tanh/sigmoid spellings of
                              GELU get re-fused by coremltools; only the exp/divide spelling survives
Each op alone is fine (sigmoid alone: 75 dB), so the loss shows up when an activation follows a
conv. That points to the ANE compiler fusing the activation into the conv's output stage, but
the compiler can't be inspected: that explanation is inferred from the numbers.

--units CPU_ONLY runs the same fp16 programs on the CPU for contrast. The CPU is a different
reference, not a better one: there the conv alone scores ~58 dB (vs ~74 on the ANE) and the
activations do not show the same split (sigmoid 65 dB, exp form 57 dB).

  .venv/bin/python ane/probe_precision.py
  .venv/bin/python ane/probe_precision.py --units CPU_ONLY        # same program on the CPU, for contrast
"""
import argparse
import contextlib
import io
import warnings

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

def convert_quietly(*args, **kwargs):
    """ct.convert without its tqdm progress bars."""
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        return ct.convert(*args, **kwargs)


def snr_db(module: nn.Module, x: np.ndarray, units) -> float:
    ep = torch.export.export(module.eval(), (torch.from_numpy(x),)).run_decompositions({})
    ml = convert_quietly(ep, inputs=[ct.TensorType("x", shape=x.shape)], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14,
                    compute_units=units)
    got = next(iter(ml.predict({"x": x}).values())).astype(np.float64)
    with torch.no_grad():
        ref = module.double()(torch.from_numpy(x.astype(np.float16).astype(np.float64))).numpy()
    return float(10 * np.log10((ref**2).sum() / ((ref - got) ** 2).sum()))


class ConvAct(nn.Module):
    def __init__(self, act):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(64, 64, 1)
        self.conv.weight.data = self.conv.weight.data.half().float()
        self.conv.bias.data = self.conv.bias.data.half().float()
        self.act = act

    def forward(self, x):
        return self.act(self.conv(x))


CASES = {
    "conv only (the floor)": lambda y: y,
    "conv -> sigmoid": torch.sigmoid,
    "conv -> 1 / (1 + exp(-y))": lambda y: 1 / (1 + torch.exp(-y)),
    "conv -> GELU (erf)": F.gelu,
    "conv -> GELU, tanh form, via exp": lambda y: y / (1 + torch.exp(-1.5957691216057308 * (y + 0.044715 * y**3))),
    "conv -> GLU": lambda y: F.glu(y, dim=1),
    "conv -> GLU via exp": lambda y: y[:, :32] / (1 + torch.exp(-y[:, 32:])),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--units", default="CPU_AND_NE", choices=("CPU_AND_NE", "CPU_ONLY"))
    p.add_argument("--scale", type=float, default=3.0, help="Std of the random input (activation size matters)")
    args = p.parse_args()

    units = getattr(ct.ComputeUnit, args.units)
    x = (np.random.default_rng(0).standard_normal((1, 64, 8, 4096)) * args.scale).astype(np.float32)
    print(f"1x1 conv + activation, input std {args.scale}, compute units {args.units}")
    for name, act in CASES.items():
        print(f"  {name:36s} {snr_db(ConvAct(act), x, units):5.1f} dB", flush=True)


if __name__ == "__main__":
    main()
