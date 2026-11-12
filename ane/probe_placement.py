"""
Which ops does Core ML put on the Neural Engine, as a function of tensor size?

Builds a tiny conv -> mul/add -> exp -> div chain on a [1, C, 1, W] tensor (or [1, C, H, 1] with
--axis h), converts it to an fp16 ML Program and asks MLComputePlan (macOS 14.4+) where every op
will run. This is how the 16384 rule in ane_modules.py was confirmed, instead of guessed.

Measured on an M1 under macOS 27.2:
  - hard cap: a dimension of 16384 runs on the ANE, 16385 runs on the CPU (same for W and H);
  - small tensors stay on the CPU too: Core ML judges the ANE dispatch not worth it. The cutoff
    depends on the element count (C=16 needs W=16384; C=256 is on the ANE from W=4096).
  In the full model every op is big enough, so only the hard cap mattered.

  .venv/bin/python ane/probe_placement.py
  .venv/bin/python ane/probe_placement.py --axis h --sizes 8192 16384 16385
"""
import argparse
import collections
import contextlib
import io
import shutil
import tempfile
import warnings

import coremltools as ct
import torch
import torch.nn as nn
from coremltools.models.compute_plan import MLComputePlan

warnings.filterwarnings("ignore")

def convert_quietly(*args, **kwargs):
    """ct.convert without its tqdm progress bars."""
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        return ct.convert(*args, **kwargs)


class Chain(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        y = self.conv(x) * 1.5 + 0.25
        return y / (1.0 + torch.exp(-y))


def placement(shape) -> str:
    ep = torch.export.export(Chain(shape[1]).eval(), (torch.randn(*shape),)).run_decompositions({})
    ml = convert_quietly(ep, inputs=[ct.TensorType("x", shape=shape)], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14,
                    compute_units=ct.ComputeUnit.CPU_AND_NE)
    tmp = tempfile.mkdtemp()
    try:
        ml.save(f"{tmp}/m.mlpackage")
        ml = ct.models.MLModel(f"{tmp}/m.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
        plan = MLComputePlan.load_from_path(ml.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
        where = collections.Counter()
        for op in plan.model_structure.program.functions["main"].block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is not None:  # consts have no usage
                where[type(usage.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")] += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if set(where) == {"NeuralEngine"}:
        return "ANE"
    return "CPU" if set(where) == {"CPU"} else str(dict(where))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--axis", choices=("w", "h"), default="w", help="Which dimension to make long")
    p.add_argument("--sizes", type=int, nargs="+", default=[1024, 4096, 8192, 10752, 16384, 16385, 21504, 32768])
    p.add_argument("--channels", type=int, nargs="+", default=[16, 64, 256])
    args = p.parse_args()

    print("[1, C, 1, N] with N varied" if args.axis == "w" else "[1, C, N, 1] with N varied")
    print(f"{'N':>8s} " + "".join(f"{'C=' + str(c):>8s}" for c in args.channels))
    for n in args.sizes:
        row = []
        for c in args.channels:
            shape = (1, c, 1, n) if args.axis == "w" else (1, c, n, 1)
            row.append(placement(shape))
        print(f"{n:8d} " + "".join(f"{r:>8s}" for r in row), flush=True)


if __name__ == "__main__":
    main()
