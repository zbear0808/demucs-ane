# How the Core ML model was built

This is the build log and recipe for `ane/models/htdemucs_<stem>_ane.mlpackage` and the fused variant: what each step does, what broke, how each problem was found, and what the evidence was. All numbers were measured on an M1 Mac mini (macOS 27.2) with coremltools 9.0 and torch 2.7.0. Treat the thresholds below as measurements on that chip and OS, not as specifications.

Speed, memory and load-time results are in [README.md](README.md) and [PERFORMANCE.md](PERFORMANCE.md). This file is about how the model gets made.

**At a glance (drums):** 87 MB fp16 ML Program, 1,696 ops, 100% on the Neural Engine (ANE) by Core ML's own compute plan, worst-case error 45.9 dB below the mix on real audio (the bar is 40 dB), about 0.37 s per 7.8 s chunk.

## 1. The recipe on one page

```
PyTorch htdemucs (checkpoint f7e0c4bc, 4 stems)
  | select_source()                     keep one stem's output weights
  v
single-stem PyTorch model
  | HTDemucsCandidateAPureAIECore       XDNA2 rewrites, reused unchanged (dense convs, polyphase upsampling, 4D)
  v
"pure AIE" core
  | ane_modules.py                      the ANE rewrites (section 4)
  v
ANE-ready PyTorch model                 static shapes, every dimension <= 16384, exp/divide activations
  | torch.export(...).run_decompositions({})
  v
ExportedProgram
  | coremltools 9.0: fp16 ML Program, macOS 14 / iOS 17 target
  v
htdemucs_drums_ane.mlpackage
  | MLComputePlan (where do ops run?)  ->  real-audio parity (is it right?)  ->  macmon (is the ANE really busy?)
  v
done: 100% of ops on the ANE, 45.9 dB
```

The ANE-specific changes are five:

1. **Static, traceable code.** No shape arithmetic in `forward()`, no einops, positional embeddings baked in, attention written as plain matmuls. The original code breaks `torch.jit.trace` (shape arithmetic) and `torch.export` (einops and transformer helper methods), so nothing converts without this.
2. **Row folding.** The time branch is up to 344,064 samples long and the ANE only accepts dimensions up to 16,384. The signal is reshaped to rows of at most 10,752 samples, and each convolution borrows a few samples from the neighbouring rows so the result is unchanged.
3. **Native `layer_norm`** instead of hand-written mean/variance.
4. **Activations written as `x / (1 + exp(-z))`** instead of sigmoid, GELU and GLU ops. This is worth about 10 dB.
5. **For the fused model only:** STFT and iSTFT as matmuls with a Fourier matrix (section 8).

One command does the whole pipeline, including the checks:

```bash
.venv/bin/python ane/export_htdemucs_ane.py                 # drums: 2 to 3 minutes
.venv/bin/python ane/export_htdemucs_ane.py --source vocals
.venv/bin/python ane/export_htdemucs_ane.py --fused --onnx  # fused Core ML model (+ ONNX)
```

## 2. Why Core ML, and what that implies

- **There is no way to program the ANE directly.** The only route is Core ML (the notes in the `neural-engine` repo cloned next to this one say the same: "there is no public API to program the ANE"). Core ML decides, op by op, whether something runs on the ANE, the GPU or the CPU. You only choose which devices are allowed (`CPU_AND_NE`, `ALL`, ...). So the job is to build a graph that Core ML chooses to put entirely on the ANE.
- **coremltools cannot read ONNX.** `ct.convert("model.onnx")` fails with "Unable to determine the type of the model, i.e. the source framework" (its message lists `tensorflow`, `pytorch` and `milinternal` as the only sources). That is why this build starts from the PyTorch model instead of from the XDNA2 ONNX files. ONNX Runtime's CoreML execution provider is a second route (it converts ONNX to Core ML when the session starts); see README.
- **The ANE computes in fp16.** Anything fp32 cannot run there. This was confirmed with the compute plan: the fp32 ops in an early iSTFT showed up as "not ANE-capable". Precision, not speed, turned out to be the hard part of this build.
- **`MLComputePlan` is the ANE counterpart of the VAI-ML partition summary.** It reports the preferred device for every op, whether the ANE could run it, and a relative cost, without running the model (macOS 14.4+ / iOS 17.4+). Every placement claim below comes from it.

How this differs from the XDNA2 flow:

| | XDNA2 (`option_a_bfp16/`) | ANE (this folder) |
| :--- | :--- | :--- |
| Precision step | Quark BFP16 quantization with calibration | plain fp16 cast by coremltools, no calibration |
| Compiler | VAI-ML / aiecompiler, ahead of time, `.rai` cache | Core ML builds the ANE program when the model loads (about 25 s), cached by the OS |
| Placement report | VAI-ML partition summary | `MLComputePlan` |
| What blocked us | asymmetric convs, InstanceNorm, DMA alignment | shape arithmetic, dimensions over 16,384, fp16 precision of fused activations |
| Export route | torch -> ONNX -> Quark -> ORT VitisAI EP | torch.export -> coremltools -> `.mlpackage` |

The notes in the `neural-engine` repo (its `docs` folder) were a useful starting list, but they cover the older NeuralNetwork format and say themselves that they may be wrong. One rule did not hold: they say broadcastable layers fall back to the CPU, but every broadcasting `mul`/`add` in this ML Program (for example `x * scale` with a `[1, C, 1, 1]` scale) runs on the ANE. The fp16 warning held and dominated the work. Dilated convs, gather, RNNs and large pooling are on that list too, but this model has none of them after the XDNA2 rewrites, so those rules were never tested here.

## 3. Starting point

### 3.1 Environment

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r ane/requirements.txt
```

torch is pinned to 2.7.0, the newest version coremltools 9.0 is tested against (it warns on 2.7.1). The first run downloads the htdemucs weights through `demucs.pretrained.get_model("f7e0c4bc")`, so it needs network access once.

### 3.2 One stem: `select_source`

htdemucs predicts four stems. The last decoder layer's transposed convolution has its output channels laid out stem by stem (frequency branch: 4 complex-as-channels per stem; time branch: 2 per stem), so keeping one stem is a slice of that layer's weights and bias. The result is a normal single-stem model with a quarter of the final-layer work. The XDNA2 flow did the same for drums; here it is a parameter, which is how the vocals model was made with no other change.

### 3.3 The XDNA2 "pure AIE" core

`HTDemucsCandidateAPureAIECore` (imported from `option_a_bfp16/export_candidate_a_pure_aie.py`) is reused unchanged:

| Rewrite | What it does |
| :--- | :--- |
| Dilated 1D convs -> dense 2D kernels | `Conv1d(k=3, dilation=2)` becomes a `1x5` kernel with zeros between the taps |
| K=8, stride 4 convs -> unfold + 1x1 conv | pad, reshape into 4 phases, 8 shifted slices, concat, then a 1x1 conv |
| ConvTranspose -> polyphase 1x1 conv | 3 neighbouring inputs -> 4 output phases, interleaved by reshape |
| GroupNorm / InstanceNorm -> explicit maths | replaced again for the ANE (section 4.3) |
| Everything 4D `(B, C, H, W)` | the time signal `[B, C, L]` becomes `[B, C, 1, L]` |
| Frequency embedding and positional embeddings baked | no `arange`/`embedding` at run time |

These were kept because they remove awkward ops and are already verified against stock Demucs, not because each was proven necessary for the ANE. They were never ablated on the ANE, so whether a dilated conv or a ConvTranspose would have run there is unknown.

## 4. The ANE rewrites

Each subsection gives the symptom, how it was diagnosed, the fix and the evidence. All of it lives in `ane_modules.py`.

### 4.1 Make it convert: static shapes

**Symptoms.**
- `ct.convert(torch.jit.trace(core, ...))` stopped with `TypeError: only 0-dimensional arrays can be converted to Python scalars` in coremltools' `_int` handler.
- `torch.export.export(core, ...)` failed inside einops (`rearrange` in the cross-transformer), and after that with "Dynamo does not know how to trace method `_sa_block`".

**Cause.** The XDNA2 modules compute sizes from tensor shapes inside `forward()` (`N = L // 4`, `embed_dim // num_heads`, ...). TorchScript records those as `aten::Int` on tensors, which coremltools cannot evaluate, and Dynamo cannot trace through einops or the `nn.TransformerEncoderLayer` helper methods.

**Fix.** `forward()` uses only Python constants:
- The unfold and polyphase convs read their channel counts from their own conv (`self.conv1x1.in_channels // 8`) and use the constants `N_FRAMES = 336` and `self.rows`.
- `_patch_cross_transformer` bakes the 2D and 1D sinusoidal embeddings (already multiplied by `weight_pos_embed`) into buffers, then replaces the forwards of the cross-transformer and of each layer so they call `_attention` directly.
- `_attention` is one batch-1 multi-head attention written with `F.linear`, reshapes and two rank-3 `matmul`s. The `1/sqrt(head_dim)` scale is folded into the query projection's weights and bias, which keeps fp16 attention logits small.
- `HTDemucsSourceANE.forward` drives the encoders and decoders itself, with `_fold(x, length)` as the only reshape helper.

**Evidence.** It converts, and the first compute plan reads 1,429 of 1,707 ops (64% of estimated cost) on the ANE.

### 4.2 Make it fit: the 16,384 rule and row folding

**Symptom.** The 36% of cost that was not on the ANE was entirely the time branch. The offenders were `conv`, `mul`, `sub`, `reduce_mean`, `batch_norm`, `gelu`, `pad` and `concat` on tensors like `[1, 192, 1, 21504]`, `[1, 96, 1, 86016]` and `[1, 2, 1, 344068]`. Almost all were reported "not ANE-capable". The frequency branch and the transformer were already fully on the ANE.

**Diagnosis.** [`probe_placement.py`](probe_placement.py) builds a tiny `conv -> mul/add -> exp -> div` chain on `[1, C, 1, N]` and asks the compute plan where each op lands:

```
       N     C=16    C=64   C=256
    1024      CPU     CPU     CPU
    4096      CPU     CPU     ANE
    8192      CPU     ANE     ANE
   10752      CPU     ANE     ANE
   16384      ANE     ANE     ANE
   16385      CPU     CPU     CPU
   21504      CPU     CPU     CPU
   32768      CPU     CPU     CPU
```

Two rules show up:
1. **Hard cap:** 16,384 runs on the ANE and 16,385 runs on the CPU, for every channel count and for both the last and the third dimension. The channel dimension was not tested.
2. **Too small stays on the CPU:** Core ML judges tiny ops not worth dispatching, and the cutoff depends on the element count. In the full model every op is big enough, so only the hard cap mattered.

**Fix: row folding with halos.** The time branch runs at 344,064, 86,016, 21,504, 5,376 and 1,344 samples. A `[1, C, 1, L]` tensor is reshaped, for free, into `[1, C, R, L/R]` with `R` the smallest power of two that brings the row width to 16,384 or less:

| Samples | Rows x width |
| ---: | :--- |
| 344,064 | 32 x 10,752 |
| 86,016 | 8 x 10,752 |
| 21,504 | 2 x 10,752 |
| 5,376 | 1 x 5,376 |
| 1,344 | 1 x 1,344 |

A convolution along the width would now be wrong at every row boundary, so each one first borrows `p` columns from its neighbours (zeros at the first and last row), then runs with no padding:

```python
def _halo(x, p):   # x: [1, C, R, W] -> [1, C, R, W + 2p]
    tail_prev = F.pad(x[..., -p:], (0, 0, 1, 0))[:, :, :-1, :]   # previous row's last p columns
    head_next = F.pad(x[..., :p], (0, 0, 0, 1))[:, :, 1:, :]     # next row's first p columns
    return torch.cat([tail_prev, x, head_next], dim=3)
```

- **Which layers:** `HaloConv` wraps every conv whose kernel is wider than 1 along the folded axis: the 3-tap DConv convs (5 wide where they were dilated) and the decoder `rewrite` convs. The stride-4 unfold borrows 2 columns and the polyphase upsample borrows 1. Per-pixel ops need nothing.
- **Normalisations** reduce over channels, rows and width, so they still see the whole signal.
- **Upsampling** emits `[1, out, 4R, N]` rather than `[1, out, R, 4N]`. The wider form puts the row width back over the limit (for example 43,008), and that was the last two ops stuck on the CPU. Between layers, `_fold(x, length)` re-reshapes to the canonical fold for the next length, which is again free because the layout is row-major.
- **Exactness:** the folded graph matched the unfolded one to about 5e-6 max absolute error in fp32 when it was introduced, and `check_torch_parity` asserts 50 dB or better against the pure-AIE core on every build.

**Evidence, from the compute plan at each stage:**

| Stage | Ops on the ANE | Estimated cost on the ANE |
| :--- | ---: | ---: |
| Static shapes only | 1,429 / 1,707 | 64% |
| + row folding | 1,962 / 1,964 | 99.7% (two GELUs still on the CPU) |
| + upsample emits 4R rows | 1,964 / 1,964 | 100% |
| Final graph (4.3 and 4.4 changed the op count) | 1,696 / 1,696 | 100% |

The earlier stages came from throwaway scripts; the final row is what `export_htdemucs_ane.py` prints.

### 4.3 Normalisation: native `layer_norm`

The XDNA2 build wrote GroupNorm(1) as about eight ops (two-stage `ReduceMean`, subtract, square, sqrt, divide) because one reduction over millions of elements lost precision in ONNX Runtime fp32. Here each one is a single `F.layer_norm`, which Core ML lowers to its `layer_norm` op:

- **Frequency-branch DConv blocks:** GroupNorm over (channels, width) with frequency acting as batch. The tensor is transposed to `[1, H, C, W]` and normalised over `[C, W]`.
- **Time-branch blocks:** over (channels, rows, width), which is `x.shape[1:]`.
- **Transformer `norm_out`:** GroupNorm(1) on `[1, T, C]`, which normalises over `(T, C)`.
- **The affine part** (`* weight + bias`) stays separate. coremltools fuses a per-channel `mul` + `add` into a `batch_norm` op (pass `fuse_elementwise_to_batchnorm`), which is why the final graph has 64 of them.

Accuracy was not affected: native `layer_norm` and the decomposed form were within 0.5 dB of each other on every test song, and the per-layer SNR table in 6.2 stays at 53 dB or better through the time encoder. The native op is simpler: one op instead of about eight.

### 4.4 Make it accurate: fp16 and the activations

**Symptom.** The first build with every op on the ANE scored a worst-case error of only **35.5 dB** below the mix. The bar is 40 dB, and the XDNA2 BFP16 model had been checked against the same bar.

**Diagnosis, in the order it was done:**
1. *Where does the error enter?* A per-layer comparison against PyTorch fp32 (the `--debug-taps` mode, section 6.2) showed error growing steadily from the first layer instead of jumping at one point, so the cause looked like a per-op effect rather than one bad layer.
2. *Which op families matter?* Converting with `FP16ComputePrecision(op_selector=...)` to keep one op family in fp32 and re-measuring showed, on the early graph, that convs, sigmoid/GELU and `reduce_mean` moved the result most. This is only a diagnostic: fp32 ops cannot run on the ANE. On the early graph, keeping the convs in fp32 lifted the worst case to about 65 dB but took about 3 s per chunk, which is CPU/GPU speed.
3. *Is a single op accurate on the ANE?* [`probe_precision.py`](probe_precision.py) runs a 1x1 conv plus one activation on the ANE and compares it with the same module in fp64 on the fp16-rounded input:

```
conv only (the floor)                 73.6 dB
conv -> sigmoid                       51.1 dB
conv -> 1 / (1 + exp(-y))             68.3 dB
conv -> GELU (erf)                    59.5 dB
conv -> GELU, tanh form, via exp      67.3 dB
conv -> GLU                           51.1 dB
conv -> GLU via exp                   66.3 dB
```

   Each op on its own is fine (a lone sigmoid scored 75.5 dB), but a sigmoid, GELU or GLU that follows a conv loses 14 to 22 dB. *(Inferred, not observed: the ANE compiler most likely fuses the activation into the conv's output stage with a coarser approximation. The compiler cannot be inspected.)* Run on the CPU in fp16, the same `conv -> sigmoid` scores 65.1 dB, so the extra loss comes from the ANE. Replacing all 104 activations in the network recovered about 10 dB end to end (35.5 to 45.9 dB).

**Fix.** Never let the compiler see a sigmoid, GELU or GLU op:
- `sigmoid` becomes `1 / (1 + exp(-y))`.
- GLU becomes `a / (1 + exp(-b))`.
- GELU becomes the tanh approximation written with `exp`: `x / (1 + exp(-1.5957691216 * (x + 0.044715 * x^3)))`. The tanh form is used because coremltools has graph passes (`fuse_gelu_exact`, `fuse_gelu_tanh_approximation`) that turn the erf and tanh spellings back into a single `gelu` op (a hand-written erf GELU scored exactly the same as the built-in one in an earlier test), and a sigmoid-based spelling still leaves a `sigmoid` op. Only the exp/divide form stays as separate ops. The model's output differs from the exact-GELU model's by an error 61 dB below the signal, well below fp16 noise.

The final graph has 104 `exp` and 104 `real_div` ops and no `gelu`, `sigmoid` or `glu`. The exp/divide ops are about 11% of Core ML's estimated cost, and in return the worst case went from 35.5 to 45.9 dB.

**Where it ends up.** On the same three real-audio chunks (worst case, error below mix):

| | Worst error below mix |
| :--- | ---: |
| First all-ANE build (native sigmoid/GELU/GLU) | 35.5 dB |
| Final model on the ANE | **45.9 dB** |
| Final model, same fp16 program on the CPU | 45.4 dB |
| PyTorch fp32 version of the same graph, including STFT/iSTFT | 64.9 dB |

So the ANE is no worse than the CPU's fp16 path any more, and what remains looks like the cost of fp16 arithmetic in this network. Going higher would need mixed precision, which the ANE does not offer.

### 4.5 Things that did not help

These were tried and measured, none of them mattered, and none are in the final code:
- Squaring a pre-scaled difference before the variance (`diff * 0.125`), to avoid overflow.
- Computing the mean in two stages (block means, then a mean of those).
- `1 / sqrt(var + eps)` instead of `rsqrt`.
- The decomposed normalisation instead of native `layer_norm` (within 0.5 dB).
- A mean computed as a matmul with `1/n` weights, which was worse: NaN/inf in fp16.

## 5. Converting

```python
ep = torch.export.export(model, example).run_decompositions({})   # core-ATen graph coremltools accepts
ml = ct.convert(
    ep,
    inputs=[ct.TensorType("x_spec", shape=(1, 4, 2048, 336)),
            ct.TensorType("x_time", shape=(1, 2, 32, 10752))],
    outputs=[ct.TensorType("out_spec"), ct.TensorType("out_time")],
    convert_to="mlprogram",                        # ML Program, not the legacy NeuralNetwork format
    compute_precision=ct.precision.FLOAT16,        # the ANE is fp16-only (also the default for ML Programs)
    minimum_deployment_target=ct.target.macOS14,   # iOS 17 equivalent: spec v8, opset CoreML7
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
ml.save("htdemucs_drums_ane.mlpackage")
```

- **`torch.export`, not `torch.jit.trace`:** trace records shape arithmetic as tensor ops (section 4.1), export does not. This needs the rewrites above, because Dynamo cannot trace einops.
- **`run_decompositions({})`** lowers the program to the core-ATen op set that coremltools has converters for.
- **Shapes are fixed.** There are no flexible shapes, so the ANE program is one fixed graph for 7.8 s chunks. Example inputs must be distinct tensors (see the pitfall table).
- **Target:** iOS 17 / macOS 14 is what was built and tested. Older targets were not tried.
- **I/O types:** coremltools 9 declares the model's inputs and outputs as fp16 for this target. The Python `predict()` accepts fp32 numpy arrays and returns fp32. Feeding fp16 arrays instead made no measurable difference (369 vs 372 ms). About 4 ms of every call is array handling around the model (measured with a model that does nothing but has the same I/O).
- **Op mix of the result** (non-constant ops): `mul` 334, `add` 233, `slice_by_index` 192, `transpose` 116, `exp` 104, `real_div` 104, `conv` 100, `layer_norm` 100, `reshape` 95, `pad` 68, `batch_norm` 64, `linear` 60, `concat` 48, `split` 48, `matmul` 20, `softmax` 10. All are basic MIL ops, with nothing custom.
- **Time:** about 6 s for `torch.export` and 35 s for `ct.convert`.

## 6. Checking the result

### 6.1 Placement: is everything on the ANE?

```python
from coremltools.models.compute_plan import MLComputePlan
model = ct.models.MLModel("htdemucs_drums_ane.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
plan = MLComputePlan.load_from_path(model.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
for op in plan.model_structure.program.functions["main"].block.operations:
    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)   # None for constants
    cost = plan.get_estimated_cost_for_mlprogram_operation(op)          # .weight = relative cost
    # usage.preferred_compute_device, usage.supported_compute_devices
```

`ane_tools.compute_plan_summary()` wraps this and prints op counts and cost share per device, then each op type that is not on the ANE, labelled by whether the ANE *could* run it:
- **"not ANE-capable"** means the op in this form (its size, dtype or constants) cannot run there. This is the one to fix. It found the oversized time-branch tensors and the iSTFT's fp32 constant.
- **"ANE-capable"** but placed on the CPU means Core ML's cost model chose that, presumably for cheap ops that sit between CPU ops.

### 6.2 Numerics: is it right?

Three levels, cheapest first.

1. **Build-time asserts** (`check_torch_parity`): the ANE graph in PyTorch fp32 against the pure-AIE core on random input, 53.3 dB (spec) and 80.9 dB (time). The build stops below 50 dB. The only intended difference is the tanh GELU.
2. **Real-audio parity** (`parity_ane.py`, run at the end of every build) on three chunks of real music (`option_a_bfp16/parity_real_audio.npz`). Random noise hides precision problems, which is why the XDNA2 flow uses real audio too. For each chunk it reports SNR against the PyTorch fp32 model for the raw outputs (`spec`, `time`), against stock Demucs for the final waveform (`wav`), and `mix`, the pass/fail number: how far below the input mix the error sits. Pass is at least 40 dB. For the ANE drums model the worst case is 45.9 dB. A chunk with nearly silent drums can show a poor `wav` figure (5.8–12 dB) while `mix` is 70 dB, which is why `mix` is the criterion.
3. **Per-layer comparison** (`export_htdemucs_ane.py --debug-taps`): exports a copy that also returns every encoder, bottleneck and decoder tensor, then prints SNR against PyTorch fp32 at each point. Song 1 of the drums model:

```
enc0 55.5   enc1 49.2   enc2 47.4   enc3 49.8        out_spec 40.2
tenc0 58.9  tenc1 53.6  tenc2 53.5  tenc3 53.3       out_time 57.3
bottleneck 49.4   bottleneck_t 46.7
dec0 50.1   dec1 52.3   dec2 51.9
tdec0 49.9  tdec1 49.8  tdec2 50.0
```

   It should stay flat and high, with no cliff. Before the 4.4 fix, the same kind of table decayed steadily, from about 44 dB at `enc0` to about 28 dB at `enc3` (measured on song 2, so not directly comparable with the numbers above).

### 6.3 Routing: is the ANE really doing it?

The compute plan is a prediction. The runtime check is `measure_ane_usage.py`, which samples the SoC power rails with [`macmon`](https://github.com/vladkens/macmon) (`brew install macmon`, no sudo) while the model loops, against an idle baseline:

| Compute units | ANE | GPU | CPU | Time per chunk |
| :--- | :--- | :--- | :--- | ---: |
| `CPU_AND_NE` (default) | +2.5 W | within noise | within noise | 366 ms |
| `CPU_ONLY` (contrast) | 0 | within noise | +5.5 W | 503 ms |

The Python process itself spends only 2 to 4 ms of CPU time per chunk on the ANE. The same fused program on the CPU spent about 1.3 s of CPU time per chunk.

One caution when reproducing it: the ANE is shared. Another process using it at the same time roughly halves the speed (750 ms chunks instead of 390 ms were seen). `measure_ane_usage.py` warns when ANE power is not zero at baseline.

## 7. Loading: compile once

Loading `ct.models.MLModel("x.mlpackage")` costs about 25 s every time. The compile of the package takes only 0.3 s. The rest is Core ML building the ANE program, which the OS caches keyed by the path of the compiled model, and `MLModel(mlpackage)` compiles into a fresh temporary directory on each launch, so the cache never hits. `ane_tools.load_coreml()` compiles once into `ane/models/compiled/<name>_<build time>.mlmodelc` and loads that fixed path with `ct.models.CompiledMLModel`. After the first launch, loads take about 0.1 s, even from a new process. The build time is part of the folder name, so a rebuilt model never reuses a stale ANE program. The cache can evict entries: after many extra programs had been built during profiling, a model that had been cached needed the full 25 s again. An app should expect a slow first load after an OS update and on first install.

## 8. The fused model

`htdemucs_<stem>_fused_ane.mlpackage` adds the STFT and iSTFT to the same program (`dsp_ane.py`), so the spectrogram never leaves the ANE. The input is the reflect-padded mix, the output the spectral branch as waveform blocks, and the CPU only pads, crops and adds the two branches (about 1 ms). Design decisions:

- **A DFT as one dense matmul.** The ANE has no FFT, but 336 frames x 4096 x 4096 is only about 11 GMAC. `STFTANE` builds each frame from four slices of a `[343, 1024]` block view (frame *f* is blocks *f* to *f+3*), so no dimension exceeds 16,384, then multiplies by a windowed `[4096, 4096]` matrix whose columns are the real and imaginary parts. `ISTFTANE` multiplies by the inverse matrix and overlap-adds with four shifted `F.pad` adds.
- **Normalisation inside.** The mean and standard deviation of the spectrogram are computed in the program and reused by the iSTFT, as in `htdemucs_dsp.py`.
- **Reflect padding and the final crop stay on the CPU.** Padding is a data-dependent edge copy on a vector wider than 16,384, and the output crop starts at sample 3,584, which is not a multiple of the 1,024 block size, so on the ANE it would be an unaligned slice of a wide tensor.
- **Accuracy:** the fp32 STFT and iSTFT match numpy to about 118 dB. On the ANE the fused model scores 46.0 dB worst case, within 0.1 dB of the split pipeline.
- **Cost:** two `4096 x 4096` fp16 matrices add 65 MB (87 MB becomes 152 MB) and about 4 s of cold-load time. In exchange the CPU work per chunk drops from about 39 ms to about 1 ms.

## 9. Pitfalls table

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `TypeError: only 0-dimensional arrays can be converted to Python scalars` in coremltools `_int` | Shape arithmetic in `forward()` recorded as `aten::Int` by tracing | Use Python constants; export with `torch.export` |
| Dynamo error in einops, or "does not know how to trace method `_sa_block`" | Dynamo cannot trace einops or `nn.TransformerEncoderLayer` helpers | Inline the logic into `forward()` |
| 36% of estimated cost on the CPU, all in the time branch | Tensor dimension over 16,384 | Fold into rows with halo exchange |
| Two GELU ops still on the CPU after folding | Upsampling re-widened the rows to 21,504 and 43,008 | Emit 4R rows of width N |
| Everything on the ANE but 35.5 dB, not 40 dB | Sigmoid/GELU/GLU after a conv lose 15 to 22 dB in the ANE's fused path | Write them with `exp` and divide |
| `mul` and two `cast` ops on the CPU, "not ANE-capable" (iSTFT) | A constant of about 1e8 (1 / window sum at the edges) does not fit fp16, so coremltools keeps the op in fp32 | Clamp it to 100; only cropped-away samples are affected |
| iSTFT 25 dB worse than expected | `torch.export` merges example inputs that are the same tensor object, which tied `mean_s` to `std_s` | Pass distinct tensors (zeros and ones) |
| 25 s start-up on every launch | `MLModel("x.mlpackage")` recompiles into a new temp path and the ANE cache is path-keyed | `load_coreml()` (compile once) |
| Chunks randomly twice as slow | Another process is using the ANE | Check ANE power at baseline (`measure_ane_usage.py` warns) |
| `ct.convert("model.onnx")` -> "Unable to determine the type of the model" | coremltools has no ONNX frontend | Start from the PyTorch model |

## 10. Reusing the recipe on another model

1. Start from PyTorch. Remove shape arithmetic, einops and anything Dynamo cannot trace, and export with `torch.export`.
2. Convert early, and read the compute plan before optimising anything. The "not ANE-capable" list with tensor shapes tells you what to fix.
3. Check every dimension against 16,384 and every constant against the fp16 range (about 6e-8 to 65,504).
4. Measure accuracy per op on the ANE, not just end to end. A good end-to-end number can hide an op that loses 20 dB, and a bad one is hard to localise without per-layer taps. `probe_precision.py` is the template.
5. Judge accuracy on real audio, with a metric tied to what you care about.
6. Confirm placement with power, and re-check with the GPU idle and nothing else using the ANE.
7. Compile once and load the compiled model by path.

## 11. Reproducing and where things live

```bash
.venv/bin/python ane/export_htdemucs_ane.py                  # build + placement audit + real-audio parity
.venv/bin/python ane/export_htdemucs_ane.py --debug-taps     # per-layer SNR table
.venv/bin/python ane/probe_placement.py                      # the 16,384 table (section 4.2)
.venv/bin/python ane/probe_precision.py                      # the activation table (section 4.4)
.venv/bin/python ane/parity_ane.py                           # accuracy only, on an existing model
.venv/bin/python ane/measure_ane_usage.py                    # CPU/GPU/ANE routing by power
```

| File | Role |
| :--- | :--- |
| `ane_modules.py` | all the ANE rewrites; `build_ane_model()`, `select_source()` |
| `dsp_ane.py` | STFT / iSTFT / fused pipeline modules |
| `export_htdemucs_ane.py` | torch.export -> coremltools -> audit -> parity |
| `ane_tools.py` | `compute_plan_summary`, `load_coreml` (compiled-model cache), reference-data helpers |
| `parity_ane.py`, `measure_ane_usage.py` | accuracy and routing checks |
| `probe_placement.py`, `probe_precision.py` | the two experiments that set the design rules |

## 12. What is not verified

- **One chip and OS.** The 16,384 cap and the small-tensor placement were measured on an M1 under macOS 27.2. Other chips and OS versions can behave differently.
- **The fused-activation explanation** in 4.4 is an inference from measurements, not an observation of the compiler.
- **Not run on an iPhone.** The package targets iOS 17 and uses only standard MIL ops, but no iOS device was available, so memory and placement there are untested.
- **The XDNA2 rewrites** (dilated convs, ConvTranspose) were not ablated on the ANE.
- **Other stems.** Drums and vocals were built and checked. Bass and other are supported by `--source` but not built.
- **The tanh-GELU** deviates from the original GELU by about 61 dB. This was accepted on purpose, because it is far below the fp16 noise.
- **The scripts behind the early stage numbers** in 4.2 (64%, 99.7%) were throwaway and are not in the repo. The final numbers are reproducible with the commands above.
