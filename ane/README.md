# HTDemucs Drums on the Apple Neural Engine

The Apple-side version of `option_a_bfp16/`. It takes the same HTDemucs (`f7e0c4bc`) drums-only model and builds it for the Apple Neural Engine (ANE). The output is a **Core ML ML Program** (`.mlpackage`) plus an **ONNX** copy that ONNX Runtime's CoreML execution provider can run.

Measured on an Apple M1 (macOS 27.2):

| | Ops on ANE | Model latency / 7.8 s chunk | Worst "error below mix" (real audio) |
| :--- | :--- | :--- | :--- |
| `htdemucs_drums_ane.mlpackage` (Core ML, `CPU_AND_NE`) | **1696 / 1696 (100%)** | **~0.44–0.50 s (~16–18x RT)** | **45.9 dB** (PASS, ≥ 40) |
| `htdemucs_drums_ane_fp16.onnx` (ORT CoreML EP) | 1782 / 1782 nodes, 1 partition | ~0.38 s (~20x RT) | 46.0 dB (PASS) |
| `htdemucs_drums_ane.onnx` fp32 (ORT CoreML EP) | runs on CPU/GPU, not on the ANE | ~1.2 s | 64.9 dB |

End to end, including STFT/iSTFT and overlap-add, a 3-minute track separates in 17.3 s (**10.4x real time**). On top of that, each run of the script spends about 27 s loading the model, which is a one-time cost per run. For comparison, the XDNA2 pipelined runner does about 2.1x.

### Measured device usage (`measure_ane_usage.py`, macmon power rails, M1)

| Compute units | ANE power | Extra GPU power | Extra CPU power | Model latency |
| :--- | :--- | :--- | :--- | :--- |
| `CPU_AND_NE` (default) | **+2.0 W** | none | none | 497 ms |
| `ALL` | +2.0 W | none | none | 519 ms |
| `CPU_AND_GPU` | 0 | +3.5 W | none | 569 ms |
| `CPU_ONLY` | 0 | none | +4.2 W | 720 ms |

With `CPU_AND_NE`, the ANE draws power and the GPU and CPU stay at their idle baseline, so the compute runs on the ANE. `ALL` also chooses the ANE. These numbers were taken while a Blender GPU render was running in the background (GPU ~80% busy at idle). That load slows everything slightly, and the GPU row is pessimistic.

## Quick start

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r ane/requirements.txt

# Build: parity check -> Core ML -> ANE audit -> (ONNX) -> real-audio parity on the ANE
.venv/bin/python ane/export_htdemucs_ane.py --onnx

# Separate drums
.venv/bin/python ane/separate_drums_ane.py --single-file song.wav
.venv/bin/python ane/separate_drums_ane.py --input-dir ~/Music/folder --output-dir ~/Music/folder/drums
.venv/bin/python ane/separate_drums_ane.py --single-file song.wav --onnx ane/models/htdemucs_drums_ane_fp16.onnx

# Parity and latency only (any model / compute unit)
.venv/bin/python ane/parity_ane.py --units CPU_AND_NE
.venv/bin/python ane/parity_ane.py --onnx ane/models/htdemucs_drums_ane_fp16.onnx
```

The parity check uses `option_a_bfp16/parity_real_audio.npz`, the same reference and metrics as the XDNA2 flow. To rebuild it on the Mac, run `option_a_bfp16/parity_real_audio.py make-ref --song-dir <folder>`.

## Files

| File | Purpose |
| :--- | :--- |
| `ane_modules.py` | ANE rewrite of the pure-AIE core (`HTDemucsDrumsANE`) |
| `export_htdemucs_ane.py` | Build script: Core ML `.mlpackage` and optional ONNX (fp32 and fp16) |
| `ane_tools.py` | Shared helpers: SNR, I/O folding, **Core ML compute-plan audit** (per-op device placement) |
| `parity_ane.py` | Real-audio parity and latency check for Core ML or ONNX |
| `measure_ane_usage.py` | Checks CPU/GPU/ANE usage by measuring power while the model runs (`brew install macmon`, no sudo), plus a latency breakdown |
| `separate_drums_ane.py` | Batch runner; CPU STFT/iSTFT of neighbouring chunks overlaps the ANE run |

## Model I/O

The DSP code is the same as XDNA2 (`option_a_bfp16/htdemucs_dsp.py`). The one difference is that the waveform is **row-folded**:

- `x_spec`: `[1, 4, 2048, 336]` (unchanged)
- `x_time`: `[1, 2, 32, 10752]`, which is just `x_time.reshape(1, 2, 32, 10752)` of the usual `[1, 2, 1, 344064]`
- `out_spec`: `[1, 4, 2048, 336]`; `out_time`: `[1, 2, 32, 10752]`. Reshape back before `cpu_postproc_numpy`.

## What had to change for the ANE (and why)

There is no public ANE compiler. The only route onto the ANE is Core ML, and Core ML decides per op whether the ANE is used. `MLComputePlan` (wrapped in `ane_tools.compute_plan_summary`) reports where each op will land. It played the role that VAI-ML's partition summary played on XDNA2.

1. **Reuse the pure-AIE rewrites.** Dilated convs become dense kernels, stride-4 convs are unfolded into 1x1 convs, ConvTranspose becomes polyphase 1x1 convs, and everything is 4D. The ANE dislikes dilated convs, transposed convs and rank > 4 tensors, so these carried over directly. With nothing else changed, **64%** of the estimated cost ran on the ANE.
2. **Static shapes.** coremltools' TorchScript frontend cannot lower `H // 4`-style shape maths, and einops in the cross transformer has the same problem. All forwards were rewritten with constant shapes, positional embeddings were pre-baked, and the export now goes through `torch.export`.
3. **The 16384 dimension limit (the big one).** The ANE rejects any tensor dimension larger than 16384. The time branch runs at 344064, 86016 and 21504 samples, so every time-branch op fell back to the CPU. Fix: fold `[1, C, 1, L]` into `[1, C, R, L/R]`, which is a free reshape. Every conv along W then gets a **halo** of `p` columns copied from the neighbouring rows, so the result stays bit-exact. The polyphase upsampler emits 4x more rows instead of 4x wider rows. Result: **100% on the ANE**.
4. **fp16 precision.** The ANE computes in fp16 only. The first all-ANE build missed the bar at 35.5 dB below the mix (target 40).
   - Per-op microbenchmarks showed that most ops are at fp16 precision (~66–75 dB). The exception: when the ANE compiler **fuses sigmoid/GELU into the preceding conv**, it uses a low-precision approximation (45–52 dB per op).
   - Writing the activations as `x / (1 + exp(-z))` avoids the fused path. GLU uses this directly, and GELU uses the tanh approximation (~61 dB away from erf-GELU on real audio, well under the fp16 noise floor). This took the worst case from 35.5 dB to **45.9 dB**.
   - Things that made no difference: staged reductions, rsqrt vs `1/sqrt`, pre-scaled variance, decomposed norm vs native `layer_norm` (the native one is kept because it is simpler).
5. **ONNX / ORT CoreML EP.** ORT's CoreML EP only reaches the ANE if the graph is fp16, so the build also writes `_fp16.onnx`. `Neg` (unsupported by the EP) and `Gather` were removed so the whole graph is one CoreML partition. Before that change it was split into 25 partitions and ran at 1.3 s per chunk. The first session load compiles the model (~100 s). The result is cached in `models/ort_coreml_cache/`, which brings later loads down to about 27 s.

Use `export_htdemucs_ane.py --debug-taps` to print per-layer ANE-vs-FP32 SNR, which shows where the fp16 error builds up.

## Known limits

- The worst-case spec SNR (27.6 dB on song 3) comes from a chunk where the drums are nearly silent. The error there is still 70.8 dB below the mix.
- Core ML compiles the `.mlpackage` the first time it loads (~20 s); later loads are cached by the OS.
