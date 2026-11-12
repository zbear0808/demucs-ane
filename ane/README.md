# HTDemucs Drums & Vocals on the Apple Neural Engine

The Apple-side version of `option_a_bfp16/`. It takes the same HTDemucs (`f7e0c4bc`) model, pruned to a single stem (drums or vocals; bass and other also work via `--source`), and builds it for the Apple Neural Engine (ANE). The output is a **Core ML ML Program** (`.mlpackage`) plus an **ONNX** copy that ONNX Runtime's CoreML execution provider can run.

Measured on an Apple M1 (macOS 27.2), drums:

| | Ops on ANE | Model latency / 7.8 s chunk | Worst "error below mix" (real audio) |
| :--- | :--- | :--- | :--- |
| `htdemucs_drums_ane.mlpackage` (Core ML, `CPU_AND_NE`) | **1696 / 1696 (100%)** | **~0.44–0.50 s (~16–18x RT)** | **45.9 dB** (PASS, ≥ 40) |
| `htdemucs_drums_ane_fp16.onnx` (ORT CoreML EP) | 1782 / 1782 nodes, 1 partition | ~0.38 s (~20x RT) | 46.0 dB (PASS) |
| `htdemucs_drums_ane.onnx` fp32 (ORT CoreML EP) | runs on CPU/GPU, not on the ANE | ~1.2 s | 64.9 dB |

End to end, including STFT/iSTFT and overlap-add, a 3-minute track separates in 17.3 s (**10.4x real time**). On top of that, each run of the script spends about 27 s loading the model, which is a one-time cost per run. For comparison, the XDNA2 pipelined runner does about 2.1x.

### Vocals (`--source vocals`)

| | Ops on ANE | Model latency / 7.8 s chunk | Worst "error below mix" (real audio) |
| :--- | :--- | :--- | :--- |
| `htdemucs_vocals_ane.mlpackage` (Core ML, `CPU_AND_NE`) | **1696 / 1696 (100%)** | **367 ms (~21x RT)** | **61.8 dB** (PASS) |
| `htdemucs_vocals_ane_fp16.onnx` (ORT CoreML EP) | 1782 / 1782 nodes, 1 partition | 381 ms | 61.6 dB (PASS) |

The vocals model is the same graph with different final-layer weights, so it has the same size, op count and speed. A 3-minute track separates in 12.3 s (**14.6x real time**). That was measured with the GPU idle; the drums numbers above were taken during a Blender render. On ANE power: vocals draws +2.45 W on the ANE with no GPU and +0.13 W CPU. `CPU_ONLY` is 505 ms but costs +5.9 W of CPU power.

The vocals reference (`ane/parity_real_audio_vocals.npz`) is built automatically from the music chunks stored in the drums reference, so no audio files are needed. It is not committed (99 MB).

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

.venv/bin/python ane/export_htdemucs_ane.py --source vocals --onnx

# Separate
.venv/bin/python ane/separate_ane.py --single-file song.wav                  # drums
.venv/bin/python ane/separate_ane.py --single-file song.wav --source vocals
.venv/bin/python ane/separate_ane.py --input-dir ~/Music/folder --output-dir ~/Music/folder/drums
.venv/bin/python ane/separate_ane.py --single-file song.wav --onnx ane/models/htdemucs_drums_ane_fp16.onnx
.venv/bin/python ane/benchmark_pipeline.py                       # builds STFT/iSTFT models, ANE vs CPU timing
.venv/bin/python ane/separate_ane.py --single-file song.wav --dsp ane   # STFT/iSTFT on the ANE

# Parity and latency only (any model / compute unit)
.venv/bin/python ane/parity_ane.py --units CPU_AND_NE
.venv/bin/python ane/parity_ane.py --source vocals
.venv/bin/python ane/measure_ane_usage.py --source vocals
.venv/bin/python ane/parity_ane.py --onnx ane/models/htdemucs_drums_ane_fp16.onnx
```

The parity check uses `option_a_bfp16/parity_real_audio.npz`, the same reference and metrics as the XDNA2 flow. To rebuild it on the Mac, run `option_a_bfp16/parity_real_audio.py make-ref --song-dir <folder>`.

## Files

| File | Purpose |
| :--- | :--- |
| `ane_modules.py` | ANE rewrite of the pure-AIE core (`HTDemucsSourceANE`), plus `select_source()` to prune to one stem |
| `export_htdemucs_ane.py` | Build script: Core ML `.mlpackage` and optional ONNX (fp32 and fp16) |
| `ane_tools.py` | Shared helpers: SNR, I/O folding, **Core ML compute-plan audit** (per-op device placement) |
| `parity_ane.py` | Real-audio parity and latency check for Core ML or ONNX |
| `measure_ane_usage.py` | Checks CPU/GPU/ANE usage by measuring power while the model runs (`brew install macmon`, no sudo), plus a latency breakdown |
| `dsp_ane.py` | STFT / iSTFT written as matmul models for the ANE, plus a fused STFT + model + iSTFT pipeline |
| `benchmark_pipeline.py` | Per-stage ANE vs CPU timing for CPU-DSP, ANE-DSP and fused pipelines, with accuracy |
| `separate_ane.py` | Batch runner; CPU STFT/iSTFT of neighbouring chunks overlaps the ANE run |

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

## STFT / iSTFT on the ANE

The ANE has no FFT. At these sizes, though, a DFT is cheap as one dense matmul: 336 frames x 4096 x 4096 is about 11 GMAC, a few percent of the network. `dsp_ane.py` writes the STFT as framing (4 slices of a `[343, 1024]` block view, so no dimension exceeds 16384) followed by a matmul with a windowed DFT matrix. It writes the iSTFT as a matmul with an inverse-DFT matrix followed by a 4-way shifted-add overlap-add. In fp32 both match `htdemucs_dsp.py` to about 118 dB. Both run **100% on the ANE** (22 and 23 ops); macmon shows ANE power and no extra GPU power.

`benchmark_pipeline.py` results on M1, drums, 30 real-audio chunks, median per 7.8 s chunk:

| Stage | cpu-dsp (current) | ane-dsp | fused (1 Core ML call) |
| :--- | ---: | ---: | ---: |
| CPU: numpy STFT + normalise | 13.9 ms | - | - |
| CPU: reflect pad + time-branch stats | - | 1.0 ms | 1.0 ms |
| ANE: STFT model | - | 6.3 ms | (inside) |
| ANE: separation model | 367.4 ms | 365.5 ms | 386.3 ms (whole fused call) |
| ANE: iSTFT model | - | 5.9 ms | (inside) |
| CPU: numpy iSTFT + combine / crop + combine | 24.4 ms | 0.3 ms | 0.3 ms |
| **Total** | **405.7 ms** | **379.0 ms** | **387.6 ms** |
| ANE / CPU split | 91% / 9% | ~100% / <1% | ~100% / <1% |
| Worst error below mix | 45.9 dB | 46.0 dB | 46.0 dB |

- About **4 ms of each separation-model call is CPU work** inside `predict()`: converting the fp32 inputs and outputs to and from fp16, plus the copies. This was measured with a passthrough model that has the same I/O. The rest, about 363 ms, is ANE compute.
- On the ANE, the STFT is **6–7 ms vs 14–16 ms** in numpy and the iSTFT **6 ms vs 24 ms**. Accuracy is unchanged (ANE STFT 61 dB vs the numpy STFT).
- Per chunk in sequence, ANE DSP saves about 7% (406 to 379 ms). End to end the gain is much smaller: `separate_ane.py` already runs the numpy DSP on worker threads while the ANE works. A 3-minute track takes 12.3 s with CPU DSP and 12.1 s with ANE DSP (`--dsp ane`). The two outputs differ by an error 56.7 dB below the mix. The real win is CPU time freed up, not speed.
- Fused is not faster: the single program (386 ms) is about 9 ms slower than the three separate ANE calls (378 ms). The ~5 ms of copies it saves is less than whatever the bigger graph costs in scheduling.

### What stays off the ANE, and why

| Op | Where | Why |
| :--- | :--- | :--- |
| Reflect padding of the 344064-sample chunk | CPU | Data-dependent edge copies on a vector wider than the ANE's 16384 limit; ~1 ms in numpy |
| Final iSTFT crop | CPU | Starts 3584 samples into the buffer, not on a 1024-block boundary, so it is an unaligned slice of a >16384-wide tensor. In numpy it is a free view |
| Overlap-add across chunks | CPU | Spans separate model calls |
| Time-branch mean/std (two scalars) and the final add | CPU | Could run on the ANE, but it is ~1 ms and needs the unaligned crop anyway |
| fp32 <-> fp16 input/output casts | CPU | The ANE is fp16-only, so Core ML always casts fp32 I/O on the CPU (~4 ms per model call). Declaring fp16 model I/O would move this into numpy; it was not worth it here |
| *Fixed:* the first iSTFT's final `mul` + 2 casts | CPU -> ANE | The 1/window-sum constant reached ~1e8 at the buffer edges, which is outside fp16 range, so coremltools kept that op in fp32. fp32 ops can't run on the ANE. Clamping it to 100 (only cropped-away samples are affected) put it back on the ANE |
| *Fixed earlier:* the whole time branch | CPU -> ANE | Tensors 344064 / 86016 / 21504 wide broke the 16384 limit. Fixed by row folding with halo convs |

The GPU is never used in the default `CPU_AND_NE` setting.

## iPhone / iPad

The `.mlpackage` files run on iOS unchanged. This has not been tested on a device yet: this machine has no Xcode and no iPhone attached.

- **Format:** ML Program spec v8, opset `CoreML7`, meaning **iOS 17+ / iPadOS 17+** (same as macOS 14). Every op is a basic MIL op (`conv`, `matmul`, `layer_norm`, `softmax`, `exp`, `real_div`, ...). There are no custom layers and no Mac-only features. Inputs and outputs are fp32 multi-arrays.
- **ANE hardware:** the M1 Neural Engine is believed to be the same as the **A14** one (hollance/neural-engine `supported-devices.md`). So **iPhone 12 and newer** should place ops the same way, 100% on the ANE. A15 and later have the same or faster 16-core ANEs. A12/A13 (iPhone XS/11) have an older 8-core ANE; Core ML might push some ops to the GPU or CPU there.
- **Memory:** on the Mac, process footprint with the ANE is about 340 MB after loading and running (`footprint`), against 700 MB peak RSS on `CPU_ONLY`. The largest tensors are the 110 MB fp16 attention maps (`[8, 2688, 2688]`). That should fit on 4 GB+ iPhones, but check it on a device.
- **Size:** 87 MB per stem (fp16 weights).
- **Not portable:** the Python runners. An iOS app has to reimplement `htdemucs_dsp.py` (reflect padding, STFT/iSTFT with a 4096 Hann window and hop 1024) in Swift, e.g. with Accelerate/vDSP, along with the chunk overlap-add. The ONNX route also works on iOS (onnxruntime-objc/-swift with the CoreML EP, using the `_fp16.onnx` file).
- **How to confirm on a device:** open the `.mlpackage` in Xcode 15+, go to the *Performance* tab, run a report on a connected iPhone, and check the per-op compute unit and latency. Or use Instruments' Core ML template.
