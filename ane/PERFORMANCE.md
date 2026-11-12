# Where the time goes: ANE deployment options compared

This compares the three ways to run HTDemucs (one stem) on the Apple Neural Engine. All numbers are from an M1 (macOS 27.2), drums model. Vocals is the same graph with different weights and measures the same. Reproduce with:

```bash
.venv/bin/python ane/profile_variants.py            # sizes, load, memory, runtime, per-stage breakdown
.venv/bin/python ane/benchmark_pipeline.py          # runtime + accuracy per pipeline
```

| Name | What runs where | Files |
| :--- | :--- | :--- |
| **split** (default runner) | separation model on the ANE; STFT / iSTFT in numpy on the CPU | `htdemucs_drums_ane.mlpackage` |
| **split3** (`--dsp ane`) | separation model, STFT model and iSTFT model, all on the ANE (3 calls) | + `htdemucs_stft_ane`, `htdemucs_istft_ane` |
| **fused** ("full NPU") | one Core ML program: STFT -> separation -> iSTFT on the ANE | `htdemucs_drums_fused_ane.mlpackage` |

In all three, a little CPU work is left: reflect padding (~1 ms), the unaligned output crop (free), overlap-add between chunks, and audio decode/encode. See the README for why these stay on the CPU.

## Summary

| | split | split3 | fused |
| :--- | ---: | ---: | ---: |
| Size on disk (.mlpackage / compiled .mlmodelc) | **87 MB** / 87 MB | 152 MB / 151 MB | 152 MB / 151 MB |
| Cold load (first launch, ANE program built) | **23–25 s** | 27–28 s | 28–30 s |
| Warm load (cached) | **0.12 s** | 0.14 s | 0.14–0.17 s |
| Memory: footprint added by loading | +16–20 MB | +21 MB | +20–24 MB |
| Memory: peak process footprint while running | 326–331 MB | 348 MB | **276–279 MB** |
| Time per 7.8 s chunk, sequential (median, GPU idle) | 405–420 ms | 379–410 ms | 376–428 ms |
| CPU share of that time | ~9–10% | <1% | <1% |
| 3-minute track end to end (runner overlaps CPU and ANE) | 12.6 s | 12.1 s | - |
| Worst accuracy, error below mix | 45.9 dB | 46.0 dB | 46.0 dB |

How to read it:

- **Size.** split3 and fused are 65 MB bigger. That is the two 4096 x 4096 fp16 DFT matrices (32 MB each) that replace the FFT. The separation weights are the same 87 MB in all three.
- **Load.** Each variant has the same two load costs: a ~25 s one-time ANE build, then ~0.1 s loads after that. More models or more ops add a few seconds to the build: split3 builds three programs, and fused has 45 more ops in one program. See *Model loading* below; the cache is what turns 25 s into 0.1 s.
- **Memory.** fused has the lowest peak because the spectrogram never comes back to the CPU as an fp32 array. split and split3 each hold ~11 MB fp32 spectrogram copies on both sides of a call, plus numpy's FFT buffers in split.
- **Runtime.** The three come within ~5–10% of each other. The run-to-run spread (thermal state, background load) is about as large as the differences, so there is no stable winner between split3 and fused. Both beat split by removing ~38 ms of numpy DSP per chunk. In a full song, `separate_ane.py` already runs the numpy DSP on worker threads while the ANE works, so that saving mostly disappears: 12.6 s vs 12.1 s for 3 minutes of audio.

**Recommendation:** keep **split** as the default (smallest, fastest to build, proven). Use **fused** when memory matters, or when the caller cannot spare CPU threads for the DSP, such as an iOS app. Skip split3; it has fused's size without fused's memory win.

## Where the time goes, per variant

One 7.8 s chunk, median of 30 real-audio chunks, from the cleanest run (GPU idle):

### split (default)

```
CPU   numpy STFT + normalise          14 ms   ███
ANE   separation model call          368 ms   ████████████████████████████████████████████████████████████████████████
        of which CPU-side I/O          4 ms     (moving the arrays into and out of Core ML)
CPU   numpy iSTFT + combine           24 ms   █████
----------------------------------------------
      total                          407 ms   = 19.2x real time   (ANE ~89%, CPU ~11%)
```

### split3 (`--dsp ane`)

```
CPU   reflect pad + time-branch stats  1 ms
ANE   STFT model call               7-10 ms   ██    (~0.6 ms of it is array I/O)
ANE   separation model call          389 ms   ████████████████████████████████████████████████████████████████████████████
ANE   iSTFT model call              6-10 ms   ██
CPU   crop + combine                 0.3 ms
----------------------------------------------
      total                          410 ms   = 19.1x real time   (ANE ~98%, CPU-side I/O ~1.5%, numpy <0.5%)
```

### fused ("full NPU")

```
CPU   reflect pad + time-branch stats  1 ms
ANE   STFT + separation + iSTFT      381 ms   ███████████████████████████████████████████████████████████████████████████
CPU   crop + combine                 0.3 ms
----------------------------------------------
      total                          382 ms   = 20.4x real time   (ANE ~99%)
```

The ANE's STFT and iSTFT each cost 6–10 ms, against 14–17 ms and 24–29 ms in numpy. The numpy iSTFT is slow because its overlap-add is a Python loop over 340 frames.

## Where the time goes inside the separation model (~370 ms)

The same in all three variants. Each stage was measured as its own ANE program with small inputs and outputs. The decoders were measured by subtraction: the whole model minus the whole model with that decoder removed. The stage programs add up to ~435 ms against 367 ms for the whole model, because every separate program pays its own start-up and sync, so the table uses shares:

| Stage | Measured | Share | What it is |
| :--- | ---: | ---: | :--- |
| Cross-transformer (bottleneck) | 158 ms | **36%** | 5 layers x 2 branches; attention over 2688 freq tokens and 1344 time tokens |
| Frequency decoder (4 layers) | 101 ms | 23% | polyphase upsampling convs + DConv blocks on up-to-[48, 2048, 336] tensors |
| Time encoder (4 layers) | 72 ms | 16% | row-folded 1D convs on 344064 -> 1344 samples |
| Frequency encoder (4 layers) | 54 ms | 12% | unfold + 1x1 convs + DConv |
| Time decoder (4 layers) | 51 ms | 12% | polyphase upsampling back to 344064 samples |

Core ML's own cost estimate (`MLComputePlan`) shows *which kind* of work dominates:

| Op type | Est. share | Count |
| :--- | ---: | ---: |
| `add` | 15.0% | 233 |
| `mul` | 14.0% | 334 |
| `transpose` | 12.8% | 116 |
| `real_div` | 6.6% | 104 |
| `conv` | 6.5% | 100 |
| `matmul` | 6.5% | 20 |
| `softmax` | 6.2% | 10 |
| `linear` | 5.7% | 60 |
| `slice_by_index` | 5.2% | 192 |
| `exp` | 4.4% | 104 |

What this means:

- **The model is limited by memory traffic, not maths.** Element-wise ops (add / mul / div / exp) and layout changes (transpose / slice) make up about 60% of the estimated cost. Convs and matmuls, the work the ANE is built for, are under 20%. HTDemucs runs many cheap ops over large activations, up to 8M values per tensor, and each element-wise op streams its whole tensor through memory once. That is also why a busy GPU, which shares the memory bandwidth, slows the ANE by ~25%.
- **The transformer is the most expensive single block** because its attention maps are big. Each layer builds an `[8, 2688, 2688]` score tensor (110 MB in fp16) for the frequency branch and a `[8, 2688, 1344]` one for the cross attention, then runs softmax over them.
- **Transposes come from layout conversions:** spectrogram to tokens and back at the bottleneck, the per-head reshapes in attention, the tap reordering in the unfold/polyphase convs, and the `layer_norm` over (C, W) in the frequency DConv blocks.
- **The `exp` / `real_div` ops are the precision-safe activations** (`x / (1 + exp(-z))`). They cost ~11% but buy ~10 dB of accuracy over the fused sigmoid/GELU path; see the README.

## Model loading, explained

What happens when a model loads:

1. **mlpackage -> mlmodelc compile:** about 0.3 s (protobuf to on-disk compiled form).
2. **ANE program build:** about **23–28 s**. Core ML (through the system's ANE compiler service) plans tiling, memory and scheduling for the whole graph. Bigger graphs take longer: split3 builds three programs, fused builds one with 45 extra ops.
3. **Weight upload / session setup:** about 0.1 s.

Step 2 is cached by the OS. Two details of that cache matter:

- **The cache is keyed by the compiled model's path.** `coremltools.models.MLModel("x.mlpackage")` compiles into a *new temporary directory* on every launch, so it pays the 25 s every time. That is why the scripts used to take ~25 s to start. `ane_tools.load_coreml()` now compiles once into `ane/models/compiled/<name>_<build-time>.mlmodelc` and loads that fixed path. After the first launch, loads take **~0.1 s, even from a new process**. Rebuilding a package changes the name, so a stale ANE program is never reused.
- **The cache is size-limited and LRU.** While profiling, after building many extra programs (stage sub-models and cold copies), the least recently used entry was evicted and its next load paid the 25 s again. In normal use, with 1–3 models, this doesn't happen. A host app should still expect a slow first load after an OS update, after cache eviction, and on first install (on iOS, Core ML builds the ANE program on device the first time).

The cache's survival across reboots was not tested.

ONNX Runtime's CoreML EP (`_fp16.onnx`) sits on top of the same mechanism. Its `ModelCacheDirectory` option keeps the converted model, which cut its load from ~106 s to ~27 s. The remaining ~25 s is most likely the same ANE build. ORT probably hands Core ML a new path each session; this was not verified.
