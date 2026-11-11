# Option A (BFP16 Pure AIE) — Production NPU Drum Stem Separation

This directory contains the production-ready implementation of **HTDemucs Drums Separation on the AMD Ryzen AI NPU (XDNA 2 / "Strix Point")** using Block Floating Point 16 (BFP16) and Pure AIE Graph Lowering.

---

## 1. Quick Start

Ensure you are in the Ryzen AI environment (`ryzen-ai-1.8.0-demucs`):

### Option 1: Pipelined High-Throughput Runner (Recommended)
* **Speed:** **2.13x Real-Time** (~63s for a 2.25-minute song)
* **RAM:** **~890 – 1,000 MB**
* **Hardware:** 100% NPU compute duty cycle via dual-slot IOBinding interleaving.

```powershell
# Run batch separation on a folder:
C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0-demucs\python.exe option_a_bfp16\batch_separate_drums_npu_pipelined.py `
    --input-dir "C:\Users\zubair\Music\skrillex leaks\LEAKS" `
    --output-dir "C:\Users\zubair\Music\skrillex leaks\LEAKS\npu_drums_pipelined"

# Or process a single track:
C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0-demucs\python.exe option_a_bfp16\batch_separate_drums_npu_pipelined.py `
    --single-file "C:\Users\zubair\Music\skrillex leaks\LEAKS\SKRILLEX - QUEST VAULT.wav"
```

### Option 2: Sequential Low-RAM Runner
* **Speed:** **1.65x Real-Time** (~81s for a 2.25-minute song)
* **RAM:** **~560 – 585 MB** (Absolute minimum memory footprint)
* **Hardware:** Single-threaded back-to-back execution.

```powershell
C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0-demucs\python.exe option_a_bfp16\batch_separate_drums_npu.py `
    --input-dir "C:\Users\zubair\Music\skrillex leaks\LEAKS" `
    --output-dir "C:\Users\zubair\Music\skrillex leaks\LEAKS\npu_drums"
```

---

## 2. Execution Profiles & Benchmark Comparison

Tested on `SKRILLEX - QUEST VAULT.wav` (134.4s audio @ 44.1 kHz stereo, 23 overlapping chunks of 7.8s):

| Metric | Sequential Runner (`batch_separate_drums_npu.py`) | Pipelined Runner (`batch_separate_drums_npu_pipelined.py`) |
| :--- | :--- | :--- |
| **Separation Time** | 81.7s | **63.0s (22.9% faster)** |
| **Real-Time Factor (RTF)** | 0.608x (1.65x Real-Time) | **0.469x (2.13x Real-Time)** |
| **Starting RAM (Chunk 1)** | 563.8 MB | 894.4 MB |
| **Peak Resident RAM** | **585.4 MB** | 1,001.1 MB |
| **IOBinding Slots** | 1 slot (Sequential) | 2 slots (Ping-pong hardware interleaved) |
| **Pre/Post Processing** | Synchronous (blocking) | Asynchronous Streaming (Non-blocking) |
| **NPU Duty Cycle** | ~65% (1.35s CPU idle gap per chunk) | **~95% (Interleaved zero-wait dispatch)** |

---

## 3. Architecture & Runtime Anatomy

### The "CPU Sandwich"
Although 94.1% of HTDemucs operators (3,056 nodes, 361 GOPs) execute natively on the XDNA 2 AIE tile array, the ONNX model is composed of three consecutive stages:

```text
Full Chunk Execution (~3.8s in a single thread):
┌───────────────────────────┬───────────────────────────────────────────┬───────────────────────────┐
│       CPU tencoder        │           XDNA 2 AIE Tile Array           │       CPU tdecoder        │
│      (650 ms, Zen 5)      │          (2,480 ms, 361.0 GOPs)           │      (700 ms, Zen 5)      │
└───────────────────────────┴───────────────────────────────────────────┴───────────────────────────┘
◄───────── CPU ─────────────►◄────────────── NPU 100% ──────────────────►◄───────── CPU ─────────────►
```

1. **`tencoder` (CPU, ~650 ms):** Four 1D downsampling convolutions executed via `CPUExecutionProvider`.
2. **`vaiml_par_6` (NPU, ~2,480 ms):** The core 3,056-node graph (2D multi-scale Convolutions, Cross-Domain Transformers, BFP16 GEMMs, BiLSTMs) executed on the physical AIE tile array.
3. **`tdecoder` (CPU, ~700 ms):** Four 1D transpose upsampling convolutions executed via `CPUExecutionProvider`.

### Why Task Manager Looks "Spiky"
In a single-threaded loop, Chunk $N+1$ cannot enter the NPU until Chunk $N$'s `tdecoder` (700 ms) and Chunk $N+1$'s `tencoder` (650 ms) finish. This creates a **1.35s CPU idle gap** between NPU executions. Combined with Windows Task Manager's 1-second rolling average sampling window, this produces visible spikes.

### How the Dual-Slot Pipeline Achieves 100% Duty Cycle
The pipelined runner launches two worker threads (`Slot 0` and `Slot 1`) phase-shifted by 1.2s:
* While **Slot 0** is executing `tdecoder` and post-processing, **Slot 1** is already executing on the NPU.
* While **Slot 1** is executing `tdecoder`, **Slot 0** has already begun executing on the NPU.
* This interleaving eliminates the 1.35s idle gap and raises real-time throughput from 1.65x to **2.13x**.

---

## 4. Memory Optimization Deep Dive

### Where Did the Initial 3.5 GB RAM Usage Go?
Standard ONNX Runtime execution with Vitis AI EP hoards **3.5 GB+** of physical RAM because:
1. `ort.InferenceSession()` loads `candidate_a_pure_aie.rai` (889 MB) into process heap.
2. The ONNX Protobuf heap parses 173 MB of model structure.
3. Host-side staging copies of `ctrl.elf` (188 MB) and `wts32.bin` (170 MB) are kept in virtual memory.
4. `quark_custom_ops_stable_abi.dll` pulls `torch_cpu.dll` (~400 MB) into virtual address space.

Once XRT allocates its physical page-locked DMA buffers (`amdnpu.sys`), **those initialization pages are never accessed again**.

#### The Trimming Solution (`trim_working_set`)
We call Windows PSAPI `EmptyWorkingSet()` immediately after the hardware warmup pass:
```python
def trim_working_set():
    gc.collect()
    h_proc = kernel32.GetCurrentProcess()
    psapi.EmptyWorkingSet(h_proc)
```
* Windows immediately evicts the ~1.8 GB of dead `.rai`, Protobuf, and unused DLL pages from physical RAM.
* The active XRT DMA buffers cannot be paged out because `amdnpu.sys` pins them in physical RAM.
* During inference, only active CPU pages fault back in, maintaining a low resident footprint.

### Why Pipelined Uses ~420 MB More than Sequential
Comparing 1,001 MB (Pipelined) vs 585 MB (Sequential):
1. **Dual ORT Execution Contexts (~390–420 MB):** In ONNX Runtime, each concurrent thread calling `sess.run_with_iobinding()` allocates independent scratchpad buffers for CPU `tencoder` and `tdecoder` intermediate tensors (~210 MB per thread).
2. **Pinned IOBinding Slot Buffers (~28 MB):** Two pre-allocated, 64-byte aligned hardware buffer slots are kept pinned in RAM.

### Streaming vs ThreadPool Futures (The 250 MB Fix)
In early versions, `ThreadPoolExecutor` held onto `Future` objects across the entire track, trapping ~250 MB of completed float32 spectrogram tensors in memory. We replaced this with a **direct streaming producer**:
* Preprocessing computes exactly 1 chunk ahead into `prep_queue(maxsize=1)`.
* Postprocessing immediately synthesizes iSTFT, accumulates overlap-add, and calls `del spec, time` to reclaim memory on the fly.

---

## 5. Command-Line Arguments Reference

### `batch_separate_drums_npu_pipelined.py`
```text
--input-dir STR          Input folder containing audio files (default: C:\Users\zubair\Music\skrillex leaks\LEAKS)
--output-dir STR         Output directory for stems (default: ...\npu_drums_pipelined)
--single-file STR        Process a single track instead of entire folder
--lookahead-depth INT    Max chunks buffered in prep queue (default: 1 for minimal RAM)
--num-slots INT          Number of alternating IOBinding hardware slots (default: 2)
--prep-workers INT       Worker threads for CPU STFT lookahead (default: 4)
--post-workers INT       Worker threads for CPU iSTFT synthesis (default: 2)
--overlap FLOAT          Chunk overlap ratio (default: 0.25 for Demucs)
```

---

## 6. Model & Hardware Cache Files

All model and cache assets are pre-compiled and self-contained:
* **Hardware ONNX Graph:** `models/htdemucs_candidate_a_pure_aie_bfp16_clean_cpu.onnx` (172.70 MB)
* **XDNA 2 Hardware Cache:** `xdna2_cache_pure_aie/candidate_a_pure_aie/candidate_a_pure_aie.rai` (889 MB)
* **Custom Ops Library:** `quark_custom_ops_stable_abi.dll` (Registered for dynamic dispatch)
* **Vitis AI Config:** `misc/vaip_config.json`

