# V2 Workflow: Compile and Run the Fixed Drums Model

This covers the v2 pipeline: the fixed FP32 export, split into an NPU **body** plus a small FP32 **CPU tail** (the last frequency decoder layer, `decoder.3`). Run every command from the repo root:

```powershell
cd C:\Users\zubair\Documents\GitHub\demucs-npu
```

Two Python environments are used:

| Alias | Path | Used for |
|---|---|---|
| `VENV` | `.venv\Scripts\python.exe` | PyTorch / Demucs / onnxsim / Quark (export, splitting, CPU checks) |
| `RAI` | `C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0-demucs\python.exe` | Ryzen AI NPU (compile, NPU checks, separating songs) |

Optional shortcuts for the current PowerShell window:

```powershell
$VENV = ".venv\Scripts\python.exe"
$RAI  = "C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0-demucs\python.exe"
```

---

## Files

| File | What it is |
|---|---|
| `option_a_bfp16\models\htdemucs_candidate_a_pure_aie_v2_simp.onnx` | Full FP32 model (fixed export) |
| `option_a_bfp16\models\htdemucs_candidate_a_pure_aie_v2_npu_body.onnx` | Everything except `decoder.3`; runs on the NPU |
| `option_a_bfp16\models\htdemucs_candidate_a_pure_aie_v2_cpu_tail.onnx` | `decoder.3` only (0.2 MB); runs on the CPU in FP32 |
| `option_a_bfp16\parity_real_audio.npz` | Reference outputs from PyTorch on 3 real song chunks |
| `option_a_bfp16\xdna2_cache_v2\v2_fp32_body\` | NPU compile cache for the body (created by step 3) |

---

## Steps 1–2: Build the models (already done)

You only need these again if you change the model code.

```powershell
# 1. Reference data from PyTorch (about 1 min)
& $VENV option_a_bfp16\parity_real_audio.py make-ref

# 2a. Export the FP32 model. This also checks ONNX against PyTorch on real audio
#     and must print "[SUCCESS] ONNX matches PyTorch on real audio" (about 10-15 min)
& $VENV option_a_bfp16\export_candidate_a_pure_aie.py

# 2b. Split it into the NPU body and the CPU tail (about 1 min)
& $VENV option_a_bfp16\split_cpu_tail.py --find-cut-in option_a_bfp16\models\htdemucs_candidate_a_pure_aie_v2_raw.onnx

# 2c. Optional sanity check on CPU: should PASS at about 100+ dB
& $VENV option_a_bfp16\parity_real_audio.py eval `
    --model option_a_bfp16\models\htdemucs_candidate_a_pure_aie_v2_npu_body.onnx `
    --tail  option_a_bfp16\models\htdemucs_candidate_a_pure_aie_v2_cpu_tail.onnx
```

---

## Step 3: Compile for the NPU (takes about 3–3.5 hours)

There is no separate compile command: the NPU compiler runs automatically the first time a session is created with a cache key that has no `.rai` file yet. The parity check below triggers the compile, then measures the result on the NPU in the same run.

> **This compile was already started on Nov 10 at 12:40 PM.** Check whether it is still running (see "Checking progress" below) before starting another one. Only run one compile at a time.

```powershell
$env:PYTHONUNBUFFERED = "1"
& $RAI -u option_a_bfp16\parity_real_audio.py eval --npu -v `
    --model     option_a_bfp16\models\htdemucs_candidate_a_pure_aie_v2_npu_body.onnx `
    --tail      option_a_bfp16\models\htdemucs_candidate_a_pure_aie_v2_cpu_tail.onnx `
    --cache-dir option_a_bfp16\xdna2_cache_v2 `
    --cache-key v2_fp32_body `
    2>&1 | Tee-Object -FilePath compile_v2_body.log
```

Leave the window open. The process mostly uses a single CPU core and about 2 GB of RAM while it compiles.

### Checking progress

```powershell
# Is a compile still running? (CPU_s keeps growing while it works)
Get-Process python | Where-Object { $_.CPU -gt 10 } |
    Select-Object Id, @{n='CPU_s';e={[int]$_.CPU}}, @{n='RAM_MB';e={[int]($_.WorkingSet64/1MB)}}, StartTime

# Has the cache been written? A large .rai file (hundreds of MB) means the compile finished.
Get-ChildItem option_a_bfp16\xdna2_cache_v2\v2_fp32_body -Recurse -Filter *.rai | Select-Object Name, Length, LastWriteTime
```

---

## Step 4: After compiling, check the NPU result

If step 3 ran to completion, its log already contains this. To re-run it (the session now loads from the cache in about 5–10 seconds):

```powershell
& $RAI option_a_bfp16\parity_real_audio.py eval --npu -v `
    --model     option_a_bfp16\models\htdemucs_candidate_a_pure_aie_v2_npu_body.onnx `
    --tail      option_a_bfp16\models\htdemucs_candidate_a_pure_aie_v2_cpu_tail.onnx `
    --cache-dir option_a_bfp16\xdna2_cache_v2 `
    --cache-key v2_fp32_body
```

How to read the output (all values are dB; higher is better):

| Metric | Old NPU model | Expected for v2 (from CPU simulation) |
|---|---|---|
| `spec` (frequency branch vs PyTorch) | 6.5–12 | about 41–47 |
| `time` (time branch vs PyTorch) | 6.5–40 | about 38–47 |
| `error below mix` | very low | 40 or more = **PASS** |

The `wav vs stock` number can look bad on chunks where the drums are almost silent (like the *FACE MY FEARS* chunk); judge by `error below mix` instead.

- **PASS:** go to step 5.
- **FAIL, but much better than the old model:** still worth listening to in step 5.
- **FAIL and close to the old numbers:** the NPU compiler is doing something worse than plain bf16 rounding. Save the log; the next step is compiling smaller pieces of the model to find which part breaks.

---

## Step 5: Separate songs with the v2 model

The pipelined runner takes the body, the tail, and the cache settings:

```powershell
# One song
& $RAI option_a_bfp16\batch_separate_drums_npu_pipelined.py `
    --model     option_a_bfp16\models\htdemucs_candidate_a_pure_aie_v2_npu_body.onnx `
    --tail      option_a_bfp16\models\htdemucs_candidate_a_pure_aie_v2_cpu_tail.onnx `
    --cache-dir option_a_bfp16\xdna2_cache_v2 `
    --cache-key v2_fp32_body `
    --single-file "C:\Users\zubair\Music\skrillex leaks\LEAKS\SKRILLEX - JUNGUNDRA (DEMO).wav" `
    --output-dir  "C:\Users\zubair\Music\skrillex leaks\LEAKS\npu_drums_v2"

# Whole folder: same command without --single-file
& $RAI option_a_bfp16\batch_separate_drums_npu_pipelined.py `
    --model     option_a_bfp16\models\htdemucs_candidate_a_pure_aie_v2_npu_body.onnx `
    --tail      option_a_bfp16\models\htdemucs_candidate_a_pure_aie_v2_cpu_tail.onnx `
    --cache-dir option_a_bfp16\xdna2_cache_v2 `
    --cache-key v2_fp32_body `
    --input-dir  "C:\Users\zubair\Music\skrillex leaks\LEAKS" `
    --output-dir "C:\Users\zubair\Music\skrillex leaks\LEAKS\npu_drums_v2"
```

Running it without `--model` / `--tail` / `--cache-*` still uses the old model, so you can A/B the old and new outputs.

Note: the sequential runner (`batch_separate_drums_npu.py`) has not been updated for the body/tail split. Use the pipelined runner for v2.

### Reference to compare against by ear

Stock Demucs on the CPU, using the same drums model (`htdemucs_ft` uses `f7e0c4bc` for drums). It's slow, but it's the "correct" output:

```powershell
& $VENV -m demucs -n htdemucs_ft --two-stems drums `
    -o "C:\Users\zubair\Music\skrillex leaks\LEAKS\cpu_reference" `
    "C:\Users\zubair\Music\skrillex leaks\LEAKS\SKRILLEX - JUNGUNDRA (DEMO).wav"
```

Compare `cpu_reference\htdemucs_ft\<song>\drums.wav` with `npu_drums_v2\<song> (Drums).wav`. Small differences are expected (stock Demucs also applies a random time shift); the constant artifacting should be gone.

---

## Troubleshooting

- **The compile restarts every time:** the `.rai` file was never written, usually because the compile was interrupted. Delete the incomplete cache folder and run step 3 again:
  ```powershell
  Remove-Item -Recurse -Force option_a_bfp16\xdna2_cache_v2\v2_fp32_body
  ```
- **You changed the model and re-exported or re-split:** the old cache no longer matches. Use a new `--cache-key` (for example `v2_fp32_body_b`) or delete the old cache folder, then compile again. Use the same `--cache-dir` / `--cache-key` in step 5.
- **Output looks wrong but parity passed:** run `--single-file` on a short song first, and compare against the CPU reference above.

---

## Optional: BFP16 quantized variant (not recommended)

Quantizing Conv/MatMul/Gemm with Quark BFP16 only reaches about 16–17 dB on the frequency branch in CPU simulation, worse than plain bf16, so it isn't the main path. To reproduce:

```powershell
& $VENV option_a_bfp16\quantize_pure_aie_bfp16.py          # writes ..._v2_bfp16_gemm.onnx
& $VENV option_a_bfp16\parity_real_audio.py eval --ort-opt none `
    --model option_a_bfp16\models\htdemucs_candidate_a_pure_aie_v2_bfp16_gemm.onnx
```

`--ort-opt none` is required: with ONNX Runtime graph optimizations on, the Quark BFP16 custom op returns NaN on CPU.
