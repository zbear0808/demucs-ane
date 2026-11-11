"""
Real-audio parity harness for the HTDemucs drums ONNX models.

Random-noise parity data hides precision problems (large reductions, BFP16
blocks, bf16 accumulation) that show up immediately on real music, so every
model in the export -> quantize -> NPU chain should be checked with this.

  1. Build reference data (needs torch + demucs, run from .venv):
       .venv\\Scripts\\python.exe option_a_bfp16\\parity_real_audio.py make-ref

  2. Evaluate any ONNX model on CPU (either env) or on the NPU (Ryzen AI env):
       python option_a_bfp16\\parity_real_audio.py eval --model option_a_bfp16\\models\\X.onnx
       python option_a_bfp16\\parity_real_audio.py eval --model X.onnx --npu --cache-key my_key

Metrics are SNR in dB against the PyTorch FP32 model:
  spec  - frequency-branch output (out_spec)
  time  - time-branch output (out_time)
  wav   - final separated drums chunk after iSTFT, vs stock Demucs forward()
  mix   - error level relative to the input mix (how far below the music the artifacts sit)
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(WORKSPACE_ROOT))
from htdemucs_dsp import SAMPLE_RATE, SEGMENT_LEN, cpu_postproc_numpy, cpu_preproc_numpy

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_REF = SCRIPT_DIR / "parity_real_audio.npz"
DEFAULT_SONG_DIR = r"C:\Users\zubair\Music\skrillex leaks\LEAKS"
TORCH_LIB = WORKSPACE_ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
QUARK_DLL = Path(
    r"C:\Users\zubair\AppData\Local\torch_extensions\torch_extensions\Cache\py312_cpu"
    r"\quark_custom_ops_stable_abi\quark_custom_ops_stable_abi.dll"
)
VAIP_CONFIG = r"C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0-demucs\Lib\site-packages\misc\vaip_config.json"

BANDS = [(0, 64), (64, 256), (256, 768), (768, 1536), (1536, 2048)]


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref.astype(np.float64)
    err = ref - est.astype(np.float64)
    if not np.isfinite(err).all():
        return float("nan")
    return float(10 * np.log10(np.sum(ref**2) / (np.sum(err**2) + 1e-30)))


def load_chunk(path: str, offset_sec: float) -> np.ndarray:
    cmd = [
        "ffmpeg", "-loglevel", "panic", "-ss", str(offset_sec), "-i", path,
        "-t", str(SEGMENT_LEN / SAMPLE_RATE + 0.1), "-ar", str(SAMPLE_RATE), "-ac", "2", "-f", "f32le", "pipe:1",
    ]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T
    if audio.shape[-1] < SEGMENT_LEN:
        raise ValueError(f"{path}: not enough audio after {offset_sec}s")
    audio = audio[:, :SEGMENT_LEN]
    return ((audio - audio.mean()) / (audio.std() + 1e-8)).astype(np.float32)


def make_ref(args):
    import torch
    from demucs.pretrained import get_model
    from option_a_bfp16.export_candidate_a_pure_aie import HTDemucsCandidateAPureAIECore

    songs = args.songs
    if not songs:
        exts = (".wav", ".mp3", ".flac", ".m4a")
        songs = [os.path.join(args.song_dir, f) for f in sorted(os.listdir(args.song_dir)) if f.lower().endswith(exts)]
        songs = songs[: args.num_songs]
    print(f"Building reference from {len(songs)} chunk(s) at {args.offset_sec}s:")

    stock = get_model("f7e0c4bc")
    stock = stock.models[0] if hasattr(stock, "models") else stock
    stock.eval()
    # 344064 is longer than the 343980-sample training segment; let stock forward() accept it.
    stock.use_train_segment = False
    core = HTDemucsCandidateAPureAIECore(stock).eval()

    keys = ["chunk", "x_spec", "x_time", "ref_spec", "ref_time", "stock_wav", "mean_s", "std_s", "mean_t", "std_t"]
    data = {k: [] for k in keys}
    for song in songs:
        print(f"  {os.path.basename(song)}")
        chunk = load_chunk(song, args.offset_sec)[None]
        x_spec, x_time, (mean_s, std_s, mean_t, std_t) = cpu_preproc_numpy(chunk)
        x_time = x_time.astype(np.float32)
        with torch.no_grad():
            ref_spec, ref_time = core(torch.from_numpy(x_spec), torch.from_numpy(x_time))
            stock_wav = stock(torch.from_numpy(chunk))[0, 0].numpy()  # source 0 = drums
        wav = cpu_postproc_numpy(ref_spec.numpy(), ref_time.numpy(), (mean_s, std_s, mean_t, std_t))[0]
        print(f"    PyTorch core + NumPy DSP vs stock Demucs: {snr_db(stock_wav, wav):.1f} dB")
        for k, v in zip(keys, [chunk[0], x_spec[0], x_time[0], ref_spec.numpy()[0], ref_time.numpy()[0],
                               stock_wav, mean_s, std_s, mean_t, std_t]):
            data[k].append(np.asarray(v, dtype=np.float32))

    np.savez(args.ref, songs=np.array([os.path.basename(s) for s in songs]), **{k: np.stack(v) for k, v in data.items()})
    print(f"Saved {args.ref} ({os.path.getsize(args.ref) / 2**20:.1f} MB)")


def make_session(args):
    import onnxruntime as ort

    if TORCH_LIB.exists():
        os.add_dll_directory(str(TORCH_LIB))
        os.environ["PATH"] = f"{TORCH_LIB};{os.environ.get('PATH', '')}"
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.log_severity_level = 3
    so.graph_optimization_level = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }[args.ort_opt]
    if QUARK_DLL.exists():
        so.register_custom_ops_library(str(QUARK_DLL))
    if args.npu:
        os.makedirs(args.cache_dir, exist_ok=True)
        return ort.InferenceSession(
            args.model, so,
            providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"config_file": args.vaip_config, "cacheDir": args.cache_dir, "cacheKey": args.cache_key}, {}],
        )
    return ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])


def evaluate(args):
    ref = np.load(args.ref)
    print(f"Model: {args.model} ({'NPU' if args.npu else 'CPU'})" + (f" + CPU tail {args.tail}" if args.tail else ""))
    t0 = time.perf_counter()
    sess = make_session(args)
    tail = None
    if args.tail:
        import onnxruntime as ort

        tail = ort.InferenceSession(args.tail, providers=["CPUExecutionProvider"])
    print(f"Session ready in {time.perf_counter() - t0:.1f}s")

    rows = []
    for i, song in enumerate(ref["songs"]):
        t0 = time.perf_counter()
        names = [o.name for o in sess.get_outputs()]
        outs = dict(zip(names, sess.run(None, {"x_spec": ref["x_spec"][i : i + 1], "x_time": ref["x_time"][i : i + 1]})))
        if tail is not None:
            outs["out_spec"] = tail.run(None, {t.name: outs[t.name] for t in tail.get_inputs()})[0]
        out_spec, out_time = outs["out_spec"], outs["out_time"]
        dt = time.perf_counter() - t0
        stats = tuple(ref[k][i] for k in ("mean_s", "std_s", "mean_t", "std_t"))
        wav = cpu_postproc_numpy(out_spec, out_time, stats)[0]
        r_spec, r_time = ref["ref_spec"][i : i + 1], ref["ref_time"][i : i + 1]
        row = {
            "spec": snr_db(r_spec, out_spec),
            "time": snr_db(r_time, out_time),
            "wav": snr_db(ref["stock_wav"][i], wav),
            # Error level relative to the input mix: how far below the music the artifacts sit.
            # More meaningful than "wav" on chunks where the drums are nearly silent.
            "mix": snr_db(ref["chunk"][i], ref["chunk"][i] + (wav - ref["stock_wav"][i])),
        }
        print(f"\n[{i + 1}] {song}  ({dt:.2f}s)  nan: {np.isnan(out_spec).mean():.3f}/{np.isnan(out_time).mean():.3f}")
        print(
            f"    spec {row['spec']:6.2f} dB | time {row['time']:6.2f} dB | wav vs stock {row['wav']:6.2f} dB"
            f" | error below mix {row['mix']:6.2f} dB"
        )
        if args.verbose:
            err = np.sqrt(((r_spec - out_spec) ** 2).mean(axis=(0, 1, 3)))
            sig = np.sqrt((r_spec**2).mean(axis=(0, 1, 3)))
            for lo, hi in BANDS:
                band = 20 * np.log10(sig[lo:hi].mean() / (err[lo:hi].mean() + 1e-30))
                print(f"      {lo * SAMPLE_RATE / 4096:6.0f}-{hi * SAMPLE_RATE / 4096:6.0f} Hz: {band:6.2f} dB")
            fe = np.sqrt(((r_spec - out_spec) ** 2).mean(axis=(0, 1, 2)))
            fs = np.sqrt((r_spec**2).mean(axis=(0, 1, 2)))
            fr = 20 * np.log10(fs / (fe + 1e-30))
            print(f"      per-frame spec SNR: min {fr.min():.1f} dB @ frame {fr.argmin()}, median {np.median(fr):.1f} dB")
        rows.append(row)

    worst = {k: min(r[k] for r in rows) for k in ("spec", "time", "wav", "mix")}
    print(
        f"\nWORST: spec {worst['spec']:.2f} dB | time {worst['time']:.2f} dB | wav {worst['wav']:.2f} dB"
        f" | error below mix {worst['mix']:.2f} dB"
    )
    ok = all(np.isfinite(v) for v in worst.values()) and worst["mix"] >= args.min_mix_snr
    print(f"{'PASS' if ok else 'FAIL'} (threshold: error below mix >= {args.min_mix_snr} dB)")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("make-ref", help="Build reference outputs with PyTorch (needs torch + demucs)")
    r.add_argument("--songs", nargs="*", default=None, help="Audio files to take one chunk from each")
    r.add_argument("--song-dir", default=DEFAULT_SONG_DIR)
    r.add_argument("--num-songs", type=int, default=3)
    r.add_argument("--offset-sec", type=float, default=40.0)
    r.add_argument("--ref", default=str(DEFAULT_REF))

    e = sub.add_parser("eval", help="Evaluate an ONNX model against the reference")
    e.add_argument("--model", required=True)
    e.add_argument("--tail", default=None, help="Optional CPU tail model (from split_cpu_tail.py) producing out_spec")
    e.add_argument("--ref", default=str(DEFAULT_REF))
    e.add_argument("--npu", action="store_true", help="Run with VitisAIExecutionProvider")
    e.add_argument("--cache-dir", default=str(SCRIPT_DIR / "xdna2_cache_pure_aie"))
    e.add_argument("--cache-key", default="candidate_a_pure_aie")
    e.add_argument("--vaip-config", default=VAIP_CONFIG)
    e.add_argument("--min-mix-snr", type=float, default=40.0, help="Required error level below the mix, in dB")
    e.add_argument(
        "--ort-opt", choices=["all", "basic", "none"], default="all",
        help="ORT graph optimization level. Use 'none' for Quark BFP16 CPU simulation: "
             "with optimizations on, the BFPQuantizeDequantize custom op graph returns NaN.",
    )
    e.add_argument("-v", "--verbose", action="store_true", help="Per-band and per-frame breakdown")

    args = p.parse_args()
    if args.cmd == "make-ref":
        make_ref(args)
        return 0
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
