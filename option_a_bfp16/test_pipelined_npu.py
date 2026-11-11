import os
import sys
import time
import queue
import threading
import numpy as np
import onnxruntime as ort

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

torch_lib = r"C:\Users\zubair\Documents\GitHub\demucs-npu\.venv\Lib\site-packages\torch\lib"
os.add_dll_directory(torch_lib)
os.environ["PATH"] = f"{torch_lib};{os.environ.get('PATH', '')}"

quark_dll = r"C:\Users\zubair\AppData\Local\torch_extensions\torch_extensions\Cache\py312_cpu\quark_custom_ops_stable_abi\quark_custom_ops_stable_abi.dll"
cfg = r"C:\Users\zubair\miniconda3\envs\ryzen-ai-1.8.0-demucs\Lib\site-packages\misc\vaip_config.json"
model_path = r"option_a_bfp16\models\htdemucs_candidate_a_pure_aie_bfp16_clean_cpu.onnx"
cache_dir = r"option_a_bfp16\xdna2_cache_pure_aie"
cache_key = "candidate_a_pure_aie"

CHUNK_SAMPLES = 344064
CHUNK_DURATION = CHUNK_SAMPLES / 44100.0  # 7.80s

def main():
    print("=" * 76)
    print("Testing Continuous Pipelined NPU Execution (Double-Buffering)")
    print("=" * 76)

    so = ort.SessionOptions()
    so.log_severity_level = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_cpu_mem_arena = True
    so.enable_mem_pattern = True
    so.enable_mem_reuse = True
    so.intra_op_num_threads = 8
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.register_custom_ops_library(quark_dll)

    print("Loading NPU Session...")
    sess = ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"config_file": cfg, "cacheDir": cache_dir, "cacheKey": cache_key}, {}],
        disabled_optimizers=["NchwcTransformer"],
    )
    print("Session ready!")

    # Setup 2 ping-pong IOBinding buffers (Double Buffering)
    # While Buffer 0 is running on NPU, Buffer 1 is being prepared or read
    def create_io_pair():
        iob = sess.io_binding()
        inputs = {}
        outputs = {}
        for inp in sess.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            buf = np.zeros(shape, dtype=np.float32)
            ort_val = ort.OrtValue.ortvalue_from_numpy(buf, "cpu", 0)
            inputs[inp.name] = (buf, ort_val)
            iob.bind_ortvalue_input(inp.name, ort_val)
        for out in sess.get_outputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in out.shape]
            buf = np.zeros(shape, dtype=np.float32)
            ort_val = ort.OrtValue.ortvalue_from_numpy(buf, "cpu", 0)
            outputs[out.name] = (buf, ort_val)
            iob.bind_ortvalue_output(out.name, ort_val)
        return iob, inputs, outputs

    iob0, in0, out0 = create_io_pair()
    iob1, in1, out1 = create_io_pair()

    # Warmup
    print("Performing initial warmup...")
    sess.run_with_iobinding(iob0)
    print("Warmup done!")

    # Test back-to-back continuous execution of 6 chunks using alternating buffers
    # with ZERO sleep and overlapping preparation
    print("\nStarting 6 Continuous Back-to-Back Chunks on NPU...")
    iobs = [iob0, iob1]
    inputs_list = [in0, in1]
    
    t_start = time.perf_counter()
    chunk_times = []

    for i in range(6):
        buf_idx = i % 2
        curr_iob = iobs[buf_idx]
        curr_in = inputs_list[buf_idx]

        # In-place fill input buffer
        np.copyto(curr_in["x_spec"][0], np.random.randn(1, 4, 2048, 336).astype(np.float32))
        np.copyto(curr_in["x_time"][0], np.random.randn(1, 2, 1, 344064).astype(np.float32))

        t0 = time.perf_counter()
        sess.run_with_iobinding(curr_iob)
        dt = (time.perf_counter() - t0) * 1000.0
        chunk_times.append(dt)
        print(f"  Chunk {i+1}/6: {dt:7.2f} ms | Sustained NPU Active Time: {dt/1000:.2f}s")

    total_time = time.perf_counter() - t_start
    print("-" * 76)
    print(f"Total Wall-Clock Time: {total_time:.2f}s for {6 * CHUNK_DURATION:.1f}s of audio")
    print(f"Mean Chunk Latency   : {np.mean(chunk_times):.2f} ms")
    print(f"Effective Throughput : {(6 * CHUNK_DURATION) / total_time:.2f}x Real-Time Speed")
    print("=" * 76)

if __name__ == "__main__":
    main()

