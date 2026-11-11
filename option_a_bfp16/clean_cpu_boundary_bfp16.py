import json
import os
import sys
from collections import deque
import onnx

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# Exact boundary tensors of vaiml_par_6 from the .rai inspection
NPU_INPUTS = {
    "glu_14_DequantizeLinear_Output",
    "pad_6",
    "x_spec",
}
NPU_OUTPUTS = {
    "out_spec",
    "add_167_DequantizeLinear_Output",
}

RAI_PATH = r"option_a_bfp16\xdna2_cache_pure_aie\candidate_a_pure_aie\candidate_a_pure_aie.rai"


def get_rai_partition_nodes():
    """Extracts exact list of node output tensors inside vaiml_par_6 from .rai"""
    if not os.path.exists(RAI_PATH):
        return None
    with open(RAI_PATH, "rb") as f:
        data = f.read()
    idx = data.find(b'"metaDef"')
    if idx == -1:
        return None
    start = data.rfind(b'{', 0, idx)
    depth = 0
    end = start
    while end < len(data):
        c = data[end:end+1]
        if c == b'{': depth += 1
        elif c == b'}':
            depth -= 1
            if depth == 0:
                end += 1
                break
        end += 1
    j = json.loads(data[start:end].decode("ascii", errors="ignore"))
    return set(j["metaDef"][0]["nodes"])


def find_npu_and_cpu_nodes(model: onnx.ModelProto):
    """
    Partitions graph nodes into:
      - npu_nodes: All compute AND weight/bias BFPQuantizeDequantize nodes inside vaiml_par_6
      - cpu_nodes: All nodes outside vaiml_par_6 (tencoder stages 0-2 and tdecoder stages 1-3)
    """
    nodes = list(model.graph.node)
    producer_of = {}
    consumers_of = {}

    for idx, node in enumerate(nodes):
        for out_t in node.output:
            producer_of[out_t] = idx
        for in_t in node.input:
            consumers_of.setdefault(in_t, []).append(idx)

    # 1. Forward BFS from NPU_INPUTS, stopping at NPU_OUTPUTS
    forward_npu = set()
    queue = deque()
    for t in NPU_INPUTS:
        for c_idx in consumers_of.get(t, []):
            forward_npu.add(c_idx)
            queue.append(c_idx)

    while queue:
        curr = queue.popleft()
        for out_t in nodes[curr].output:
            if out_t in NPU_OUTPUTS:
                continue
            for nxt in consumers_of.get(out_t, []):
                if nxt not in forward_npu:
                    forward_npu.add(nxt)
                    queue.append(nxt)

    # 2. Backward BFS from NPU_OUTPUTS, stopping at NPU_INPUTS
    backward_npu = set()
    queue = deque()
    for t in NPU_OUTPUTS:
        if t in producer_of:
            p_idx = producer_of[t]
            backward_npu.add(p_idx)
            queue.append(p_idx)

    while queue:
        curr = queue.popleft()
        for in_t in nodes[curr].input:
            if in_t in NPU_INPUTS:
                continue
            if in_t in producer_of:
                prv = producer_of[in_t]
                if prv not in backward_npu:
                    backward_npu.add(prv)
                    queue.append(prv)

    # Core NPU compute nodes lie on the intersection of forward and backward cones
    core_npu = forward_npu & backward_npu

    # 3. Include all weight/bias/constant producer chains feeding core_npu
    #    (Walk backward from core_npu without crossing NPU_INPUTS)
    full_npu = set(core_npu)
    queue = deque(core_npu)
    while queue:
        curr = queue.popleft()
        for in_t in nodes[curr].input:
            if in_t in NPU_INPUTS:
                continue
            if in_t in producer_of:
                prv = producer_of[in_t]
                if prv not in full_npu:
                    full_npu.add(prv)
                    queue.append(prv)

    cpu_nodes = set(range(len(nodes))) - full_npu
    return full_npu, cpu_nodes


def strip_cpu_bfp_nodes(input_onnx: str, output_onnx: str):
    print(f"Loading {input_onnx}...")
    model = onnx.load(input_onnx)
    nodes = list(model.graph.node)

    rai_partition_nodes = get_rai_partition_nodes()
    if rai_partition_nodes is not None:
        print(f"[RAI Check] Loaded {len(rai_partition_nodes)} exact partition node outputs from candidate_a_pure_aie.rai.")

    npu_indices, cpu_indices = find_npu_and_cpu_nodes(model)

    # Cross check BFS with RAI if available
    if rai_partition_nodes is not None:
        bfs_npu_outputs = {out for i in npu_indices for out in nodes[i].output}
        overlap = bfs_npu_outputs & rai_partition_nodes
        print(f"[RAI Check] Overlap with BFS NPU nodes: {len(overlap)} / {len(rai_partition_nodes)}")
        
        # If there are any nodes recorded in RAI that weren't captured by BFS, add them
        for i, n in enumerate(nodes):
            if any(out in rai_partition_nodes for out in n.output):
                npu_indices.add(i)
        cpu_indices = set(range(len(nodes))) - npu_indices

    npu_bfp = sum(
        1 for i in npu_indices
        if nodes[i].domain == "com.amd.quark" and nodes[i].op_type == "BFPQuantizeDequantize"
    )
    cpu_bfp_indices = [
        i for i in cpu_indices
        if nodes[i].domain == "com.amd.quark" and nodes[i].op_type == "BFPQuantizeDequantize"
    ]

    print(f"Total Graph Nodes : {len(nodes)}")
    print(f"NPU Partition Nodes: {len(npu_indices)} (includes {npu_bfp} BFPQuantizeDequantize nodes)")
    print(f"CPU Host Nodes     : {len(cpu_indices)} (includes {len(cpu_bfp_indices)} BFPQuantizeDequantize nodes to strip)")

    producer_of = {out_t: idx for idx, n in enumerate(nodes) for out_t in n.output}
    graph_inputs = {inp.name for inp in model.graph.input}
    initializers = {init.name for init in model.graph.initializer}
    graph_outputs = {out.name for out in model.graph.output}

    indices_to_remove = set()

    for idx in cpu_bfp_indices:
        bfp_node = nodes[idx]
        src_tensor = bfp_node.input[0]
        dst_tensor = bfp_node.output[0]

        # Case A: If dst_tensor is an NPU boundary input (like glu_14_DequantizeLinear_Output)
        # or a graph output, keep the tensor name dst_tensor by having its producer emit dst_tensor directly.
        if (dst_tensor in NPU_INPUTS or dst_tensor in graph_outputs) and src_tensor in producer_of:
            prod_node = nodes[producer_of[src_tensor]]
            # Redirect any other consumers of src_tensor to dst_tensor
            for n in nodes:
                for k, in_t in enumerate(n.input):
                    if in_t == src_tensor:
                        n.input[k] = dst_tensor
            for k, out_t in enumerate(prod_node.output):
                if out_t == src_tensor:
                    prod_node.output[k] = dst_tensor
            indices_to_remove.add(idx)
        else:
            # Case B: Standard CPU activation or weight/bias initializer — rewire consumers of dst_tensor to src_tensor
            for n in nodes:
                for k, in_t in enumerate(n.input):
                    if in_t == dst_tensor:
                        n.input[k] = src_tensor
            indices_to_remove.add(idx)

    # Rebuild node list without the stripped CPU BFP nodes
    cleaned_nodes = [n for i, n in enumerate(nodes) if i not in indices_to_remove]
    del model.graph.node[:]
    model.graph.node.extend(cleaned_nodes)

    onnx.save(model, output_onnx)
    print(f"[OK] Stripped {len(indices_to_remove)} CPU BFPQuantizeDequantize nodes -> {output_onnx}")


if __name__ == "__main__":
    strip_cpu_bfp_nodes(
        "option_a_bfp16/models/htdemucs_candidate_a_pure_aie_bfp16.onnx",
        "option_a_bfp16/models/htdemucs_candidate_a_pure_aie_bfp16_clean_cpu.onnx",
    )

