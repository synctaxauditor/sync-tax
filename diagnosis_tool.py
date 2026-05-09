import argparse
import gc
import json
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pickle
import re
import sys
from functools import partial
from Graph import *
from cpu_event_chunking import chunk_all
import heapq
from collections import Counter
from collections import defaultdict
import math
from cycler import cycler

sys.setrecursionlimit(100000)

THRESHOLD = 3 # microseconds
COLORS2HEX = {
    "petrol":     "#264653",
    "gold":       "#E9C46A",
    "teal":       "#2A9D8F",
    "sand":       "#F4A261",
    "terracotta": "#E76F51",
}

def parse_log(input_path, exclude_pre_profiler=False):
    with open(input_path, "r") as f:
        data = json.load(f)
    f.close()
    metadata = data.get("distributedInfo", None)
    pg_config = {}
    for pg in metadata["pg_config"]:
        pg_config[pg["pg_name"]] = {
            "desc": pg["pg_desc"],
            "ranks": pg["ranks"],
            "streams": set()
        }

    events = data.get("traceEvents", [])
    local_timeline = {"driver": {}, "Stream_runtime": []}
    kernel_correlations = set()
    events.sort(key=lambda e:e.get("ts", 0))
    gpu_ts = None
    if exclude_pre_profiler:
        gpu_ts = next(
            (
                entry["ts"]
                for entry in events
                if "Profiler" in entry.get("name", "")
                and entry.get("cat") == "gpu_user_annotation"
            ),
            None
        )
    base_ts = events[0]["ts"]
    if exclude_pre_profiler:
        gemm_stream = next(
            (
                entry["args"]["stream"]
                for entry in events
                if "nvjet" in entry.get("name", "") or "gemm" in entry.get("name", "")
            ),
            None
        )
    # Pre-pass: build correlation → MNK/attn/comm suffix maps
    ext_id_to_dims = {}
    ext_id_to_comm_size = {}
    correlation_to_ext_id = {}
    for e in events:
        if "cat" not in e: continue
        if e["cat"] == "cpu_op" and e["name"] == "aten::mm":
            ext_id = e["args"].get("External id")
            dims = e["args"].get("Input Dims")
            if ext_id is not None and dims is not None:
                ext_id_to_dims[ext_id] = dims
        elif e["cat"] == "cpu_op" and ("attention" in e["name"] or "Attention" in e["name"]):
            ext_id = e["args"].get("External id")
            dims = e["args"].get("Input Dims")
            if ext_id is not None and dims is not None:
                ext_id_to_dims[ext_id] = dims[:3]
        elif e["cat"] == "cpu_op" and e["name"] == "record_param_comms":
            ext_id = e["args"].get("External id")
            in_nelems = e["args"].get("In msg nelems", 0)
            out_nelems = e["args"].get("Out msg nelems", 0)
            if ext_id is not None:
                ext_id_to_comm_size[ext_id] = max(in_nelems, out_nelems)
        elif e["cat"] in ("kernel", "gpu_memcpy"):
            ext_id = e["args"].get("External id")
            if ext_id is not None:
                correlation_to_ext_id[e["args"]["correlation"]] = ext_id

    def _mnk_suffix(correlation):
        ext_id = correlation_to_ext_id.get(correlation)
        if ext_id is None: return ""
        dims = ext_id_to_dims.get(ext_id)
        if dims is None or len(dims) < 2: return ""
        M, K, N = dims[0][0], dims[0][1], dims[1][1]
        return f"_{M}_{K}_{N}"
    def _attn_suffix(correlation):
        ext_id = correlation_to_ext_id.get(correlation)
        if ext_id is None: return ""
        dims = ext_id_to_dims.get(ext_id)
        if dims is None or len(dims) == 0: return ""
        return "_" + "_".join(str(d) for d in dims[0])
    def _comm_suffix(correlation):
        ext_id = correlation_to_ext_id.get(correlation)
        if ext_id is None: return ""
        size = ext_id_to_comm_size.get(ext_id)
        if size is None: return ""
        return f"_{size}"
    kernel_durs = []
    for e in events:
        if "cat" not in e: continue
        if e["cat"] in ("kernel", "gpu_memcpy") and (not exclude_pre_profiler or gpu_ts is None or e["ts"] > gpu_ts):
            stream = "Stream_" + str(e["args"]["stream"])
            if stream not in local_timeline:
                local_timeline[stream] = []
            corr = e["args"]["correlation"]
            name = e["name"]
            if "nvjet" in name or "gemm" in name:
                name = name + _mnk_suffix(corr)
                kernel_durs.append((name, e["dur"]))
            elif "sdpa" in name or "MemEffAttention" in name:
                name = name + _attn_suffix(corr)
            elif "ncclDevKernel" in name:
                name = name + _comm_suffix(corr)
            local_timeline[stream].append((e["ts"] - base_ts, e["dur"], name, corr))
            kernel_correlations.add(e["args"]["correlation"])
            if "ncclDevKernel" in e["name"]:
                pg_id = e["args"].get("Process Group Name", "N/A")
                if pg_id == "N/A":
                    continue
                pg_config[pg_id]["streams"].add(stream)
        elif e["cat"] == "cpu_op" or ("DataLoader" in e["name"] and e["cat"] == "user_annotation"):
            tid = "Thread_"
            if tid not in local_timeline:
                local_timeline[tid] = []
            local_timeline[tid].append((e["ts"] - base_ts, e["dur"], e["name"]))
        elif "cudaStreamSync" in e["name"] or "cudaDeviceSync" in e["name"] or "cudaEventSync" in e["name"]:
            tid = "Thread_"
            if tid not in local_timeline:
                local_timeline[tid] = []
    if exclude_pre_profiler:
        local_timeline["Stream_" + str(gemm_stream)].insert(0, (0, local_timeline["Stream_" + str(gemm_stream)][0][0], "gpu_start", -1))       
 
    local_timeline["Thread_"].insert(0, (0, local_timeline["Thread_"][0][0], "cpu_start"))
    for e in events:
        if "cat" not in e: continue
        if e["name"] not in ("cudaLaunchKernel", "cuLaunchKernelEx", "cuLaunchKernelExC", "cuLaunchKernel", "cudaMemcpyAsync"): continue
        if e["args"]["correlation"] not in kernel_correlations: continue
        local_timeline["driver"][e["args"]["correlation"]] = (e["ts"] - base_ts, e["dur"], e["name"], e["tid"])
    
    del events, data, metadata, kernel_correlations, correlation_to_ext_id, ext_id_to_dims, ext_id_to_comm_size
    gc.collect()
    return local_timeline, pg_config, base_ts, kernel_durs

def handle_synchronization_events(G, root, sync_events, gpu_events, last_comm_ts):
    i = 0
    j = 0
    # Preserve build/queue order. Matching by global GPU end-time can wire future GPU work
    # to earlier CPU nodes and introduce cycles.
    while i < len(sync_events) and j < len(gpu_events):
        gpu_end = gpu_events[j].timestamp + gpu_events[j].duration
        if gpu_end > last_comm_ts:
            break
        while i < len(sync_events) and (sync_events[i][0][0] + sync_events[i][0][1]) <= gpu_end:
            src_id = root.id if j == 0 else gpu_events[j - 1].id
            G.add_edge(src_id, sync_events[i][1])
            cpu_node = G.get_node(sync_events[i][1])
            i += 1
        j += 1
    if i < len(sync_events):
        print(f"Warning: {len(sync_events) - i} unmatched synchronization events")

def handle_cpu_events(timeline, G, kernel_launch_map, id, root, gpu_events, last_comm_ts):
    tmp_timeline = {key: timeline[key] for key in timeline if key.startswith("Thread_")}
    sync_events = []
    sync_event_queue = []
    count = 0
    for corr, launch in timeline["driver"].items():
        tmp_timeline[f"Thread_"].append((launch[0], launch[1], launch[2], corr))
    for stream in tmp_timeline:
        if not stream.startswith("Thread_"):
            continue
        tmp_timeline[stream] = chunk_all(tmp_timeline[stream])
        prev = root
        for event in tmp_timeline[stream]:
            # If it's a kernel launch event
            if len(event) == 4:
                if  event[3] not in kernel_launch_map:
                    count += 1
                    continue
                node = G.get_node(kernel_launch_map[event[3]])
                G.add_edge(prev.id, node.id)
                prev = node
            elif event[2].startswith("cudaStreamSynchronize"):
                sync_event_queue.append(event)
            else:
                node = Node(id=id, name=event[2], timestamp=event[0], duration=event[1], is_gpu=False, stream=-1)
                G.add_node(node)
                G.add_edge(prev.id, node.id)
                prev = node
                id += 1
                while len(sync_event_queue) > 0:
                    sync_ts = sync_event_queue[0][0] + sync_event_queue[0][1]
                    if sync_ts > event[0] + event[1]:
                        break
                    sync_events.append((sync_event_queue[0], node.id))
                    sync_event_queue.pop(0)
    assert len(sync_event_queue) == 0
    handle_synchronization_events(G, root, sync_events, gpu_events, last_comm_ts)               
    return

def create_graph(timeline, comm_stream):

    G = DiGraph()
    driver = timeline["driver"]

    dummy = Node(id=0, name="dummy", timestamp=0.0, duration=0.0, is_gpu=False, stream=-1)
    G.add_node(dummy)

    streams = []
    stream_names = []

    for s, events in timeline.items():
        if s == "driver" or s.startswith("Thread_"):
            continue
        streams.append(events)
        stream_names.append(s)

    stream_pos = [0]*len(streams)

    active_heap = []
    last_finished = None
    last_stream_node = {}

    edges = []

    node_id = 1
    comm_nodes = set()
    gpu_events = []
    last_comm_end = 0

    kernel_launch_map = {}

    while True:
        #find most recent event in tl
        best_stream = None
        best_time = float("inf")

        for s_id, events in enumerate(streams):
            pos = stream_pos[s_id]
            if pos < len(events):
                t = events[pos][0]
                if t < best_time:
                    best_time = t
                    best_stream = s_id

        if best_stream is None:
            break

        event = streams[best_stream][stream_pos[best_stream]]
        stream_pos[best_stream] += 1

        start = event[0]
        duration = event[1]
        name = event[2]
        corr = event[3]
        stream = stream_names[best_stream]
        
        #pop off heap until reach unqualified event
        while active_heap and active_heap[0][0] <= start:
            _, finished_id = heapq.heappop(active_heap)
            last_finished = finished_id

        if last_finished is not None:
            edges.append((last_finished, node_id))
        else:
            edges.append((0, node_id))

        prev = last_stream_node.get(stream)
        if prev is not None:
            edges.append((prev, node_id))

        last_stream_node[stream] = node_id

        heapq.heappush(active_heap, (start+duration, node_id))

        node = Node(node_id, name, start, duration, True, stream)
        G.add_node(node)

        if stream == comm_stream:
            comm_nodes.add(node_id)
            last_comm_end = max(last_comm_end, start+duration)
        else:
            gpu_events.append(node)

        launch = driver.get(corr)
        if launch:
            launch_id = node_id + 1
            launch_node = Node(launch_id,"cudaLaunchKernel",launch[0],launch[1],False, -1)
            G.add_node(launch_node)
            edges.append((launch_id,node_id))
            kernel_launch_map[corr] = launch_id
            node_id += 1

        node_id += 1

    for u,v in edges:
        G.add_edge(u,v)

    handle_cpu_events(timeline, G, kernel_launch_map, node_id, dummy, gpu_events, last_comm_end)

    return G, comm_nodes

def dfs(G, node_id, start_node_id, comm_nodes, i):
    comp_time = 0
    path_nodes = {}
    path_counts = defaultdict(int)
    cpu_side = False
    kernel_gaps = 0
    
    while True:
        node = G.get_node(node_id)
        same_stream_parent = None
        if "Launch" in G.get_node(node_id).name:
            cpu_side = True
        if (node_id != start_node_id and node_id in comm_nodes) or node_id == 0:
            node = G.get_node(node_id)
            return comp_time, node_id, path_nodes, cpu_side, kernel_gaps

        parents = G.neighbors_in(node_id)
        if len(parents) == 0:
            return comp_time, node_id, path_nodes, cpu_side, kernel_gaps
        latest = G.get_node(parents[0])
        current_stream = G.get_node(node_id).stream

        if latest.stream == current_stream:
            same_stream_parent = latest
        for p_id in parents[1:]:
            p = G.get_node(p_id)
            if p.stream == current_stream:
                same_stream_parent = p
            parent_end = p.timestamp + p.duration
            latest_end = latest.timestamp + latest.duration
            if parent_end > latest_end:
                latest = p
        if same_stream_parent is not None:
            latest_end = latest.timestamp + latest.duration
            same_end = same_stream_parent.timestamp + same_stream_parent.duration
        
            if abs(latest_end - same_end) < THRESHOLD:
                latest = same_stream_parent
                
        if node_id != start_node_id:
            comp_time += node.duration
            name = node.name
            key = (sys.intern(name), path_counts[name])
            path_nodes[key] = node.duration
            path_counts[name] += 1
        
        gap = node.timestamp - (latest.timestamp + latest.duration)
        if gap > 0:
            kernel_gaps += gap   
        node_id = latest.id
    
def traverse_graph(G, comm_nodes):
    comms = []
    comps = []
    paths = []
    cpu_paths = []
    comm_nodes_list = sorted(list(comm_nodes))
    comm_node_to_index = {comm_node_id: i for i, comm_node_id in enumerate(comm_nodes_list)}
    comm_node_to_index[0] = -1
    ancestors = []
    all_gaps = []
   
    for i, comm_node in enumerate(comm_nodes_list):
        comp, ancestor, comp_paths, on_cpu, kernel_gaps = dfs(G, comm_node, comm_node, comm_nodes, i)
        comms.append((G.get_node(comm_node).duration, G.get_node(comm_node).name))
        comps.append(comp)
        ancestors.append(comm_node_to_index[ancestor])
        paths.append(comp_paths)
        cpu_paths.append(on_cpu)
        all_gaps.append(kernel_gaps)
    return comms, comps, ancestors, paths, cpu_paths, all_gaps

def shift_timeline(tl, adjustment):
    for key, events in list(tl.items()):
        if key == "driver":
            for corr, ev in list(events.items()):
                events[corr] = (ev[0] + adjustment, ev[1], ev[2], ev[3])
        else:
            tl[key] = [(ev[0] + adjustment,) + ev[1:] for ev in events]
    tl["Thread_"].insert(0, (0, adjustment, "process_launch_delay"))

def process_iteration(subdir_path, comm_stream_to_analyze, exclude_pre_profiler):
    subdir = os.path.basename(subdir_path)
    print(f"Processing {subdir}...")
    file_paths = [os.path.join(subdir_path, f) for f in sorted(os.listdir(subdir_path))]

    all_comms = {}
    all_comps = {}
    all_ancestors = {}
    all_paths = {}
    local_mesh = {}
    all_cpu_flags = {}
    all_kernel_gaps = {}
    local_analytical = {"comm": {}, "comp": {}}
    all_kernels = {}

    culprit_counter = Counter() 
    culprit_impact = defaultdict(float)

    # Phase 1: parse all rank logs and determine comm_stream per rank
    parsed = {}  # rank -> (local_timeline, comm_stream)
    for f in file_paths:
        print(f"\t Parsing {f}...")
        local_timeline, pg_config, base_ts, itr_kernels = parse_log(f, exclude_pre_profiler)
        comm_streams = {}
        for pg_info in pg_config.values():
            if pg_info["streams"] is None: continue
            desc = pg_info["desc"]
            if desc not in local_mesh:
                local_mesh[desc] = []
            if pg_info["ranks"] not in local_mesh[desc]:
                local_mesh[desc].append(pg_info["ranks"])
            if bool(pg_info["streams"]):
                comm_streams[desc] = pg_info["streams"]
        candidate_streams = comm_streams.get(comm_stream_to_analyze, set())

        if not candidate_streams:
            raise RuntimeError(f"No streams found for {comm_stream_to_analyze}")

        def is_reduce_scatter(name):
            name = name.lower()
            return "scatter" in name and "reduce" in name

        selected_stream = None
        max_matches = 0

        for stream in candidate_streams:
            events = local_timeline.get(stream, [])
            
            count = sum(
                1 for (_, _, name, _) in events
                if "ncclDevKernel" in name and is_reduce_scatter(name)
            )

            if count > max_matches:
                max_matches = count
                selected_stream = stream

        if selected_stream is None:
            raise RuntimeError(
                f"No ReduceScatter stream found in {comm_stream_to_analyze}. "
                f"Candidates: {candidate_streams}"
            )

        comm_stream = selected_stream

        print(f"Selected comm stream for {comm_stream_to_analyze}: {comm_stream} "
            f"(ReduceScatter count={max_matches})")
        
        match = re.search(r"rank(\d+)_trace\.json", f.split("/")[-1])
        rank = int(match.group(1))
        parsed[rank] = (local_timeline, comm_stream, base_ts)
        all_kernels[rank] = itr_kernels
    # Phase 2: compute process-launch adjustment per rank from first comm-stream
    # kernel end-time relative to base_ts; align across ranks within each group.
    for group in local_mesh[comm_stream_to_analyze]:
        alignments = {}
        for rank in group:
            alignments[rank] = parsed[rank][2]
        min_end = min(alignments.values())
        adjustments = {}
        for rank in group:
            adjustment = alignments[rank] -  min_end
            adjustments[rank] = adjustment
            shift_timeline(parsed[rank][0], adjustment)
    # Phase 3: build graph and traverse for each rank
    for rank, (local_timeline, comm_stream, _) in parsed.items():
        G, comm_nodes = create_graph(local_timeline, comm_stream)
        comms, comps, ancestors, comp_paths, cpu_paths, kernel_gaps = traverse_graph(G, comm_nodes)
        all_comms[rank] = comms
        all_comps[rank] = comps
        all_ancestors[rank] = ancestors
        all_paths[rank] = comp_paths
        all_cpu_flags[rank] = cpu_paths
        all_kernel_gaps[rank] = kernel_gaps
        local_analytical["comm"][rank] = [dur for dur, _ in comms]

    assert len(set(len(c) for c in all_comms.values())) == 1, "Length of comms is not the same across ranks"
    num_events = len(next(iter(all_comms.values())))
    local_comm_delays = []
    local_comp_delays = []
    local_comm_names = []
    local_mixed_comm = []
    local_mixed_comp = []
    local_comm_events = []

    for group in local_mesh[comm_stream_to_analyze]:
        sub_comms = {rank: all_comms[rank] for rank in group}
        sub_comps = {rank: all_comps[rank] for rank in group}
        sub_ancestors = {rank: all_ancestors[rank] for rank in group}
        sub_diffs = {rank: [0] * num_events for rank in group}
        sub_paths = {rank: all_paths[rank] for rank in group}
        
        mins = [min(sub_comms[r][j][0] for r in group) for j in range(num_events)]
        baseline_comm = sum(mins) / len(mins)
        sub_cpu_flags = {rank: all_cpu_flags[rank] for rank in group}
        sub_gaps = {rank: all_kernel_gaps[rank] for rank in group}

        for event_idx in range(num_events):
            rank_op_durations = {}
            rank_op_counts = {}
            gaps = {}
            comm_times = {}
            comp_times = {}
            comm_names = set()
            oldest_ancestor = min([sub_ancestors[rank][event_idx] for rank in sub_ancestors])

            rank_op_durations = {}
            rank_op_counts = {}

            for rank in group:
                comm_dur, comm_name = sub_comms[rank][event_idx]
                comm_names.add(re.sub(r'_\d+$', '', comm_name))
                comm_times[rank] = comm_dur
                comp_times[rank] = sub_comps[rank][event_idx]
                gaps[rank] = sub_gaps[rank][event_idx]

                rank_op_durations[rank] = dict(sub_paths[rank][event_idx])
                rank_op_counts[rank] = defaultdict(int)
                for (name, idx) in sub_paths[rank][event_idx]:
                    rank_op_counts[rank][name] = max(rank_op_counts[rank][name], idx + 1)
                    
                if sub_ancestors[rank][event_idx] == oldest_ancestor:
                    continue
                j = sub_ancestors[rank][event_idx]
                while j >= 0 and sub_ancestors[rank][j] > oldest_ancestor:
                    gaps[rank] += sub_gaps[rank][j]
                    comp_times[rank] += (sub_comps[rank][j] + sub_comms[rank][j][0])
                    for (name, idx), dur in sub_paths[rank][j].items():
                        offset = rank_op_counts[rank][name]
                        rank_op_durations[rank][(name, idx + offset)] = dur
                    for (name, idx) in sub_paths[rank][j]:
                        rank_op_counts[rank][name] = max(rank_op_counts[rank][name], idx + offset + 1)
                    j = sub_ancestors[rank][j]

                max_comp = max(sub_comps[i][j] for i in group)
                diff = max_comp - sub_comps[rank][j]
                gaps[rank] += sub_gaps[rank][j]

                comp_times[rank] += (sub_comps[rank][j] + sub_comms[rank][j][0]) # baseline_comm + diff)
      
                for (name, idx), dur in sub_paths[rank][j].items():
                    offset = rank_op_counts[rank][name]
                    rank_op_durations[rank][(name, idx + offset)] = dur
                for (name, idx) in sub_paths[rank][j]:
                    rank_op_counts[rank][name] = max(rank_op_counts[rank][name], idx + offset + 1)

            assert len(comm_names) == 1
            local_comm_names.extend(list(comm_names))
            fastest_rank = max(comm_times, key=comm_times.get)
            slowest_rank = min(comm_times, key=comm_times.get)

            best_culprit = None
            best_culprit_variance = -1
            for key, dur in rank_op_durations[slowest_rank].items():
                durations_across_ranks = [rank_op_durations[rank].get(key, 0.0) for rank in group]
                op_variance = dur - min(durations_across_ranks)
                if op_variance > best_culprit_variance:
                    best_culprit_variance = op_variance
                    best_culprit = key[0]
            
            if best_culprit is not None:
                culprit_counter[best_culprit] += 1
                culprit_impact[best_culprit] += best_culprit_variance

            rank_op_durations.clear()
            rank_op_counts.clear()
            mixed = False
            flags = [sub_cpu_flags[rank][event_idx] for rank in group]
            if any(flags) and not all(flags):
                mixed = True

            if not mixed:
                local_comm_delays.append(comm_times[fastest_rank] - comm_times[slowest_rank])
                local_comp_delays.append(comp_times[slowest_rank] - comp_times[fastest_rank])
            else:
                comp_times[slowest_rank] += gaps[slowest_rank]
                comp_times[fastest_rank] += gaps[fastest_rank]

                local_mixed_comm.append(comm_times[fastest_rank] - comm_times[slowest_rank])
                local_mixed_comp.append(comp_times[slowest_rank] - comp_times[fastest_rank])

            comm_name = next(iter(comm_names))
            local_comm_events.append({
                "comm_times": dict(comm_times),
                "comp_times": dict(comp_times),
                "slowest_rank": slowest_rank,
                "fastest_rank": fastest_rank,
                "comm_name": comm_name,
                "cpu_bottleneck_flags": {rank: sub_cpu_flags[rank][event_idx] for rank in group},
                "is_cpu_bottleneck": any(sub_cpu_flags[rank][event_idx] for rank in group),
            })
    del parsed, all_comms, all_comps, all_paths
    gc.collect()
    return subdir, local_comm_delays, local_comp_delays, local_comm_names, local_analytical, local_mixed_comm, local_mixed_comp, culprit_counter, culprit_impact, local_comm_events, itr_kernels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose stragglers in distributed training traces.")
    parser.add_argument("input_dir", help="Path to directory containing iteration subdirectories")
    parser.add_argument("comm_stream", help="Communication stream to analyze (e.g. mesh_tp)")
    parser.add_argument("--exclude-pre-profiler", action="store_true", default=False,
                        help="Exclude GPU events before the GPU profiler annotation starts")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1 = single-process, no Pool)")
    parser.add_argument("--injected-delay-us", type=float, default=0.0,
                        help="Amount of delay injected into the CPU thread (microseconds)")
    args = parser.parse_args()

    input_dir = args.input_dir
    comm_stream_to_analyze = args.comm_stream
    exclude_pre_profiler = args.exclude_pre_profiler
    
    write_dir = f"../plots/diagnosis_v4/{input_dir.split('/')[-1]}"
    os.makedirs(write_dir, exist_ok=True)

    subdir_paths = sorted(
        os.path.join(input_dir, d)
        for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    )
    worker = partial(process_iteration,
                     comm_stream_to_analyze=comm_stream_to_analyze,
                     exclude_pre_profiler=exclude_pre_profiler)

    mesh = {}
    comm_delays = {}
    comp_delays = {}
    mix_comm_delays = {}
    mix_comp_delays = {}
    all_comm_names = {}
    analytical_analysis = {}
    all_culprit_counter = Counter() 
    all_culprit_impact = defaultdict(float)
    all_comm_events={}
    kernel_durs = {}
    def aggregate(results_iter):
        for subdir, local_comm_delays, local_comp_delays, local_comm_names, local_analytical, local_mixed_comm, local_mixed_comp, culprit_counter, culprit_impact, local_comm_events, itr_kernels in results_iter:
            comm_delays[subdir] = local_comm_delays
            comp_delays[subdir] = local_comp_delays
            all_comm_names[subdir] = local_comm_names
            analytical_analysis[subdir] = local_analytical
            mix_comm_delays[subdir] = local_mixed_comm
            mix_comp_delays[subdir] = local_mixed_comp
            all_comm_events[subdir] = local_comm_events
            kernel_durs[subdir] = itr_kernels
            for k, v in culprit_counter.items():
                all_culprit_counter[k] += v
            for k, v in culprit_impact.items():
                all_culprit_impact[k] += v

    if args.num_workers == 1:
        aggregate(map(worker, subdir_paths))
    else:
        with multiprocessing.Pool(processes=args.num_workers, maxtasksperchild=1) as pool:
            aggregate(pool.imap_unordered(worker, subdir_paths))

    
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'Fira Code',
        'axes.prop_cycle': cycler(color=list(COLORS2HEX.values())),
    })

    flat_comm_delays = [d for v in comm_delays.values() for d in v]
    flat_comp_delays = [d for v in comp_delays.values() for d in v]

    plt.scatter(flat_comp_delays, flat_comm_delays)
    slope, intercept = np.polyfit(flat_comp_delays, flat_comm_delays, 1)
    r = np.corrcoef(flat_comp_delays, flat_comm_delays)[0, 1]
    max_limit = max(max(flat_comp_delays), max(flat_comm_delays)) * 1.05
    x_vals = np.linspace(0, max_limit, 100)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, linestyle='--', linewidth=2.5)
    plt.xlabel("Compute Variation (us)", fontsize=16)
    plt.ylabel("Comm Variation (us)", fontsize=16)
    eq_str = rf"$y = {slope:.3f}x {'+' if intercept >= 0 else '-'} {abs(intercept):.3f}$"
    r_str = rf"$r = {r:.3f}$"
    plt.text(
        0.05, 0.95,
        eq_str + "\n" + r_str,
        transform=plt.gca().transAxes,
        verticalalignment="top"
    )
    plt.savefig(f"{write_dir}/scatter_kernels.png", bbox_inches='tight')
    plt.close()

    flat_mix_comm_delays = [d for v in mix_comm_delays.values() for d in v]
    flat_mix_comp_delays = [d for v in mix_comp_delays.values() for d in v]
    if len(flat_mix_comm_delays) > 0 and len(flat_mix_comp_delays) > 0:
        plt.scatter(flat_mix_comp_delays, flat_mix_comm_delays)
        slope, intercept = np.polyfit(flat_mix_comp_delays, flat_mix_comm_delays, 1)
        r = np.corrcoef(flat_mix_comp_delays, flat_mix_comm_delays)[0, 1]
        max_limit = max(max(flat_mix_comp_delays), max(flat_mix_comm_delays)) * 1.05
        x_vals = np.linspace(0, max_limit, 100)
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, linestyle='--', linewidth=2.5)
        plt.xlabel("Mixed Compute Variation (us)", fontsize=16)
        plt.ylabel("Mixed Comm Variation (us)", fontsize=16)
        eq_str = rf"$y = {slope:.3f}x {'+' if intercept >= 0 else '-'} {abs(intercept):.3f}$"
        r_str = rf"$r = {r:.3f}$"
        plt.text(
            0.05, 0.95,
            eq_str + "\n" + r_str,
            transform=plt.gca().transAxes,
            verticalalignment="top"
        )
        plt.savefig(f"{write_dir}/mix_scatter_kernels.png", bbox_inches='tight')
        plt.close()

    os.makedirs("../validation-data", exist_ok=True)
    
    straggler_path = f"../straggler-data/{input_dir.split('/')[-1]}.pkl"
  
    with open(straggler_path, "wb") as f:
        pickle.dump({
            "injected_delay_us": args.injected_delay_us,
            "comm_delays": comm_delays,
            "comp_delays": comp_delays,
            "mix_comm_delays": flat_mix_comm_delays,
            "mix_comp_delays": flat_mix_comp_delays,
            "all_culprit_impacts": all_culprit_impact,
            "all_culprit_counts": all_culprit_counter,
            "comm_names": all_comm_names,
            "comm_events": all_comm_events, 
            "kernel_durs": kernel_durs,
        }, f)
   
    pkl_path = f"../validation-data/{input_dir.split('/')[-1]}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(analytical_analysis, f)
    print(f"Saved analytical_analysis to {pkl_path}")
