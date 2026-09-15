import argparse
import gc
import json
import multiprocessing
import os
import pickle
import re
import sys
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.lines import Line2D

from diagnosis_tool import (
    COLORS2HEX,
    create_graph,
    parse_log,
    shift_timeline,
    traverse_graph,
)

sys.setrecursionlimit(100000)

BW_FACTORS = [1.0, 1.2, 1.4]

DATA_DIR = "<DATA_PATH>"
Y_MIN = 0.7
Y_MAX = 1.01
BAR_GROUP_WIDTH = 0.5
FIG_WIDTH_PER_WORKLOAD = 1.5
PAPER_FIG_SIZE = (6.8, 4.6)
FS_AXIS_LABEL = 18
FS_TICK = 16
FS_LEGEND = 15
BAR_YLIM = (0.8, 1.006)

STEP_YLABEL = "Norm. Step Time"
KNEE_XLABEL = "Bandwidth vs. today's NVLink"
FS_STANDALONE = {"label": FS_AXIS_LABEL, "tick": FS_TICK, "legend": FS_LEGEND}

PAPER_FIG = "<OUTPUT_PATH>"

KNEE_SWEEP = [1.15, 1.35, 1.6, 1.9, 2.3, 2.8, 3.5, 4.5, 6.0, 8.0,
              11.0, 15.0, 21.0, 30.0, 45.0, 65.0, 95.0, 130.0]
OPT_ALG = "ring"
OPT_ELASTICITY = -0.5
OPT_M = 4  #group size
OPT_SERIES = [
    (0, "Today's BW (NVLink peak)"),
    (1, "Optimal BW (no sync tax)"),
    (2, "Optimal BW (w/ sync tax)"),
]

OPT_YLIM = (0.82, 1.10)

KNEE_YLIM = (0.7, 1.005)

PAPER_FIG_OPT = "<OUTPUT_PATH>"
PAPER_FIG_KNEE = "<OUTPUT_PATH>"
PAPER_FIG_PAIR = "<OUTPUT_PATH"

PAIR_FIG_SIZE = (11.4, 4.55)
FS_PAIR = {"label": 19, "tick": 16, "legend": 17}
PAIR_BAR_YLIM = (KNEE_YLIM[0], BAR_YLIM[1])
PAIR_TITLES = ("(a) Effect of BW on step time", "(b) B* with and without $\\tau$")


MODEL_NAMES = {"llama70b": "Llama-70B", "llama8b": "Llama-8B", "qwen32b": "Qwen-32B"}
HW_NAMES = {"a100": "A100", "h100": "H100", "h200": "H200"}


def pretty_label(workload):
    parts = workload.split("-")
    model = next((v for k, v in MODEL_NAMES.items() if k in parts), workload)
    device = next((v for k, v in HW_NAMES.items() if k in parts), "")
    return f"{model}\n{device}" if device else model


def compact_label(workload):
    model, _, device = pretty_label(workload).partition("\n")
    return "\n".join(model.split("-") + ([device] if device else []))


def factor_label(f):
    if abs(f - 1.0) < 1e-9:
        return "Today's BW"
    return f"{(f - 1) * 100:+.0f}% BW"


def split_workload(workload):
    base, arch = workload.rsplit("-", 1)
    n = int(re.match(r"n(\d+)", base.split("-")[0]).group(1))
    return base, arch, n


def optimal_bw_factors(workload):
    import bw_scaling as ss

    base, arch, n = split_workload(workload)
    durs, covs, params, _ = ss.gather_llama70b_combo_stats(workload=base, archs=[arch])
    s_gb, dtype = ss.get_max_reducescatter_gb(
        ss.TRACE_SAMPLE_TEMPLATE.format(workload=base, arch=arch))
    dur = float(np.mean(durs))
    cov = float(np.mean(covs))
    alpha = ss.ALPHA_US / 1e3
    p, q = ss.calc_p_q_for_alg(OPT_ALG, n)
    peak = ss.NVLINK_PEAK[arch]

    # calc_optimal_bw_* return GB/ms; NVLINK_PEAK is GB/s.
    bw_base = ss.calc_optimal_bw_baseline(p, q, alpha, s_gb, OPT_ELASTICITY) * 1e3
    bw_sync = ss.calc_optimal_bw_ours(
        p, q, alpha, s_gb, float(params["loc"]), float(params["scale"]),
        n / OPT_M, dur, cov, OPT_ELASTICITY, float(params["shape"])) * 1e3

    info = {
        "n": n, "arch": arch, "alg": OPT_ALG, "elasticity": OPT_ELASTICITY,
        "M": OPT_M, "k": n / OPT_M, "S_gb": s_gb, "dtype": dtype,
        "dur_ms": dur, "cov": cov, "num_combos": len(durs),
        "nvlink_peak": peak, "bw_baseline": bw_base, "bw_sync_tax": bw_sync,
    }
    print(f"  optimal BW (ring, N={n}, k={n / OPT_M:g}, e={OPT_ELASTICITY}): "
          f"S={s_gb:.4f} GB ({dtype}), dur={dur:.1f} ms, cov={cov:.5f}")
    print(f"    NVLink peak {peak:.0f} GB/s | no sync tax {bw_base:.1f} GB/s "
          f"({bw_base / peak:.2f}x) | w/ sync tax {bw_sync:.1f} GB/s "
          f"({bw_sync / peak:.2f}x)")
    factors = [1.0, bw_base / peak, bw_sync / peak]
    factors += [f for f in KNEE_SWEEP
                if all(abs(f - g) > 1e-9 for g in factors)]
    return factors, info


def detect_comm_stream(workload_dir):
    import glob
    trace = sorted(glob.glob(os.path.join(workload_dir, "*", "rank0_trace.json")))[0]
    with open(trace) as f:
        pgs = {p["pg_desc"] for p in json.load(f)["distributedInfo"]["pg_config"]}
    if "mesh_tp" in pgs:
        return "mesh_tp"
    if "default_pg" in pgs:
        return "default_pg"
    raise ValueError(f"No mesh_tp or default_pg process group in {workload_dir}")

def path_collective_time(path_nodes):
    return sum(dur for (name, _), dur in path_nodes.items() if "nccl" in name)


def timeline_end(timeline):
    end = 0.0
    for key, events in timeline.items():
        if key == "driver" or key.startswith("Thread_"):
            continue
        for ev in events:
            end = max(end, ev[0] + ev[1])
    return end


def first_gpu_ts(timeline):
    start = float("inf")
    for key, events in timeline.items():
        if key == "driver" or key.startswith("Thread_"):
            continue
        for ev in events:
            if ev[2] == "gpu_start":
                continue
            start = min(start, ev[0])
    return 0.0 if start == float("inf") else start


def reachable_collectives(group, ancestors, last):
    seen = {last}
    stack = [last]
    while stack:
        j = stack.pop()
        for rank in group:
            anc = ancestors[rank][j]
            assert anc < j, f"ancestor {anc} of collective {j} is not earlier"
            if anc >= 0 and anc not in seen:
                seen.add(anc)
                stack.append(anc)
    return sorted(seen)


def replay_chain(group, comps, path_comms, ancestors, pure_comm, bw_factor,
                 scale_path_collectives, order, last):
    done = {}
    crit_rank = {}
    eff_comp = {}
    for j in order:
        best = float("-inf")
        best_rank = None
        best_comp = 0.0
        candidates = [r for r in group if ancestors[r][j] >= 0] or list(group)
        for rank in candidates:
            anc = ancestors[rank][j]
            base = done[anc] if anc >= 0 else 0.0
            comp = comps[rank][j]
            if scale_path_collectives:
                nccl = path_comms[rank][j]
                comp = (comp - nccl) + nccl / bw_factor
            arrival = base + comp
            if arrival > best:
                best = arrival
                best_rank = rank
                best_comp = comp
        done[j] = best + pure_comm[j] / bw_factor
        crit_rank[j] = best_rank
        eff_comp[j] = best_comp

    critical_path = []
    j = last
    while j >= 0:
        critical_path.append(j)
        j = ancestors[crit_rank[j]][j]
    critical_path.reverse()

    split = {
        "comm": sum(pure_comm[j] / bw_factor for j in critical_path),
        "comp": sum(eff_comp[j] for j in critical_path),
    }
    return done, done[last], critical_path, split


def process_iteration(subdir_path, comm_stream_to_analyze, exclude_pre_profiler,
                      bw_factors, scale_path_collectives):
    subdir = os.path.basename(subdir_path)
    print(f"Processing {subdir}...")
    file_paths = [os.path.join(subdir_path, f) for f in sorted(os.listdir(subdir_path))]

    local_mesh = {}
    parsed = {}
    for f in file_paths:
        local_timeline, pg_config, base_ts, _ = parse_log(f, exclude_pre_profiler)
        comm_streams = {}
        for pg_info in pg_config.values():
            if pg_info["streams"] is None:
                continue
            desc = pg_info["desc"]
            if desc not in local_mesh:
                local_mesh[desc] = []
            if pg_info["ranks"] not in local_mesh[desc]:
                local_mesh[desc].append(pg_info["ranks"])
            if bool(pg_info["streams"]):
                comm_streams[desc] = pg_info["streams"]

        assert len(comm_streams[comm_stream_to_analyze]) == 1, \
            f"Expected exactly one stream for {comm_stream_to_analyze}, " \
            f"but found {len(comm_streams[comm_stream_to_analyze])}"
        comm_stream, = comm_streams[comm_stream_to_analyze]
        rank = int(re.search(r"rank(\d+)_trace\.json", f.split("/")[-1]).group(1))
        parsed[rank] = (local_timeline, comm_stream, base_ts)

    for group in local_mesh[comm_stream_to_analyze]:
        min_end = min(parsed[rank][2] for rank in group)
        for rank in group:
            shift_timeline(parsed[rank][0], parsed[rank][2] - min_end)

    all_comps = {}
    all_ancestors = {}
    all_comms = {}
    all_path_comms = {}
    all_last_comm_end = {}
    all_trace_end = {}
    all_trace_start = {}

    for rank, (local_timeline, comm_stream, _) in parsed.items():
        G, comm_nodes = create_graph(local_timeline, comm_stream)
        comms, comps, ancestors, paths, _, _ = traverse_graph(G, comm_nodes)
        all_comms[rank] = comms
        all_comps[rank] = comps
        all_ancestors[rank] = ancestors
        all_path_comms[rank] = [path_collective_time(p) for p in paths]
        all_last_comm_end[rank] = max(
            G.get_node(n).timestamp + G.get_node(n).duration for n in comm_nodes
        )
        all_trace_end[rank] = timeline_end(local_timeline)
        all_trace_start[rank] = first_gpu_ts(local_timeline)
        del G, comm_nodes, paths

    assert len(set(len(c) for c in all_comms.values())) == 1, \
        "Length of comms is not the same across ranks"
    num_events = len(next(iter(all_comms.values())))

    results = {}
    for group in local_mesh[comm_stream_to_analyze]:
        pure_comm = [min(all_comms[r][j][0] for r in group) for j in range(num_events)]

        tail = max(all_trace_end[r] - all_last_comm_end[r] for r in group)
        origin = min(all_trace_start[r] for r in group)
        measured_e2e = max(all_trace_end[r] for r in group) - origin

        last = num_events - 1
        order = reachable_collectives(group, all_ancestors, last)

        per_factor = {}
        crit_len = {}
        crit_split = {}
        for bw in bw_factors:
            _, last_done, critical_path, split = replay_chain(
                group, all_comps, all_path_comms, all_ancestors, pure_comm, bw,
                scale_path_collectives, order, last,
            )
            per_factor[bw] = last_done + tail - origin
            crit_len[bw] = len(critical_path)
            crit_split[bw] = {"comm": split["comm"], "comp": split["comp"] - origin}

        total_kernel = sum(all_comms[r][j][0] for r in group for j in range(num_events))
        total_pure = sum(pure_comm)
        fixed_wait = total_kernel - len(group) * total_pure
        comm_by_factor = {
            bw: fixed_wait + len(group) * total_pure / bw for bw in bw_factors
        }

        results[tuple(group)] = {
            "comm_by_factor": comm_by_factor,
            "total_kernel_comm": total_kernel,
            "group_size": len(group),
            "e2e_by_factor": per_factor,
            "critical_path_len_by_factor": crit_len,
            "critical_path_split_by_factor": crit_split,
            "num_reachable": len(order),
            "tail": tail,
            "origin": origin,
            "measured_e2e": measured_e2e,
            "total_pure_comm": sum(pure_comm),
            "num_events": num_events,
        }

    del parsed, all_comms, all_comps, all_ancestors, all_path_comms
    gc.collect()
    return subdir, results


def run_workload(workload, data_dir, exclude_pre_profiler, num_workers,
                 scale_path_collectives, bw_factors=None, factor_info=None):
    bw_factors = list(BW_FACTORS if bw_factors is None else bw_factors)
    workload_dir = os.path.join(data_dir, workload)
    comm_stream = detect_comm_stream(workload_dir)
    print(f"\n=== {workload}  (stream: {comm_stream})")

    subdir_paths = sorted(
        os.path.join(workload_dir, d)
        for d in os.listdir(workload_dir)
        if os.path.isdir(os.path.join(workload_dir, d))
    )
    attempts = [True, False] if exclude_pre_profiler is None else [exclude_pre_profiler]

    per_iteration = {}
    for attempt, exclude in enumerate(attempts):
        worker = partial(process_iteration,
                         comm_stream_to_analyze=comm_stream,
                         exclude_pre_profiler=exclude,
                         bw_factors=bw_factors,
                         scale_path_collectives=scale_path_collectives)
        per_iteration = {}
        try:
            if num_workers == 1:
                for subdir, results in map(worker, subdir_paths):
                    per_iteration[subdir] = results
            else:
                with multiprocessing.Pool(processes=num_workers,
                                          maxtasksperchild=1) as pool:
                    for subdir, results in pool.imap_unordered(worker, subdir_paths):
                        per_iteration[subdir] = results
        except AssertionError as e:
            if attempt + 1 < len(attempts):
                print(f"  exclude_pre_profiler={exclude} failed ({e}); retrying")
                continue
            raise
        print(f"  exclude_pre_profiler={exclude}")
        break

    samples = defaultdict(list)
    normalized = defaultdict(list)
    comm_samples = defaultdict(list)
    comm_normalized = defaultdict(list)
    wait_share = []
    measured, crit_len, reachable, total_colls = [], [], [], []
    for results in per_iteration.values():
        for entry in results.values():
            base = entry["e2e_by_factor"][1.0]
            for bw, e2e in entry["e2e_by_factor"].items():
                samples[bw].append(e2e)
                normalized[bw].append(e2e / base)
            comm_base = entry["comm_by_factor"][1.0]
            for bw, c in entry["comm_by_factor"].items():
                comm_samples[bw].append(c)
                comm_normalized[bw].append(c / comm_base)

            transfer = entry["group_size"] * entry["total_pure_comm"]
            wait_share.append(1 - transfer / entry["total_kernel_comm"])
            measured.append(entry["measured_e2e"])
            crit_len.append(entry["critical_path_len_by_factor"][1.0])
            reachable.append(entry["num_reachable"])
            total_colls.append(entry["num_events"])

    n = len(measured)
    print(f"  iterations x groups:              {n}")
    print(f"  collectives on {comm_stream}:{'':<{max(0, 18 - len(comm_stream))}}"
          f"{np.mean(total_colls):9.1f}")
    print(f"  reachable from final collective:  {np.mean(reachable):9.1f}")
    print(f"  collectives on critical path:     {np.mean(crit_len):9.1f}")
    print(f"  measured mean step time:          {np.mean(measured):9.1f} us")
    print(f"  replayed mean step time (1.0x):   {np.mean(samples[1.0]):9.1f} us "
          f"({100 * (np.mean(samples[1.0]) / np.mean(measured) - 1):+.2f}%)")
    for bw in bw_factors:
        vals = np.array(normalized[bw])
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        print(f"    {bw:>4g}x BW -> normalized step time {vals.mean():.4f} "
              f"+/- {sem:.4f} (SEM)   [{np.mean(samples[bw]):.1f} us]")
    print(f"  wait share of collective-kernel time: {100 * np.mean(wait_share):.1f}%")
    for bw in bw_factors:
        vals = np.array(comm_normalized[bw])
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        print(f"    {bw:>4g}x BW -> normalized comm time {vals.mean():.4f} "
              f"+/- {sem:.4f} (SEM)   [ceiling if all transfer: {1 / bw:.4f}]")

    return {
        "comm_stream": comm_stream,
        "bw_factors": bw_factors,
        "factor_info": factor_info,
        "per_iteration": per_iteration,
        "e2e_by_factor": {b: samples[b] for b in bw_factors},
        "normalized_by_factor": {b: normalized[b] for b in bw_factors},
        "comm_by_factor": {b: comm_samples[b] for b in bw_factors},
        "normalized_comm_by_factor": {b: comm_normalized[b] for b in bw_factors},
        "wait_share": wait_share,
        "measured_e2e": measured,
        "critical_path_len": crit_len,
        "num_reachable": reachable,
        "num_collectives": total_colls,
    }


def save_figure(out_path):
    plt.savefig(out_path, format="pdf")
    png_path = os.path.splitext(out_path)[0] + ".png"
    plt.savefig(png_path, format="png", dpi=200)
    print(f"\nSaved {out_path} and {png_path}")


def workload_factors(per_workload, workload, fallback):
    return list(per_workload[workload].get("bw_factors") or fallback)


def _step_bar_panel(ax, workloads, per_workload, series, fallback_factors,
                    key, ylabel, ylim, fs):
    bar_colors = [COLORS2HEX["petrol"], COLORS2HEX["terracotta"], COLORS2HEX["gold"],
                  COLORS2HEX["teal"], COLORS2HEX["sand"]]
    x = np.arange(len(workloads))
    width = BAR_GROUP_WIDTH / len(series)

    for i, (idx, label) in enumerate(series):
        means, sems = [], []
        for w in workloads:
            factors = workload_factors(per_workload, w, fallback_factors)
            vals = np.array(per_workload[w][key][factors[idx]])
            means.append(vals.mean())
            sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=sems, capsize=4,
               color=bar_colors[idx % len(bar_colors)], label=label)

    ax.axhline(1.0, linestyle="--", linewidth=1.2, color="0.4", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([compact_label(w) for w in workloads], fontsize=fs["tick"])
    ax.set_ylabel(ylabel, fontsize=fs["label"])
    ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=fs["tick"])
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def plot_bars(workloads, per_workload, out_path, series, fallback_factors,
              key="normalized_by_factor", ylabel=STEP_YLABEL, ylim=BAR_YLIM):
    plt.rcParams.update({
        "font.size": 16,
        "font.family": "Fira Code",
        "axes.prop_cycle": cycler(color=list(COLORS2HEX.values())),
    })
    fig, ax = plt.subplots(figsize=PAPER_FIG_SIZE)
    _step_bar_panel(ax, workloads, per_workload, series, fallback_factors, key,
                    ylabel, ylim, FS_STANDALONE)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=len(series),
               fontsize=FS_LEGEND, frameon=False, bbox_to_anchor=(0.5, 1.01),
               bbox_transform=ax.transAxes, columnspacing=1.2, handlelength=1.2)

    fig.subplots_adjust(left=0.175, right=0.985, top=0.87, bottom=0.235)
    save_figure(out_path)
    plt.close()


def _bar_panel(ax, workloads, values, errs, series, colors, width, xtick_size):
    x = np.arange(len(workloads))
    for i, (idx, label) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(x + offset, values[i], width, yerr=errs[i], capsize=3,
               color=colors[idx % len(colors)], label=label)
    ax.axhline(1.0, linestyle="--", linewidth=1.2, color="0.4", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_label(w) for w in workloads], fontsize=xtick_size)
    ax.tick_params(axis="y", labelsize=14)


def plot_step_and_bandwidth(workloads, per_workload, out_path, series,
                            fallback_factors, ylim=None):
    plt.rcParams.update({
        "font.size": 16,
        "font.family": "Fira Code",
        "axes.prop_cycle": cycler(color=list(COLORS2HEX.values())),
    })
    colors = [COLORS2HEX["petrol"], COLORS2HEX["terracotta"], COLORS2HEX["gold"],
              COLORS2HEX["teal"], COLORS2HEX["sand"]]
    width = BAR_GROUP_WIDTH / len(series)

    step_v, step_e, bw_v = [], [], []
    for idx, _ in series:
        mv, me, bv = [], [], []
        for w in workloads:
            factors = workload_factors(per_workload, w, fallback_factors)
            vals = np.array(per_workload[w]["normalized_by_factor"][factors[idx]])
            mv.append(vals.mean())
            me.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            bv.append(factors[idx])
        step_v.append(mv); step_e.append(me); bw_v.append(bv)

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_PER_WORKLOAD * len(workloads) * 1.6 + 3.0, 5.0))

    _bar_panel(ax0, workloads, step_v, step_e, series, colors, width, 12)
    ax0.set_ylabel("Normalized step time", fontsize=17)
    if ylim is not None:
        ax0.set_ylim(*ylim)
    ax0.set_title("(a) Step time achieved", fontsize=17)

    _bar_panel(ax1, workloads, bw_v, [None] * len(series), series, colors, width, 12)
    ax1.set_yscale("log")
    ax1.set_ylabel("Bandwidth / today's peak", fontsize=17)
    ax1.set_ylim(0.6, 400)
    ax1.set_title("(b) Bandwidth required", fontsize=17)

    xs = np.arange(len(workloads))
    for i, (idx, _) in enumerate(series):
        if idx == 0:
            continue
        offset = (i - (len(series) - 1) / 2) * width
        for xi, v in zip(xs, bw_v[i]):
            ax1.text(xi + offset, v * 1.15, f"{v:.1f}x" if v < 10 else f"{v:.0f}x",
                     ha="center", va="bottom", fontsize=10, rotation=90)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(series), fontsize=14,
               frameon=False, bbox_to_anchor=(0.5, 1.06), columnspacing=1.5,
               handlelength=1.4)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"\nSaved {out_path}")


def _knee_panel(ax, workloads, per_workload, fallback_factors, series_key,
                ylabel, fs):

    arch_color = {"a100": COLORS2HEX["petrol"], "h100": COLORS2HEX["gold"],
                  "h200": COLORS2HEX["teal"]}
    model_style = {"llama70b": "-", "qwen32b": "--", "llama8b": ":"}
    palette = list(COLORS2HEX.values())

    all_x = []
    for i, w in enumerate(workloads):
        factors = workload_factors(per_workload, w, fallback_factors)
        parts = w.split("-")
        color = next((c for a, c in arch_color.items() if a in parts),
                     palette[i % len(palette)])
        style = next((st for m, st in model_style.items() if m in parts), "-")
        xs = sorted(factors)
        all_x.extend(xs)
        ys, es = [], []
        for f in xs:
            vals = np.array(per_workload[w][series_key][f])
            ys.append(vals.mean())
            es.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        ax.errorbar(xs, ys, yerr=es, color=color, linestyle=style, linewidth=2.5,
                    marker="o", markersize=3.5, capsize=2)

        if per_workload[w].get("factor_info") and len(factors) >= 3:
            for idx, marker, size in ((2, "D", 11), (1, "^", 13)):
                f = factors[idx]
                ax.plot([f], [np.mean(per_workload[w][series_key][f])], marker=marker,
                        markersize=size, color=color, markeredgecolor="black",
                        markeredgewidth=1.0, linestyle="none", zorder=5)

    ax.set_ylim(*KNEE_YLIM)
    ax.set_xscale("log")
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100) if t <= max(all_x) * 1.3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}x" for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(KNEE_XLABEL, fontsize=fs["label"])
    ax.set_ylabel(ylabel, fontsize=fs["label"])
    ax.tick_params(axis="both", labelsize=fs["tick"])
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    parts_all = {p for w in workloads for p in w.split("-")}
    handles = [Line2D([], [], color=arch_color[a], linewidth=3, label=HW_NAMES[a])
           for a in ("a100", "h100", "h200") if a in parts_all]
    models = [Line2D([], [], color="0.35", linestyle=model_style[m], linewidth=2.5,
                     label=MODEL_NAMES[m])
              for m in ("llama70b", "qwen32b", "llama8b") if m in parts_all]
    handles += [Line2D([], [], linestyle="none", label="")] * max(
        0, len(models) + 2 - len(handles))
    handles += models
    handles += [
        Line2D([], [], marker="D", markersize=9, linestyle="none", color="0.35",
               markeredgecolor="black", label="B* w/ sync tax"),
        Line2D([], [], marker="^", markersize=11, linestyle="none", color="0.35",
               markeredgecolor="black", label="B* no sync tax"),
    ]
    return handles


def plot_knee(workloads, per_workload, out_path, fallback_factors,
              key="normalized_by_factor", ylabel=STEP_YLABEL):
    plt.rcParams.update({
        "font.size": 16,
        "font.family": "Fira Code",
        "axes.prop_cycle": cycler(color=list(COLORS2HEX.values())),
    })
    fig, ax = plt.subplots(figsize=PAPER_FIG_SIZE)
    handles = _knee_panel(ax, workloads, per_workload, fallback_factors, key,
                          ylabel, FS_STANDALONE)

    ax.legend(handles, [h.get_label() for h in handles], fontsize=FS_LEGEND,
              frameon=False, loc="lower left", ncol=2, columnspacing=1.1,
              handlelength=1.5, borderaxespad=0.6)

    fig.subplots_adjust(left=0.175, right=0.985, top=0.97, bottom=0.185)
    save_figure(out_path)
    plt.close()


def plot_pair(bar_workloads, bar_per_workload, bar_series, bar_factors,
              knee_workloads, knee_per_workload, knee_factors, out_path):
    plt.rcParams.update({
        "font.size": 18,
        "font.family": "Fira Code",
        "axes.prop_cycle": cycler(color=list(COLORS2HEX.values())),
    })
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=PAIR_FIG_SIZE)

    _step_bar_panel(ax0, bar_workloads, bar_per_workload, bar_series, bar_factors,
                    "normalized_by_factor", STEP_YLABEL, PAIR_BAR_YLIM, FS_PAIR)
    ax0.legend(*ax0.get_legend_handles_labels(), loc="lower center",
               ncol=len(bar_series), fontsize=FS_PAIR["legend"], frameon=False,
               bbox_to_anchor=(0.5, 0.985), columnspacing=1.2, handlelength=1.2)

    handles = _knee_panel(ax1, knee_workloads, knee_per_workload, knee_factors,
                          "normalized_by_factor", STEP_YLABEL, FS_PAIR)
    ax1.legend(handles, [h.get_label() for h in handles],
               fontsize=FS_PAIR["legend"], frameon=False, loc="lower left", ncol=2,
               columnspacing=1.0, handlelength=1.4, borderaxespad=0.15,
               labelspacing=0.3)
    fig.subplots_adjust(left=0.095, right=0.995, top=0.91, bottom=0.285, wspace=0.20)
    for ax, title in zip((ax0, ax1), PAIR_TITLES):
        box = ax.get_position()
        fig.text(box.x0 + box.width / 2, 0.015, title, ha="center", va="bottom",
                 fontsize=FS_PAIR["label"])
    save_figure(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Replay workload critical paths under reduced bandwidth.")
    parser.add_argument("workloads", nargs="*")
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--results", default=None)
    parser.add_argument("--optimal-bw", action="store_true", default=False)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--exclude-pre-profiler", dest="exclude_pre_profiler",
                        action="store_const", const=True, default=None)
    parser.add_argument("--no-exclude-pre-profiler", dest="exclude_pre_profiler",
                        action="store_const", const=False)
    parser.add_argument("--pair-with", default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--no-scale-path-collectives", action="store_true", default=False)
    args = parser.parse_args()

    assert BW_FACTORS[0] == 1.0, "BW_FACTORS[0] must be 1.0 (the normalization baseline)"
    if args.results is None:
        args.results = ("<OUTPUT_FOLDER>" if args.optimal_bw
                        else "<OUTPUT_FOLDER>")

    if args.replot:
        with open(args.results, "rb") as f:
            saved = pickle.load(f)
        per_workload = saved["per_workload"]
        workloads = args.workloads or saved["workloads"]
        missing = [w for w in workloads if w not in per_workload]
        if missing:
            raise SystemExit(f"{args.results} has no results for: {', '.join(missing)}")
        if saved.get("optimal_bw", False) != args.optimal_bw:
            raise SystemExit(
                f"{args.results} was replayed with optimal_bw={saved.get('optimal_bw', False)};")
        if not args.optimal_bw and list(saved["bw_factors"]) != list(BW_FACTORS):
            raise SystemExit(
                f"BW_FACTORS is {BW_FACTORS} but {args.results} was replayed with "
                f"{saved['bw_factors']}; re-run the replay or restore the constant.")
        print(f"Replotting {len(workloads)} workloads from {args.results}")
    else:
        if not args.workloads:
            raise SystemExit("give at least one workload, or pass --replot")
        workloads = args.workloads
        per_workload = {}
        for workload in workloads:
            factors, info = (None, None)
            if args.optimal_bw:
                print(f"\n=== {workload}")
                factors, info = optimal_bw_factors(workload)
            per_workload[workload] = run_workload(
                workload, args.data_dir, args.exclude_pre_profiler, args.num_workers,
                not args.no_scale_path_collectives, bw_factors=factors,
                factor_info=info)

    os.makedirs(os.path.dirname(PAPER_FIG), exist_ok=True)
    if args.optimal_bw:
        plot_step_and_bandwidth(workloads, per_workload, PAPER_FIG_OPT, OPT_SERIES,
                                BW_FACTORS, ylim=OPT_YLIM)
        plot_knee(workloads, per_workload, PAPER_FIG_KNEE, BW_FACTORS)
    else:
        series = [(i, factor_label(f)) for i, f in enumerate(BW_FACTORS)]
        plot_bars(workloads, per_workload, PAPER_FIG, series, BW_FACTORS)
        if args.pair_with:
            with open(args.pair_with, "rb") as f:
                knee_saved = pickle.load(f)
            plot_pair(workloads, per_workload, series, BW_FACTORS,
                      knee_saved["workloads"], knee_saved["per_workload"],
                      BW_FACTORS, PAPER_FIG_PAIR)

    if not args.replot:
        os.makedirs(os.path.dirname(args.results), exist_ok=True)
        with open(args.results, "wb") as f:
            pickle.dump({
                "bw_factors": BW_FACTORS,
                "optimal_bw": args.optimal_bw,
                "workloads": workloads,
                "scale_path_collectives": not args.no_scale_path_collectives,
                "per_workload": per_workload,
            }, f)
        print(f"Saved {args.results}")


if __name__ == "__main__":
    main()
