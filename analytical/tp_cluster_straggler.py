import argparse
import json
import os
import pickle
from collections import Counter, defaultdict
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DATA_ROOT = "<DATA_PATH>"
WORKLOAD = "<WORKLOAD_NAME>"
GPUS = ["a100", "h100", "h200"]

TP_DESC = "mesh_tp"
FSDP_DESCS = ("mesh_dp_shard_cp", "mesh_dp_shard")

CACHE_PATH = "<CASH_PATH>"

def parse_rank(task):
    iter_dir, rank = task
    path = os.path.join(iter_dir, f"rank{rank}_trace.json")
    with open(path, "r") as fp:
        data = json.load(fp)
    pgs = {pg["pg_name"]: (pg["pg_desc"], tuple(pg["ranks"]))
           for pg in data.get("distributedInfo", {}).get("pg_config", [])}
    events = [e for e in data.get("traceEvents", [])
              if "ncclDevKernel" in e.get("name", "") and e.get("cat") == "kernel"]
    stream_pg = {}
    for e in events:
        pg_id = e.get("args", {}).get("Process Group Name")
        if pg_id in pgs:
            stream_pg[e["args"]["stream"]] = pgs[pg_id]
    by_pg = defaultdict(list)
    for e in events:
        pg = stream_pg.get(e.get("args", {}).get("stream"))
        if pg is None or (pg[0] != TP_DESC and pg[0] not in FSDP_DESCS):
            continue
        by_pg[pg].append((e["ts"], e["dur"], e["name"]))
    for pg in by_pg:
        by_pg[pg].sort(key=lambda x: x[0])
    return rank, dict(by_pg)

def parse_iteration(iter_dir, pool):
    ranks = sorted(int(f[4:f.index("_")]) for f in os.listdir(iter_dir)
                   if f.startswith("rank") and f.endswith("_trace.json"))
    results = pool.map(parse_rank, [(iter_dir, r) for r in ranks])
    return {r: pgs for r, pgs in results}

def build_cache(gpus, jobs):
    cache = {}
    with Pool(jobs) as pool:
        for gpu in gpus:
            root = os.path.join(DATA_ROOT, f"{WORKLOAD}-{gpu}")
            if not os.path.isdir(root):
                print(f"[skip] missing {root}")
                continue
            iters = sorted(d for d in os.listdir(root)
                           if os.path.isdir(os.path.join(root, d)))
            for it in iters:
                iter_dir = os.path.join(root, it)
                print(f"  parsing {gpu}/{it}", flush=True)
                cache[(gpu, it)] = parse_iteration(iter_dir, pool)
    return cache

def load_cache(gpus, jobs, refresh):
    if not refresh and os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, "rb") as fp:
            cache = pickle.load(fp)
        if all(any(g == gpu for g, _ in cache) for gpu in gpus):
            print(f"Loaded cache {CACHE_PATH} ({len(cache)} iterations)")
            return cache
    cache = build_cache(gpus, jobs)
    with open(CACHE_PATH, "wb") as fp:
        pickle.dump(cache, fp)
    print(f"Wrote cache {CACHE_PATH} ({len(cache)} iterations)")
    return cache

TP_GROUPS = [(0, 1, 2, 3), (4, 5, 6, 7)]
GPU_COLOR = {"a100": "#2a78d6", "h100": "#eb6834", "h200": "#199e70"}
GPU_LABEL = {"a100": "A100", "h100": "H100", "h200": "H200"}

def align(seqs):
    ok = len({len(x) for x in seqs}) == 1
    n = min(len(x) for x in seqs)
    bad = [i for i in range(n) if len({x[i][2] for x in seqs}) != 1]
    rows = np.array([[x[i][1] for x in seqs] for i in range(n) if i not in set(bad)])
    return (rows if len(rows) else None), len(bad), ok

def tp_durations(per_rank, group):
    seqs = [per_rank.get(r, {}).get((TP_DESC, group)) for r in group]
    if any(s is None or not s for s in seqs):
        return None, 0, False
    return align(seqs)

def fsdp_durations(per_rank, align_log):
    out = {}
    for k in range(4):
        pair = (k, k + 4)
        for desc in FSDP_DESCS:
            seqs = [per_rank.get(r, {}).get((desc, pair)) for r in pair]
            if any(s is None or not s for s in seqs):
                continue
            rows, nskip, resolved = align(seqs)
            if rows is not None:
                out[k] = rows
                align_log.append((nskip, resolved))
            break
    return out

def tax(durs):
    lo = durs.min(axis=1)
    return (durs.max(axis=1) - lo) / lo

def total_wait(tp_durs, fsdp_durs):
    W = np.zeros(8)
    for group, durs in zip(TP_GROUPS, tp_durs):
        idle = durs - durs.min(axis=1, keepdims=True)
        for col, r in enumerate(group):
            W[r] += idle[:, col].sum()
    for k, durs in fsdp_durs.items():
        idle = durs - durs.min(axis=1, keepdims=True)
        W[k] += idle[:, 0].sum()
        W[k + 4] += idle[:, 1].sum()
    return W

def analyze(cache, gpus):
    stats = {}
    for gpu in gpus:
        iters = sorted(it for g, it in cache if g == gpu)
        host, peer, tp_all, fsdp_all = [], [], [], []
        straggler_counts = Counter()
        skipped = []
        n_skip = n_unresolved = n_pg = 0 
        pos_agree, pos_n = 0, 0         
        p90_a, p90_b = [], []            
        for it in iters:
            per_rank = cache[(gpu, it)]
            align_log = []
            tp_aligned = [tp_durations(per_rank, g) for g in TP_GROUPS]
            tp_durs = [d for d, _, _ in tp_aligned]
            align_log.extend((n, ok) for _, n, ok in tp_aligned)
            fsdp_durs = fsdp_durations(per_rank, align_log)
            n_skip += sum(n for n, _ in align_log)
            n_unresolved += sum(1 for _, ok in align_log if not ok)
            n_pg += len(align_log)
            if any(d is None for d in tp_durs) or len(fsdp_durs) < 4:
                skipped.append(it)
                continue

            W = total_wait(tp_durs, fsdp_durs)
            rstar = int(W.argmin())
            straggler_counts[rstar] += 1
            host_idx = 0 if rstar in TP_GROUPS[0] else 1
            host.append(tax(tp_durs[host_idx]))
            peer.append(tax(tp_durs[1 - host_idx]))
            tp_all.extend(tax(d) for d in tp_durs)
            fsdp_all.extend(tax(d) for d in fsdp_durs.values())

            n = min(len(tp_durs[0]), len(tp_durs[1]))
            pos_agree += int((tp_durs[0][:n].argmin(axis=1)
                              == tp_durs[1][:n].argmin(axis=1)).sum())
            pos_n += n
            p90_a.append(np.percentile(tax(tp_durs[0]), 90))
            p90_b.append(np.percentile(tax(tp_durs[1]), 90))

        if not host:
            print(f"[skip] no usable iterations for {gpu}")
            continue
        stats[gpu] = {
            "host": np.concatenate(host),
            "peer": np.concatenate(peer),
            "tp": np.concatenate(tp_all),
            "fsdp": np.concatenate(fsdp_all),
            "stragglers": straggler_counts,
            "iterations": len(host),
            "skipped": skipped,
            "align_skipped": n_skip,
            "align_unresolved": n_unresolved,
            "align_pgs": n_pg,
            "pos_agree": pos_agree / pos_n if pos_n else float("nan"),
            "pos_n": pos_n,
            "p90_corr": (float(np.corrcoef(p90_a, p90_b)[0, 1])
                         if len(p90_a) > 2 else float("nan")),
        }
    return stats

def wait_pct(taxes):
    t = np.asarray(taxes, dtype=float)
    return 100.0 * t / (1.0 + t)

def cdf(values):
    a = np.sort(np.asarray(values, dtype=float))
    return a, np.arange(1, a.size + 1) / a.size

def plot(stats, gpus, out_dir, xmax=100.0):
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 20, "axes.grid": True,
                         "grid.color": "#d8d8d6", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8a86", "axes.linewidth": 0.8})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    panels = [
        (axes[0], ("host", "peer"),
         ("TP group with cluster straggler", "TP group without it"),
         "(a) Sync tax by TP group"),
        (axes[1], ("tp", "fsdp"),
         ("TP collectives", "FSDP collectives"),
         "(b) Sync tax in TP vs. FSDP"),
    ]

    for ax, keys, style_labels, title in panels:
        for gpu in gpus:
            if gpu not in stats:
                continue
            for key, ls in zip(keys, ("-", "--")):
                x, y = cdf(wait_pct(stats[gpu][key]))
                ax.plot(x, y, color=GPU_COLOR[gpu], linestyle=ls, linewidth=2.5)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1)
        ax.set_xlabel("% of Comm. Time Spent Waiting", fontsize=22)
        ax.set_title(title, fontsize=22)
        ax.tick_params(labelsize=19)
        ax.set_axisbelow(True)

        style_handles = [Line2D([], [], color="#3d3d3a", linestyle=ls, linewidth=2.5,
                                label=lab)
                         for ls, lab in zip(("-", "--"), style_labels)]
        ax.legend(handles=style_handles, loc="lower right", fontsize=18,
                  frameon=False)

    axes[0].set_ylabel("CDF of collective instances", fontsize=22)
    gpu_handles = [Line2D([], [], color=GPU_COLOR[g], linewidth=3,
                          label=GPU_LABEL[g])
                   for g in gpus if g in stats]
    fig.legend(handles=gpu_handles, loc="upper center", ncol=len(gpu_handles),
               fontsize=21, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=3.0)

    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"tp_cluster_straggler.{ext}")
        fig.savefig(path, format=ext, bbox_inches="tight", dpi=200)
        print(f"Saved {path}")
    plt.close(fig)

def write_report(stats, gpus, out_dir):
    path = os.path.join(out_dir, "tp_cluster_straggler.txt")
    with open(path, "w") as fp:
        fp.write(f"Workload: {WORKLOAD}\n")
        header = (f"{'gpu':<7}{'series':<8}{'n':>9}{'median':>9}{'p90':>9}"
                  f"{'p99':>9}{'mean':>9}\n")
        fp.write(header)
        fp.write("-" * (len(header) - 1) + "\n")
        for gpu in gpus:
            if gpu not in stats:
                continue
            for key in ("host", "peer", "tp", "fsdp"):
                a = stats[gpu][key]
                fp.write(f"{gpu:<7}{key:<8}{a.size:>9}{np.median(a):>9.3f}"
                         f"{np.percentile(a, 90):>9.3f}"
                         f"{np.percentile(a, 99):>9.3f}{a.mean():>9.3f}\n")
            fp.write("\n")
        for gpu in gpus:
            if gpu not in stats:
                continue
            h, p = stats[gpu]["host"], stats[gpu]["peer"]
            fp.write(f"  {gpu}: median host {np.median(h):.3f} vs peer "
                     f"{np.median(p):.3f} "
                     f"(peer/host = {np.median(p) / np.median(h):.2f}); "
                     f"p90 {np.percentile(h, 90):.3f} vs "
                     f"{np.percentile(p, 90):.3f}\n")
        for gpu in gpus:
            if gpu not in stats:
                continue
            t, f = stats[gpu]["tp"], stats[gpu]["fsdp"]
            fp.write(f"  {gpu}: median TP {np.median(t):.3f} vs FSDP "
                     f"{np.median(f):.3f} "
                     f"(FSDP/TP = {np.median(f) / np.median(t):.1f}x)\n")
        for gpu in gpus:
            if gpu not in stats:
                continue
            c = stats[gpu]["stragglers"]
            counts = ", ".join(f"rank {r}: {n}" for r, n in sorted(c.items()))
            fp.write(f"  {gpu} ({stats[gpu]['iterations']} iterations): {counts}\n")
            if stats[gpu]["skipped"]:
                fp.write(f"    skipped: {', '.join(stats[gpu]['skipped'])}\n")
        for gpu in gpus:
            if gpu not in stats:
                continue
        for gpu in gpus:
            if gpu not in stats:
                continue
            fp.write(f"       {gpu}: {stats[gpu]['pos_agree']:.3f} "
                     f"(n={stats[gpu]['pos_n']})\n")
        for gpu in gpus:
            if gpu not in stats:
                continue
            fp.write(f"       {gpu}: r = {stats[gpu]['p90_corr']:+.3f}\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--gpus", nargs="+", default=GPUS)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "plots", "tp_cluster_straggler"))
    args = ap.parse_args()

    cache = load_cache(args.gpus, args.jobs, args.refresh)
    stats = analyze(cache, args.gpus)
    if not stats:
        raise SystemExit("No usable iterations.")
    os.makedirs(args.out, exist_ok=True)
    plot(stats, args.gpus, args.out)
    write_report(stats, args.gpus, args.out)
