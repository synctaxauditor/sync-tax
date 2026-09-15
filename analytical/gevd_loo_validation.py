import argparse
import math
import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
from scipy import stats

COLORS2HEX = {
    "petrol":     "#264653",
    "gold":       "#E9C46A",
    "teal":       "#2A9D8F",
    "sand":       "#F4A261",
    "terracotta": "#E76F51",
}

WORKLOADS = {
    "n4-tp-llama8b": {
        "fit_groups": [[0, 1, 2, 3]],
        "val_spec": (4, 4),
        "title": r"Llama-3$\,$8B, TP=4",
    },
    "n8-tp-qwen32b": {
        "fit_groups": [[0, 2, 4, 6], [1, 3, 5, 7]],
        "val_spec": (8, 8),
        "title": r"Qwen-3$\,$32B, TP=8",
    },
    "n8-fsdp-tp-llama70b": {
        "fit_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "val_spec": (8, 4),
        "title": r"Llama-3$\,$70B, TP=4$\,$x$\,$FSDP=2",
    },
}
ARCHS = ["a100", "h100", "h200"]
IGNORE = {"CPU", "MemEffAttention", "sdpa"}

ALIASES = {
    "llama8b": "n4-tp-llama8b",
    "qwen32b": "n8-tp-qwen32b",
    "llama70b": "n8-fsdp-tp-llama70b",
}

ALL_UNITS = [(w, a) for w in WORKLOADS for a in ARCHS]

PAPER_PANELS = [(w, a) for w in ["n8-tp-qwen32b", "n8-fsdp-tp-llama70b"] for a in ARCHS]

def unit_name(unit):
    return f"{unit[0]}-{unit[1]}"

def parse_unit(spec):
    s = spec.strip().lower()
    for arch in ARCHS:
        if not s.endswith("-" + arch):
            continue
        wl = s[: -(len(arch) + 1)]
        wl = ALIASES.get(wl, wl)
        if wl in WORKLOADS:
            return (wl, arch)
        raise SystemExit(f"unknown workload '{wl}' in '{spec}'. "
                         f"Known: {sorted(WORKLOADS)} or aliases {sorted(ALIASES)}")
    raise SystemExit(f"could not parse unit '{spec}'; expected <workload>-<arch> "
                     f"with arch in {ARCHS}")

def short_name(unit):
    inv = {v: k for k, v in ALIASES.items()}
    return f"{inv.get(unit[0], unit[0])}-{unit[1]}"

def short_set(units):
    if set(units) == set(ALL_UNITS):
        return "all"
    parts, seen = [], set()
    for wl in WORKLOADS:
        archs = [u[1] for u in units if u[0] == wl]
        if not archs:
            continue
        seen.update((wl, a) for a in archs)
        base = {v: k for k, v in ALIASES.items()}.get(wl, wl)
        parts.append(base if len(archs) == len(ARCHS) else f"{base}-{'+'.join(archs)}")
    return "_".join(parts) if parts else "none"


def parse_unit_set(specs):
    out = []
    for spec in specs:
        if spec.strip().lower() == "all":
            picked = list(ALL_UNITS)
        elif spec.strip().lower() in WORKLOADS or spec.strip().lower() in ALIASES:
            wl = ALIASES.get(spec.strip().lower(), spec.strip().lower())
            picked = [(wl, a) for a in ARCHS]
        elif spec.strip().lower() in ARCHS:
            picked = [(w, spec.strip().lower()) for w in WORKLOADS]
        else:
            picked = [parse_unit(spec)]
        for u in picked:
            if u not in out:
                out.append(u)
    return out

def trace_path(workload, arch):
    return f"<DATA_PATH/{workload}-{arch}.pkl"

def collect_unit(workload, arch):
    with open(trace_path(workload, arch), "rb") as f:
        path_traces = pickle.load(f)

    samples_by_combo = defaultdict(list)
    samples_by_kernel = defaultdict(list)
    groups = WORKLOADS[workload]["fit_groups"]

    for _iter, iter_data in path_traces.items():
        for group in groups:
            comm_lens = {r: len(iter_data["comm"][r]) for r in group}
            comp_lens = {r: len(iter_data["comp"][r]) for r in group}
            if len(set(comm_lens.values())) != 1:
                continue
            if len(set(comp_lens.values())) != 1:
                continue

            for ev_idx in range(comp_lens[group[0]]):
                paths = {r: iter_data["comp"][r][ev_idx] for r in group}
                if any(len(paths[r]) == 0 for r in group):
                    continue
                op_names = {r: tuple(name for name, _ in paths[r]) for r in group}
                if len(set(op_names.values())) != 1:
                    continue
                combo = next(iter(op_names.values()))
                if any(sub in name for name in combo for sub in IGNORE):
                    continue

                samples_by_combo[combo].append(
                    [sum(dur for _, dur in paths[r]) / 1000.0 for r in group])
                for k_idx in range(len(paths[group[0]])):
                    k_name = paths[group[0]][k_idx][0]
                    samples_by_kernel[k_name].append(
                        [paths[r][k_idx][1] / 1000.0 for r in group])

    return dict(samples_by_combo), dict(samples_by_kernel)


def collect_units(units):
    raw = {}
    for unit in units:
        raw[unit] = collect_unit(*unit)
        n_combo = sum(len(v) for v in raw[unit][0].values())
        print(f"  loaded {unit_name(unit)}: {len(raw[unit][0])} combos, "
              f"{len(raw[unit][1])} kernels, {n_combo} combo-observations")
    return raw


def summarize(raw, units, per_kernel):
    by_key = defaultdict(list)
    for unit in units:
        by_combo, by_kernel = raw[unit]
        src = by_kernel if per_kernel else by_combo
        for key, samples in src.items():
            by_key[(key, unit[1])].extend(samples)

    out = {}
    for key, samples in by_key.items():
        num, den = 0.0, 0
        for s in samples:
            if len(s) > 1:
                num += (len(s) - 1) * float(np.var(s, ddof=1))
                den += len(s) - 1
        pooled_std = float(np.sqrt(num / den)) if den > 0 else 0.0
        flat = [v for s in samples for v in s]
        out[key] = {
            "maxes": [float(max(s)) for s in samples],
            "sample_means": [float(np.mean(s)) for s in samples],
            "pooled_std": pooled_std,
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "n": len(flat),
        }
    return out

def z_scores_from(summary):
    zs, clusters = [], []
    for info in summary.values():
        if info["pooled_std"] == 0:
            continue
        zc = [(m - sm) / info["pooled_std"]
              for m, sm in zip(info["maxes"], info["sample_means"])]
        if zc:
            zs.extend(zc)
            clusters.append(np.asarray(zc, dtype=float))
    return np.asarray(zs, dtype=float), clusters


def fit_gevd(z, family):
    if family == "gumbel":
        _, loc, scale = stats.genextreme.fit(z, f0=0)
        return {"shape": 0.0, "loc": float(loc), "scale": float(scale)}
    c, loc, scale = stats.genextreme.fit(z)
    return {"shape": float(c), "loc": float(loc), "scale": float(scale)}


def fit_group_size(units):
    sizes = {len(g) for u in units for g in WORKLOADS[u[0]]["fit_groups"]}
    assert len(sizes) == 1, f"fit groups have mixed sizes {sizes}; m is ambiguous"
    return sizes.pop()

def calc_expected_value_gumbel(mu, sigma, k=1):
    gamma = 0.5772156649  # Euler-Mascheroni constant
    return mu + (sigma * (gamma + np.log(k)))

def calc_expected_value_frechet(mu, sigma, kappa, k=1):
    assert kappa != 0
    kappa = -1 * kappa
    return mu + (sigma * (((k ** kappa) * math.gamma(1 - kappa)) - 1) / kappa)

def expected_z(params, k):
    if params["shape"] == 0:
        return calc_expected_value_gumbel(params["loc"], params["scale"], k)
    return calc_expected_value_frechet(params["loc"], params["scale"], params["shape"], k)

def _combo_stats(combo, arch, stats_table, per_kernel):
    if per_kernel:
        if any((k, arch) not in stats_table for k in combo):
            return None
        mean = float(sum(stats_table[(k, arch)]["mean"] for k in combo))
        std = float(math.sqrt(sum(stats_table[(k, arch)]["std"] ** 2 for k in combo)))
    else:
        if (combo, arch) not in stats_table:
            return None
        mean = float(stats_table[(combo, arch)]["mean"])
        std = float(stats_table[(combo, arch)]["std"])
    return mean, std

def compute_workload(workload, arch, stats_table, params, m, per_kernel):
    with open(trace_path(workload, arch), "rb") as f:
        path_traces = pickle.load(f)

    group_total, group_size = WORKLOADS[workload]["val_spec"]
    z = expected_z(params, group_size / m)

    predicted_delays, empirical_delays, kernel_combos = [], [], []
    for _iter, iter_data in path_traces.items():
        for start_rank in range(0, group_total, group_size):
            group = range(start_rank, start_rank + group_size)
            comm_lens = {r: len(iter_data["comm"][r]) for r in group}
            comp_lens = {r: len(iter_data["comp"][r]) for r in group}
            assert len(set(comm_lens.values())) == 1, f"comm lengths differ: {comm_lens}"
            assert len(set(comp_lens.values())) == 1, f"comp lengths differ: {comp_lens}"

            for ev_idx in range(comp_lens[start_rank]):
                op_names = {r: tuple(iter_data["comp"][r][ev_idx]) for r in group}
                if any(len(op_names[r]) == 0 for r in op_names):
                    continue
                op_names = {k: tuple(x[0] for x in v) for k, v in op_names.items()}
                if len(set(op_names.values())) != 1:
                    continue
                if any(sub in name for r in op_names for name in op_names[r] for sub in IGNORE):
                    continue
                op_name = list(op_names.values())[0]
                s = _combo_stats(op_name, arch, stats_table, per_kernel)
                if s is None:
                    continue
                _, comp_dur = s
                assert comp_dur > 0
                straggler_time = min(iter_data["comm"][r][ev_idx] for r in group)
                empirical_delay = np.mean(
                    [iter_data["comm"][r][ev_idx] - straggler_time for r in group])
                empirical_delays.append(empirical_delay / 1000.0)
                kernel_combos.append(op_name)
                predicted_delays.append(comp_dur * z)

    predicted = np.array(predicted_delays)
    empirical = np.array(empirical_delays)

    xs, means, sems = [], [], []
    for lvl in np.unique(predicted):
        vals = empirical[predicted == lvl]
        if len(vals) < 2:
            continue
        xs.append(lvl)
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(len(vals)))

    combo_to_idx = {}
    for combo in kernel_combos:
        combo_to_idx.setdefault(combo, len(combo_to_idx))
    combo_idx_arr = np.array([combo_to_idx[c] for c in kernel_combos]) if kernel_combos \
        else np.array([], dtype=int)

    bar_combos, bar_pred, bar_emp_mean, bar_emp_sem = [], [], [], []
    for combo, idx in combo_to_idx.items():
        mask = combo_idx_arr == idx
        vals = empirical[mask]
        if len(vals) < 2:
            continue
        combo_mean, combo_std = _combo_stats(combo, arch, stats_table, per_kernel)
        bar_combos.append({
            "name": combo,
            "mean": combo_mean,
            "std": combo_std,
            "cov": combo_std / combo_mean,
        })
        bar_pred.append(predicted[mask][0])
        bar_emp_mean.append(vals.mean())
        bar_emp_sem.append(vals.std(ddof=1) / np.sqrt(len(vals)))

    order = np.argsort(bar_pred) if bar_pred else np.array([], dtype=int)
    bar_pred = np.array(bar_pred)[order] if len(bar_pred) else np.array([])
    bar_emp_mean = np.array(bar_emp_mean)[order] if len(bar_emp_mean) else np.array([])
    bar_emp_sem = np.array(bar_emp_sem)[order] if len(bar_emp_sem) else np.array([])
    bar_combos = [bar_combos[i] for i in order]

    if len(bar_pred):
        rel = (bar_pred - bar_emp_mean) / bar_emp_mean
        mape = float(np.mean(np.abs(rel)) * 100)
        bias = float(np.mean(rel) * 100)
    else:
        rel = np.array([])
        mape = bias = float("nan")

    return {
        "n": len(empirical),
        "z": float(z),
        "k": group_size / m,
        "xs": np.array(xs),
        "means": np.array(means),
        "sems": np.array(sems),
        "bar_pred": bar_pred,
        "bar_emp_mean": bar_emp_mean,
        "bar_emp_sem": bar_emp_sem,
        "bar_combos": bar_combos,
        "rel": np.asarray(rel, dtype=float),
        "mape": mape,
        "bias": bias,
    }

def grid_shape(n):
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols

def plot_bars(results, eval_units, out_path):
    nrows, ncols = grid_shape(len(eval_units))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    handles = None
    for idx, unit in enumerate(eval_units):
        ax = axes[idx // ncols][idx % ncols]
        title = (f"({chr(ord('a') + idx)}) {WORKLOADS[unit[0]]['title']}, "
                 f"{unit[1].upper()}")
        r = results[unit]
        bar_pred = r["bar_pred"] * 1000.0
        bar_emp_mean = r["bar_emp_mean"] * 1000.0
        bar_emp_sem = r["bar_emp_sem"] * 1000.0
        n = len(bar_pred)
        if n == 0:
            ax.set_title(f"{title} (no data)")
            continue
        x = np.arange(n)
        width = 0.4
        ax.bar(x - width / 2, bar_emp_mean, width, yerr=bar_emp_sem, capsize=3,
               label="Mean empirical ± SEM")
        ax.bar(x + width / 2, bar_pred, width, label="Predicted (held-out fit)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in range(n)])
        ax.set_xlabel("Unique compute block", fontsize=22)
        ax.set_ylabel("Sync delay (µs)", fontsize=22)
        ax.set_title(title, fontsize=21)
        if handles is None:
            handles = ax.get_legend_handles_labels()
    for extra in range(len(eval_units), nrows * ncols):
        axes[extra // ncols][extra % ncols].axis("off")
    if handles is not None:
        # constant absolute gap below the axes, whatever the row count
        fig.legend(*handles, loc="lower center", ncol=len(handles[0]),
                   bbox_to_anchor=(0.5, -0.12 / nrows), fontsize=23, frameon=False)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved bar plot to {out_path}")

ARCH_COLOR = {"a100": COLORS2HEX["petrol"],
              "h100": COLORS2HEX["teal"],
              "h200": COLORS2HEX["sand"]}
WORKLOAD_COLOR = {"n4-tp-llama8b": COLORS2HEX["petrol"],
                  "n8-tp-qwen32b": COLORS2HEX["teal"],
                  "n8-fsdp-tp-llama70b": COLORS2HEX["sand"]}
ARCH_MARKER = {"a100": "o", "h100": "s", "h200": "^"}

FAMILY_LABEL = {"gumbel": "Gumbel", "general": "Fréchet"}
WORKLOAD_MARKER = {"n4-tp-llama8b": "^", "n8-tp-qwen32b": "o", "n8-fsdp-tp-llama70b": "s"}
WORKLOAD_SHORT = {"n4-tp-llama8b": r"Llama-3$\,$8B",
                  "n8-tp-qwen32b": r"Qwen-3$\,$32B",
                  "n8-fsdp-tp-llama70b": r"Llama-3$\,$70B"}

POOLED_MARKERSIZE = 13
POOLED_FS_LABEL = 20      # axis labels
POOLED_FS_TICK = 18       # tick labels
POOLED_FS_TITLE = 20      # panel titles
POOLED_FS_LEGEND = 20
POOLED_FS_ANNOT = 17      # the MAPE / R^2 box

def _pooled_legend_handles(eval_units, groupby):
    from matplotlib.lines import Line2D
    workloads_shown = list(dict.fromkeys(u[0] for u in eval_units))
    archs_shown = list(dict.fromkeys(u[1] for u in eval_units))
    if groupby == "workload":
        handles = [Line2D([], [], linestyle="none", marker="o", color=WORKLOAD_COLOR[w],
                          markersize=POOLED_MARKERSIZE, label=WORKLOAD_SHORT.get(w, w))
                   for w in workloads_shown]
    elif groupby == "arch":
        handles = [Line2D([], [], linestyle="none", marker="o", color=ARCH_COLOR[a],
                          markersize=POOLED_MARKERSIZE, label=a.upper()) for a in archs_shown]
    else:  # colour = workload, shape = arch
        handles = [Line2D([], [], linestyle="none", marker="o", color=WORKLOAD_COLOR[w],
                          markersize=POOLED_MARKERSIZE, label=WORKLOAD_SHORT.get(w, w))
                   for w in workloads_shown]
        handles += [Line2D([], [], linestyle="none", marker=ARCH_MARKER.get(a, "o"),
                           color="0.35", markersize=POOLED_MARKERSIZE, label=a.upper())
                    for a in archs_shown]
    handles += [Line2D([], [], linestyle="--", color=COLORS2HEX["terracotta"],
                       linewidth=2.0, label="y = x")]
    return handles


def _log_ticks(ax, lo, hi):
    from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, FuncFormatter

    anchor = 10.0 ** math.floor(math.log10(hi))  
    ticks = []
    t = anchor
    while t >= lo:            
        if t <= hi:
            ticks.append(t)
        t /= 2.0
    t = anchor * 2.0
    while t <= hi:         
        ticks.append(t)
        t *= 2.0
    ticks.sort()

    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(fmt)
        axis.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1),
                                          numticks=100))
        axis.set_minor_formatter(NullFormatter())


def _draw_pooled(ax, results, eval_units, band, groupby, lim=None, box_aspect=None):
    all_pred, all_emp = [], []
    for unit in eval_units:
        r = results[unit]
        if len(r["bar_pred"]) == 0:
            continue
        pred = r["bar_pred"] * 1000.0
        emp = r["bar_emp_mean"] * 1000.0
        sem = r["bar_emp_sem"] * 1000.0
        all_pred.extend(pred)
        all_emp.extend(emp)
        if groupby == "workload":
            marker, color = "o", WORKLOAD_COLOR.get(unit[0], COLORS2HEX["petrol"])
        elif groupby == "arch":
            marker, color = "o", ARCH_COLOR.get(unit[1], COLORS2HEX["petrol"])
        else:
            marker = ARCH_MARKER.get(unit[1], "o")
            color = WORKLOAD_COLOR.get(unit[0], COLORS2HEX["petrol"])
        ax.errorbar(pred, emp, yerr=sem, fmt=marker, color=color,
                    markersize=POOLED_MARKERSIZE, markeredgecolor="white",
                    markeredgewidth=1.0, capsize=4, elinewidth=1.4,
                    linestyle="none", zorder=3)

    pred = np.array(all_pred)
    emp = np.array(all_emp)
    if len(pred) == 0:
        raise SystemExit("no compute blocks to plot")

    lo, hi = lim if lim else (min(pred.min(), emp.min()) * 0.7,
                              max(pred.max(), emp.max()) * 1.4)
    line = np.array([lo, hi])
    ax.fill_between(line, line * (1 - band), line * (1 + band),
                    color=COLORS2HEX["terracotta"], alpha=0.12, zorder=1)
    ax.plot(line, line, "--", color=COLORS2HEX["terracotta"], linewidth=2.0, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if box_aspect is None:
        ax.set_aspect("equal")
    else:
        ax.set_box_aspect(box_aspect) 
    _log_ticks(ax, lo, hi)
    ax.tick_params(axis="both", which="major", labelsize=POOLED_FS_TICK)

    mape = float(np.mean(np.abs(pred - emp) / emp) * 100)
    r2 = float(np.corrcoef(np.log10(pred), np.log10(emp))[0, 1] ** 2)
    ax.annotate(f"MAPE = {mape:.1f}%\n$R^2$ = {r2:.3f}",
                xy=(0.04, 0.96), xycoords="axes fraction", va="top", ha="left",
                fontsize=POOLED_FS_ANNOT,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85, ec="0.7"))
    return mape, r2, lo, hi

def plot_pooled_scatter(results, eval_units, out_path, band=0.25, groupby="both"):
    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    mape, r2, _, _ = _draw_pooled(ax, results, eval_units, band, groupby)
    ax.set_xlabel("Predicted sync delay (µs)", fontsize=POOLED_FS_LABEL)
    ax.set_ylabel("Empirical sync delay (µs)", fontsize=POOLED_FS_LABEL)
    handles = _pooled_legend_handles(eval_units, groupby)
    ax.legend(handles=handles, loc="lower right", fontsize=POOLED_FS_LEGEND, frameon=False,
              ncol=2 if groupby == "both" else 1,
              handletextpad=0.4, columnspacing=1.0)
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved pooled scatter to {out_path} (MAPE={mape:.1f}%, R^2={r2:.3f})")
    return mape, r2

def plot_pooled_pair(panels, eval_units, out_path, band=0.25, groupby="both"):
    lo, hi = float("inf"), 0.0
    for _, results in panels:
        for unit in eval_units:
            r = results[unit]
            if len(r["bar_pred"]) == 0:
                continue
            vals = np.concatenate([r["bar_pred"], r["bar_emp_mean"]]) * 1000.0
            lo = min(lo, vals.min() * 0.7)
            hi = max(hi, vals.max() * 1.4)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 4.0), squeeze=False)
    out = []
    for idx, (title, results) in enumerate(panels):
        ax = axes[0][idx]
        mape, r2, _, _ = _draw_pooled(ax, results, eval_units, band, groupby,
                                      lim=(lo, hi), box_aspect=0.60)
        ax.set_title(f"({chr(ord('a') + idx)}) {title}", fontsize=POOLED_FS_TITLE)
        ax.set_xlabel("Predicted sync delay (µs)", fontsize=POOLED_FS_LABEL)
        if idx == 0:
            ax.set_ylabel("Empirical sync delay (µs)", fontsize=POOLED_FS_LABEL)
        else:
            ax.tick_params(labelleft=False)
        out.append((mape, r2))

    handles = _pooled_legend_handles(eval_units, groupby)
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.09), fontsize=18, frameon=False,
               handletextpad=0.4, columnspacing=1.6)
    fig.subplots_adjust(wspace=0.06, top=0.86)
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved paired pooled scatter to {out_path}: " +
          ", ".join(f"({chr(ord('a')+i)}) MAPE={m:.1f}% R^2={r:.3f}"
                    for i, (m, r) in enumerate(out)))
    return out

def plot_scatter(results, eval_units, out_path):
    nrows, ncols = grid_shape(len(eval_units))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows), squeeze=False)
    handles = None
    for idx, unit in enumerate(eval_units):
        ax = axes[idx // ncols][idx % ncols]
        title = (f"({chr(ord('a') + idx)}) {WORKLOADS[unit[0]]['title']}, "
                 f"{unit[1].upper()}")
        r = results[unit]
        xs, means, sems = r["xs"] * 1000.0, r["means"] * 1000.0, r["sems"] * 1000.0
        if len(xs) == 0:
            ax.set_title(f"{title} (no data)")
            continue
        ax.errorbar(xs, means, yerr=sems, fmt="o", capsize=4, label="Mean empirical ± SEM")
        lim = max(xs.max(), (means + sems).max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="y = x")
        ax.set_xlabel("Predicted mean delay (µs)", fontsize=16)
        ax.set_ylabel("Mean empirical delay (µs)", fontsize=16)
        ax.set_title(title, fontsize=16)
        if handles is None:
            handles = ax.get_legend_handles_labels()
    for extra in range(len(eval_units), nrows * ncols):
        axes[extra // ncols][extra % ncols].axis("off")
    if handles is not None:
        fig.legend(*handles, loc="lower center", ncol=len(handles[0]),
                   bbox_to_anchor=(0.5, -0.10 / nrows), fontsize=16, frameon=False)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved scatter plot to {out_path}")

def plot_zfit(z, params, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4), gridspec_kw={"wspace": 0.3})
    if params["shape"] == 0:
        pdf = lambda x: stats.gumbel_r.pdf(x, params["loc"], params["scale"])
        ppf = lambda p: stats.gumbel_r.ppf(p, params["loc"], params["scale"])
    else:
        pdf = lambda x: stats.genextreme.pdf(x, params["shape"], loc=params["loc"],
                                             scale=params["scale"])
        ppf = lambda p: stats.genextreme.ppf(p, params["shape"], loc=params["loc"],
                                             scale=params["scale"])
    axes[0].hist(z, bins=min(80, max(10, len(z) // 5)), density=True,
                 color=COLORS2HEX["petrol"], alpha=0.7, label="Empirical")
    grid = np.linspace(z.min(), z.max(), 200)
    axes[0].plot(grid, pdf(grid), color=COLORS2HEX["terracotta"], linewidth=3,
                 label="Fitted GEVD")
    axes[0].set_xlabel("Z-score of max runtime across ranks")
    axes[0].set_ylabel("Density")
    axes[0].set_title("(a) PDF (fit set only)")
    axes[0].legend(fontsize=14, frameon=False, loc="best")

    probs = (np.arange(1, len(z) + 1) - 0.5) / len(z)
    theoretical, empirical = ppf(probs), np.sort(z)
    lo = min(theoretical.min(), empirical.min())
    hi = max(theoretical.max(), empirical.max())
    axes[1].plot([lo, hi], [lo, hi], color=COLORS2HEX["petrol"], linestyle="--",
                 linewidth=1.5, label="y = x")
    axes[1].scatter(theoretical, empirical, color=COLORS2HEX["teal"], s=12, alpha=0.7)
    axes[1].set_xlim(lo, hi)
    axes[1].set_ylim(lo, hi)
    axes[1].set_xlabel("Theoretical quantile")
    axes[1].set_ylabel("Empirical quantile")
    axes[1].set_title("(b) QQ on z-scores")
    axes[1].annotate(
        f"ξ = {-1 * params['shape']:.3f}\nμ = {params['loc']:.3f}\nσ = {params['scale']:.3f}",
        xy=(0.05, 0.95), xycoords="axes fraction", va="top", ha="left", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    axes[1].legend(fontsize=14, frameon=False, loc="lower right")
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved fit diagnostic to {out_path}")

def fit_and_predict(raw, fit_units, eval_units, args):
    per_kernel = args.level == "kernel"

    fit_summary = summarize(raw, fit_units, per_kernel)
    z, clusters = z_scores_from(fit_summary)
    if len(z) < 10:
        raise SystemExit(f"only {len(z)} z-scores in the fit set -- nothing to fit")
    params = fit_gevd(z, args.family)
    m = fit_group_size(fit_units)

    if args.stats_from == "fit":
        stats_units = fit_units
    elif args.stats_from == "all":
        stats_units = list(dict.fromkeys(fit_units + eval_units))
    else:
        stats_units = eval_units
    stats_table_full = summarize(raw, stats_units, per_kernel)
    stats_table = {k: {"mean": v["mean"], "std": v["pooled_std"]}
                   for k, v in stats_table_full.items() if v["pooled_std"] > 0}

    results = {u: compute_workload(u[0], u[1], stats_table, params, m, per_kernel)
               for u in eval_units}
    return params, m, z, len(clusters), stats_table, results

def holdout_fit_units(unit, rule):
    if rule == "none":
        return list(ALL_UNITS)
    if rule == "block":
        return [u for u in ALL_UNITS if u[0] != unit[0] and u[1] != unit[1]]
    if rule == "workload":
        return [u for u in ALL_UNITS if u[0] != unit[0]]
    return [u for u in ALL_UNITS if u != unit]

def run_split(raw, fit_units, eval_units, args, tag):
    params, m, z, n_clusters, stats_table, results = fit_and_predict(
        raw, fit_units, eval_units, args)
    clusters = range(n_clusters)

    print(f"\n[{tag}] fit on {[unit_name(u) for u in fit_units]}")
    print(f"[{tag}] eval on {[unit_name(u) for u in eval_units]}")
    print(f"[{tag}] {len(clusters)} clusters, {len(z)} z-scores, m = {m}")
    print(f"[{tag}] {args.family} fit: shape={params['shape']:.4f} "
          f"loc={params['loc']:.4f} scale={params['scale']:.4f}")

    for unit in eval_units:
        r = results[unit]
        print(f"[{tag}] {unit_name(unit)}: n={r['n']} k={r['k']:g} E[z]={r['z']:.3f} "
              f"MAPE={r['mape']:.1f}% bias={r['bias']:+.1f}%")

    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"{args.family}_{args.level}"
    plot_bars(results, eval_units, f"{out_dir}/loo_validation_bars_{suffix}.pdf")
    plot_scatter(results, eval_units, f"{out_dir}/loo_validation_scatter_{suffix}.pdf")
    plot_zfit(z, params, f"{out_dir}/loo_zfit_{suffix}.pdf")

    with open(f"{out_dir}/loo_summary_{suffix}.txt", "w") as f:
        f.write(f"fit units:  {[unit_name(u) for u in fit_units]}\n")
        f.write(f"eval units: {[unit_name(u) for u in eval_units]}\n")
        f.write(f"family={args.family} level={args.level} stats_from={args.stats_from} m={m}\n")
        f.write(f"z-scores={len(z)} clusters={len(clusters)}\n")
        f.write(f"params: shape={params['shape']:.6f} loc={params['loc']:.6f} "
                f"scale={params['scale']:.6f}\n\n")
        for unit in eval_units:
            r = results[unit]
            f.write(f"=== {unit_name(unit)} ===\n")
            f.write(f"n={r['n']} k={r['k']:g} E[z]={r['z']:.4f} "
                    f"MAPE={r['mape']:.2f}% bias={r['bias']:+.2f}%\n")
            for i, combo in enumerate(r["bar_combos"]):
                f.write(f"{i}\t{combo}\n\n")
            f.write("\n")
    print(f"Saved summary + combo legend to {out_dir}/loo_summary_{suffix}.txt")

    if args.save_params:
        out_fp = os.path.join(out_dir, f"loo_params_{suffix}.pkl")
        assert not os.path.exists(out_fp) or args.overwrite_params, \
            f"{out_fp} exists; pass --overwrite-params to replace it"
        with open(out_fp, "wb") as f:
            pickle.dump({"params": params, "m": m,
                         "fit_units": [unit_name(u) for u in fit_units],
                         "eval_units": [unit_name(u) for u in eval_units],
                         "family": args.family, "level": args.level,
                         "data": stats_table}, f)
        print(f"Stored held-out fit params to {out_fp}")

    return params, results


def run_mosaic(raw, panel_units, args):
    per_panel = {}
    for unit in panel_units:
        fit_units = holdout_fit_units(unit, args.mosaic_holdout)
        params, m, z, n_clusters, _, results = fit_and_predict(raw, fit_units, [unit], args)
        r = results[unit]
        per_panel[unit] = {"params": params, "m": m, "n_z": len(z),
                           "n_clusters": n_clusters, "fit_units": fit_units, "r": r}
        print(f"[mosaic/{args.mosaic_holdout}] {unit_name(unit)}: "
              f"fit on {len(fit_units)} units ({short_set(fit_units)}), "
              f"{len(z)} z-scores, loc={params['loc']:.4f} scale={params['scale']:.4f} "
              f"| k={r['k']:g} E[z]={r['z']:.3f} MAPE={r['mape']:.1f}% bias={r['bias']:+.1f}%")

    results = {u: d["r"] for u, d in per_panel.items()}

    out_dir = os.path.join(args.out_dir, f"mosaic-{args.mosaic_holdout}")
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"{args.family}_{args.level}"
    pooled = plot_pooled_scatter(results, panel_units,
                                 f"{out_dir}/pooled_scatter_{suffix}.pdf",
                                 groupby=args.pooled_groupby)
    if not args.pooled_only:
        plot_bars(results, panel_units, f"{out_dir}/mosaic_bars_{suffix}.pdf")
        plot_scatter(results, panel_units, f"{out_dir}/mosaic_scatter_{suffix}.pdf")

    with open(f"{out_dir}/mosaic_summary_{suffix}.txt", "w") as f:
        f.write(f"Per-panel held-out fits (rule = {args.mosaic_holdout}).\n")
        f.write(f"family={args.family} level={args.level} stats_from={args.stats_from}\n")
        f.write(f"pooled over all panels: MAPE={pooled[0]:.2f}% "
                f"R^2(log10)={pooled[1]:.4f}\n\n")
        for idx, unit in enumerate(panel_units):
            d = per_panel[unit]
            r = d["r"]
            f.write(f"=== ({chr(ord('a') + idx)}) {unit_name(unit)} ===\n")
            f.write(f"fit on: {[unit_name(u) for u in d['fit_units']]}\n")
            f.write(f"z-scores={d['n_z']} clusters={d['n_clusters']} m={d['m']}\n")
            f.write(f"params: shape={d['params']['shape']:.6f} "
                    f"loc={d['params']['loc']:.6f} scale={d['params']['scale']:.6f}\n")
            f.write(f"n={r['n']} k={r['k']:g} E[z]={r['z']:.4f} "
                    f"MAPE={r['mape']:.2f}% bias={r['bias']:+.2f}%\n")
            for i, combo in enumerate(r["bar_combos"]):
                f.write(f"{i}\t{combo}\n\n")
            f.write("\n")
    print(f"Saved per-panel fit provenance to {out_dir}/mosaic_summary_{suffix}.txt")

    if args.save_errors:
        per_unit = {unit_name(u): per_panel[u]["r"]["rel"] for u in panel_units}
        pooled_rel = np.concatenate([v for v in per_unit.values() if len(v)])
        payload = {
            "rel": pooled_rel,
            "per_unit": per_unit,
            "k_by_unit": {unit_name(u): per_panel[u]["r"]["k"] for u in panel_units},
            "meta": {
                "holdout": args.mosaic_holdout,
                "family": args.family,
                "level": args.level,
                "stats_from": args.stats_from,
                "pooled_mape": pooled[0],
                "pooled_r2_log10": pooled[1],
            },
        }
        with open(args.errors_out, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved {len(pooled_rel)} held-out relative errors to {args.errors_out}")

    return [(f"mosaic/{args.mosaic_holdout}", unit_name(u),
             results[u]["mape"], results[u]["bias"]) for u in panel_units]

def main():
    p = argparse.ArgumentParser(
        description="Leave-one-out GEVD fit + sync-delay validation (extrapolation test).")
    p.add_argument("--fit", nargs="+", default=["all"])
    p.add_argument("--fit-exclude", nargs="+", default=[])
    p.add_argument("--eval", nargs="+", default=None)
    p.add_argument("--loo", action="store_true")
    p.add_argument("--loo-workload", action="store_true")
    p.add_argument("--loo-arch", action="store_true")
    p.add_argument("--pooled-compare", action="store_true")
    p.add_argument("--pooled-scatter", action="store_true")
    p.add_argument("--pooled-groupby", choices=["both", "workload", "arch"],
                   default="both")
    p.add_argument("--pooled-only", action="store_true")
    p.add_argument("--mosaic", action="store_true")
    p.add_argument("--mosaic-holdout", choices=["block", "unit", "workload", "none"],
                   default="block")
    p.add_argument("--loo-block", action="store_true")
    p.add_argument("--family", choices=["gumbel", "general"], default="gumbel")
    p.add_argument("--pooled-families", nargs="+", choices=["gumbel", "general"])
    p.add_argument("--level", choices=["combo", "kernel"], default="combo")
    p.add_argument("--stats-from", choices=["eval", "fit", "all"], default="eval")
    p.add_argument("--out-dir", default="../<OUTPUT_PATH>/gevd-loo")
    p.add_argument("--save-errors", action="store_true")
    p.add_argument("--errors-out", default="../<DATA_PATH>/loo_rel_errors.pkl")
    p.add_argument("--save-params", action="store_true")
    p.add_argument("--overwrite-params", action="store_true")
    args = p.parse_args()

    plt.rcParams.update({
        "font.size": 16,
        "font.family": "Fira Code",
        "axes.prop_cycle": cycler(color=list(COLORS2HEX.values())),
    })

    sweeps = [args.loo, args.loo_workload, args.loo_arch, args.loo_block,
              args.mosaic or args.pooled_scatter, args.pooled_compare]
    if sum(bool(x) for x in sweeps) > 1:
        raise SystemExit("pick at most one of --loo / --loo-workload / --loo-arch / "
                         "--loo-block / --mosaic / --pooled-scatter")

    if args.pooled_compare:
        panel_units = parse_unit_set(args.eval) if args.eval else PAPER_PANELS
        print("Loading path traces...")
        raw = collect_units(ALL_UNITS)
        families = args.pooled_families or [args.family]
        panels = []
        for family in families:
            args.family = family
            for rule, sub in [("none", "fit on all"),
                              (args.mosaic_holdout, "held out")]:
                results = {}
                for unit in panel_units:
                    fit_units = holdout_fit_units(unit, rule)
                    params, _, _, _, _, res = fit_and_predict(raw, fit_units, [unit], args)
                    results[unit] = res[unit]
                    print(f"[{family}/{rule}] {unit_name(unit)}: fit on {len(fit_units)} "
                          f"units, shape={params['shape']:.4f} loc={params['loc']:.4f} "
                          f"scale={params['scale']:.4f} MAPE={res[unit]['mape']:.1f}%")
                title = f"{FAMILY_LABEL[family]}, {sub}" if len(families) > 1 else (
                    "Fit on all workloads" if rule == "none" else
                    "Held out: unseen model + GPU")
                panels.append((title, results))
        out_dir = os.path.join(args.out_dir, f"compare-none-vs-{args.mosaic_holdout}")
        os.makedirs(out_dir, exist_ok=True)
        plot_pooled_pair(panels, panel_units,
                         f"{out_dir}/pooled_compare_{'-'.join(families)}_{args.level}.pdf",
                         groupby=args.pooled_groupby)
        return

    if args.pooled_scatter:
        args.mosaic = True
        args.pooled_only = True

    if args.mosaic:
        panel_units = parse_unit_set(args.eval) if args.eval else PAPER_PANELS
        print("Loading path traces...")
        raw = collect_units(ALL_UNITS)
        table = run_mosaic(raw, panel_units, args)
        print("\n=== held-out accuracy (per-panel fits) ===")
        print(f"{'split':<26} {'unit':<28} {'MAPE':>8} {'bias':>8}")
        for tag, name, mape, bias in table:
            print(f"{tag[:26]:<26} {name:<28} {mape:>7.1f}% {bias:>+7.1f}%")
        return

    if any(sweeps):
        splits = []
        if args.loo_workload:
            for wl in WORKLOADS:
                held = [(wl, a) for a in ARCHS]
                splits.append(([u for u in ALL_UNITS if u not in held], held,
                               f"holdout-{short_set(held)}"))
        elif args.loo_arch:
            for arch in ARCHS:
                held = [(w, arch) for w in WORKLOADS]
                splits.append(([u for u in ALL_UNITS if u not in held], held,
                               f"holdout-{arch}"))
        elif args.loo_block:
            for unit in ALL_UNITS:
                # drop the unit's whole row (same workload) and column (same arch)
                fit = [u for u in ALL_UNITS if u[0] != unit[0] and u[1] != unit[1]]
                splits.append((fit, [unit], f"block-{short_name(unit)}"))
        else:
            for unit in ALL_UNITS:
                splits.append(([u for u in ALL_UNITS if u != unit], [unit],
                               f"holdout-{short_name(unit)}"))
        needed = ALL_UNITS
    else:
        eval_units = parse_unit_set(args.eval) if args.eval else parse_unit_set(args.fit_exclude)
        if not eval_units:
            raise SystemExit("nothing to validate on: pass --eval (or --fit-exclude)")
        fit_units = parse_unit_set(args.fit)
        if args.fit_exclude:
            drop = set(parse_unit_set(args.fit_exclude))
            fit_units = [u for u in fit_units if u not in drop]
            if not fit_units:
                raise SystemExit("fit set is empty after --fit-exclude")
        else:
            held_out = [u for u in fit_units if u not in set(eval_units)]
            if held_out:
                fit_units = held_out
        overlap = set(fit_units) & set(eval_units)
        if overlap:
            print(f"WARNING: fit and eval overlap on {[unit_name(u) for u in overlap]} "
                  f"-- this is no longer a held-out test")
        tag = "fit-" + short_set(fit_units) + "__eval-" + short_set(eval_units)
        splits = [(fit_units, eval_units, tag)]
        needed = list(dict.fromkeys(fit_units + eval_units))

    print("Loading path traces...")
    raw = collect_units(needed)

    table = []
    for fit_units, eval_units, tag in splits:
        _, results = run_split(raw, fit_units, eval_units, args, tag)
        for unit in eval_units:
            table.append((tag, unit_name(unit), results[unit]["mape"], results[unit]["bias"]))

    print("\n=== held-out accuracy ===")
    print(f"{'split':<26} {'unit':<28} {'MAPE':>8} {'bias':>8}")
    for tag, name, mape, bias in table:
        print(f"{tag[:26]:<26} {name:<28} {mape:>7.1f}% {bias:>+7.1f}%")


if __name__ == "__main__":
    main()
