import os
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy import stats
import argparse

COLORS2HEX = {
    "petrol":     "#264653",
    "gold":       "#E9C46A",
    "teal":       "#2A9D8F",
    "sand":       "#F4A261",
    "terracotta": "#E76F51",
}

plt.rcParams.update({
    'font.size': 16,
    'font.family': 'Fira Code',
    'axes.prop_cycle': cycler(color=list(COLORS2HEX.values())),
})


def _fit_and_qq(ax_hist, ax_qq, samples):
    c, loc, scale = stats.genextreme.fit(samples, f0=0)
    ax_hist.hist(samples, bins=min(80, max(10, len(samples) // 5)),
                 density=True, color=COLORS2HEX["petrol"], alpha=0.7,
                 label="Empirical")
    xs = np.linspace(samples.min(), samples.max(), 200)
    ax_hist.plot(xs, stats.genextreme.pdf(xs, c, loc=loc, scale=scale),
                 color=COLORS2HEX["terracotta"], linewidth=3, label="Fitted Gumbel")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(f"(a) PDF")
    ax_hist.legend(fontsize=14, frameon=False, loc="best")

    probs = (np.arange(1, len(samples) + 1) - 0.5) / len(samples)
    theoretical = stats.genextreme.ppf(probs, c, loc=loc, scale=scale)
    empirical = np.sort(samples)
    lo = min(theoretical.min(), empirical.min())
    hi = max(theoretical.max(), empirical.max())
    ax_qq.fill_between([lo, hi], [lo, hi], hi, color=COLORS2HEX["terracotta"],
                       alpha=0.15, label="Higher than predicted")
    ax_qq.plot([lo, hi], [lo, hi], color=COLORS2HEX["petrol"],
               linestyle="--", linewidth=1.5, label="y = x")
    ax_qq.scatter(theoretical, empirical, color=COLORS2HEX["teal"], s=12, alpha=0.7)
    ax_qq.set_xlim(lo, hi)
    ax_qq.set_ylim(lo, hi)
    ax_qq.set_xlabel("Theoretical quantile")
    ax_qq.set_ylabel("Empirical quantile")
    ax_qq.set_title("(b) QQ (Gumbel) on z-scores")
    fit_text = f"ξ = {c:.3f}\nμ = {loc:.3f}\nσ = {scale:.3f}"
    ax_qq.annotate(fit_text, xy=(0.05, 0.95), xycoords="axes fraction",
                   va="top", ha="left", fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax_qq.legend(fontsize=14, frameon=False, loc="lower right")
    return c, loc, scale


WORKLOADS = {
    "n4-tp-llama8b":       [[0, 1, 2, 3]],
    "n8-tp-qwen32b":       [[0, 1, 2, 3], [4, 5, 6, 7]],
    "n8-fsdp-tp-llama70b": [[0, 1, 2, 3], [4, 5, 6, 7]],
}
ARCHS = ["a100", "h100", "h200"]
IGNORE = {"CPU", "MemEffAttention", "sdpa"} # Only analyze GEMMs because dimensions are fixed

def collect_by_combo():
    """Return (by_combo, by_kernel).

    For each (combo/kernel, arch), data from all groups is combined.

    by_combo:  {(combo, arch): {"maxes": [...], "mean": float, "std": float}}
    by_kernel: {(kernel_name, arch): {"maxes": [...], "mean": float, "std": float}}
    """
    # Keyed by (combo/kname, arch, group_key) so groups stay separate
    maxes_by_combo = defaultdict(list)
    samples_by_combo = defaultdict(list)   # list of per-iteration rank-runtime lists
    maxes_by_kernel = defaultdict(list)
    samples_by_kernel = defaultdict(list)
    for workload, groups in WORKLOADS.items():
        for arch in ARCHS:
            fp = f"<PATH_TO_VALIDATION_DATA>/path-traces/{workload}-{arch}.pkl" # ../validation-data/path-traces/{workload}-{arch}_v4.pkl
            with open(fp, "rb") as f:
                path_traces = pickle.load(f)

            for _iter, iter_data in path_traces.items():
                for g_idx, group in enumerate(groups):
                    gkey = (workload, g_idx)
                    comm_lens = {r: len(iter_data['comm'][r]) for r in group}
                    comp_lens = {r: len(iter_data['comp'][r]) for r in group}
                    if len(set(comm_lens.values())) != 1:
                        continue
                    if len(set(comp_lens.values())) != 1:
                        continue
                    num_events = comp_lens[group[0]]

                    for ev_idx in range(num_events):
                        paths = {r: iter_data['comp'][r][ev_idx] for r in group}
                        if any(len(paths[r]) == 0 for r in group):
                            continue
                        op_names = {r: tuple(name for name, _ in paths[r]) for r in group}
                        if len(set(op_names.values())) != 1:
                            continue
                        combo = next(iter(op_names.values()))
                        if any(sub in name for name in combo for sub in IGNORE):
                            continue
                        key = (combo, arch, gkey)
                        runtimes = [sum(dur for _, dur in paths[r]) / 1000.0 for r in group]
                        maxes_by_combo[key].append(max(runtimes))
                        samples_by_combo[key].append(runtimes)

                        path_len = len(paths[group[0]])
                        for k_idx in range(path_len):
                            k_name = paths[group[0]][k_idx][0]
                            k_durs = [paths[r][k_idx][1] / 1000.0 for r in group]
                            kkey = (k_name, arch, gkey)
                            maxes_by_kernel[kkey].append(max(k_durs))
                            samples_by_kernel[kkey].append(k_durs)

    def _combine_and_summarize(maxes_d, samples_d):
        # Regroup by (combo_or_kname, arch), combining all groups
        by_key_maxes = defaultdict(list)
        by_key_samples = defaultdict(list)
        for (k, arch, gkey), samples in samples_d.items():
            by_key_maxes[(k, arch)].extend(maxes_d[(k, arch, gkey)])
            by_key_samples[(k, arch)].extend(samples)
        out = {}
        for key, samples in by_key_samples.items():
            maxes = by_key_maxes[key]
            sample_means = [float(np.mean(s)) for s in samples]
            num = 0.0
            den = 0
            for s in samples:
                if len(s) > 1:
                    num += (len(s) - 1) * float(np.var(s, ddof=1))
                    den += len(s) - 1
            pooled_std = float(np.sqrt(num / den)) if den > 0 else 0.0
            flat = [v for s in samples for v in s]
            out[key] = {
                "maxes": maxes,
                "sample_means": sample_means,
                "pooled_std": pooled_std,
                "mean": float(np.mean(flat)),
                "std": float(np.std(flat)),
                "n": len(flat),
            }
        return out

    return (_combine_and_summarize(maxes_by_combo, samples_by_combo),
            _combine_and_summarize(maxes_by_kernel, samples_by_kernel))

if __name__ == "__main__":
    by_combo, by_kernel = collect_by_combo()

    z_scores = []
    z_by_arch = defaultdict(list)
    for key, info in by_combo.items():
        if info["pooled_std"] == 0:
            continue
        combo, arch = key
        for m, sm in zip(info["maxes"], info["sample_means"]):
            z = (m - sm) / info["pooled_std"]
            z_scores.append(z)
            z_by_arch[arch].append(z)
    z_scores = np.array(z_scores)
    print("n combos =", len(by_combo), "n z-scores =", len(z_scores))

    kernel_data = {}
    for key, info in by_kernel.items():
        if info["pooled_std"] == 0:
            continue
        kernel_data[key] = {"mean": info["mean"], "std": info["pooled_std"]}
    
    combo_data = {}
    for key, info in by_combo.items():
        if info["pooled_std"] == 0:
            continue
        combo_data[key] = {"mean": info["mean"], "std": info["pooled_std"]}

    out_dir = "<PATH_TO_OUTPUT_DIR>"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Aggregate z-score fit (all arches)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4), gridspec_kw={"wspace": 0.2})
    c, loc, scale = _fit_and_qq(axes[0], axes[1], z_scores)
    params = {"shape": c, "loc": loc, "scale": scale}
    axes[0].set_xlabel("Z-score of max runtime across ranks")
    print(f"[aggregate] GEV fit: ξ={c:.4f}, loc={loc:.4f}, scale={scale:.4f}")
    fig.savefig(f"{out_dir}/aggregate.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


    # 2) Per-architecture aggregate z-score fits
    for arch, zs in z_by_arch.items():
        arch_dir = f"{out_dir}/{arch}"
        os.makedirs(arch_dir, exist_ok=True)
        zs = np.array(zs)
        if len(zs) < 5:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"wspace": 0.3})
        c, loc, scale = _fit_and_qq(axes[0], axes[1], zs)
        axes[0].set_xlabel("Z-score of max runtime across ranks")
        print(f"[{arch}] GEV fit: ξ={c:.4f}, loc={loc:.4f}, scale={scale:.4f}")
        fig.tight_layout()
        fig.savefig(f"{arch_dir}/aggregate.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    # 4) Per-kernel (single-op) QQ plots: aggregate, per-arch aggregate, and per-kernel
    k_z_scores = []
    k_z_by_arch = defaultdict(list)
    for (kname, arch), info in by_kernel.items():
        if info["pooled_std"] == 0:
            continue
        for m, sm in zip(info["maxes"], info["sample_means"]):
            z = (m - sm) / info["pooled_std"]
            k_z_scores.append(z)
            k_z_by_arch[arch].append(z)
    k_z_scores = np.array(k_z_scores)
    print("n kernels =", len(by_kernel), "n kernel z-scores =", len(k_z_scores))

    kernel_out = f"{out_dir}/per_kernel"
    os.makedirs(kernel_out, exist_ok=True)

    if len(k_z_scores) >= 5:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"wspace": 0.3})
        c, loc, scale = _fit_and_qq(axes[0], axes[1], k_z_scores)
        params = {"shape": c, "loc": loc, "scale": scale}
        axes[0].set_xlabel("z-score of max(runtime) per kernel")
        print(f"[kernel-aggregate] GEV fit: ξ={c:.4f}, loc={loc:.4f}, scale={scale:.4f}")
        fig.tight_layout()
        fig.savefig(f"{kernel_out}/aggregate.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    for arch, zs in k_z_by_arch.items():
        arch_dir = f"{kernel_out}/{arch}"
        os.makedirs(arch_dir, exist_ok=True)
        zs = np.array(zs)
        if len(zs) < 5:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"wspace": 0.3})
        c, loc, scale = _fit_and_qq(axes[0], axes[1], zs)
        axes[0].set_xlabel("z-score of max(runtime) per kernel")
        print(f"[kernel-{arch}] GEV fit: ξ={c:.4f}, loc={loc:.4f}, scale={scale:.4f}")
        fig.tight_layout()
        fig.savefig(f"{arch_dir}/aggregate.png", dpi=150, bbox_inches="tight")
        plt.close(fig)