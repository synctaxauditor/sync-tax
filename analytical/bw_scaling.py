import os
import math
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

COLORS2HEX = {
    "petrol":     "#264653",
    "gold":       "#E9C46A",
    "teal":       "#2A9D8F",
    "sand":       "#F4A261",
    "terracotta": "#E76F51",
}

WORKLOAD = "<WORKLOAD_NAME>" # e.g., "n8-fsdp-tp-llama70b"
ARCHS = ["<ARCH_NAME>"]  # e.g., ["h200"]
COMBO_DATA_FP = "<COMBO_DATA_FP>" # e.g., "../validation-data/agg_combo_data_gumbel.pkl"
COMBO_DATA_FP_GENERAL = "<COMBO_DATA_FP_GENERAL>" # e.g., "../validation-data/agg_combo_data_general.pkl"
PATH_TRACE_TEMPLATE = "<PATH_TRACE_TEMPLATE>" # e.g., "../validation-data/path-traces/{workload}-{arch}_v4.pkl"
TRACE_SAMPLE_TEMPLATE = "<TRACE_SAMPLE_TEMPLATE>"
IGNORE = {"CPU", "MemEffAttention", "sdpa"} # Only analyze GEMMs because dimensions are fixed

BYTES_PER_DTYPE = {
    "Float": 4, "Double": 8, "Half": 2, "BFloat16": 2,
    "Long": 8, "Int": 4, "Byte": 1, "Char": 1, "Bool": 1,
}

BASELINES = ["ring", "bucket", "sharp"]
BASELINE_TITLES = {"ring": "Ring", "bucket": "3D Torus", "sharp": "Ideal Fully Connected"}
ALPHA_US = 5.0
M = 4    # group size for n8-fsdp-tp-llama70b
ELASTICITY = [-0.1, -0.5]
N_ALL = sorted({8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096})
FLOPS_SCALING = [1, 5, 10, 15, 20]
N_FLOPS_LIST = [256, 4096]
ALG_FLOPS = "sharp"

def calc_expected_value_gumbel(mu, sigma, k=1):
    gamma = 0.5772156649  # Euler-Mascheroni constant
    return mu + (sigma * (gamma + np.log(k)))

def calc_expected_value_frechet(mu, sigma, kappa, k=1):
    assert kappa != 0
    kappa = -1 * kappa
    return mu + (sigma * (((k ** kappa) * math.gamma(1 - kappa)) - 1) / kappa)


def calc_p_q_for_alg(alg, n):
    if alg == "ring":
        return 2 * (n - 1), 2 * (n - 1) / n
    elif alg == "bucket":
        d = 3
        return 2 * d * ((n ** (1 / d)) - 1), 2 * (n - 1) / n
    elif alg == "sharp":
        return 2, 2 * (n - 1) / n
    else:
        raise ValueError(f"Unknown algorithm: {alg}")

def calc_optimal_bw_ours(p, q, a, s, mu, sigma, k, dur, cov, elasticity, kappa=0, e2e=False):
    assert -1 <= elasticity <= 0
    assert cov >= 0
    assert dur >= 0
    assert k >= 1
    if kappa == 0:
        z = calc_expected_value_gumbel(mu, sigma, k)
    else:
        z = calc_expected_value_frechet(mu, sigma, kappa, k)
    sync_delay = cov * dur * z
    if not e2e:
        return -(1 + elasticity) * q * s / (elasticity * (p * a + sync_delay))
    else:
        return -(1 + elasticity) * q * s / (elasticity * (p * a + sync_delay + dur))

def calc_elasticity(p, q, a, s, bw, dur):
    # Back out the elasticity that would yield the given bandwidth under our model
    return - (q * S) / (bw * (p * a + dur) + q * S)


def calc_optimal_bw_baseline(p, q, a, s, elasticity, e2e=False, dur=None):
    assert -1 <= elasticity <= 0
    if not e2e:
        return -(1 + elasticity) * q * s / (elasticity * p * a)
    else:
        assert dur is not None
        return -(1 + elasticity) * q * s / (elasticity * (p * a + dur))

def is_perfect_cube(n):
    r = round(n ** (1 / 3))
    return r ** 3 == n

def get_max_reducescatter_gb(trace_path):
    with open(trace_path, "r") as f:
        data = json.load(f)
    max_bytes = 0
    for e in data.get("traceEvents", []):
        if e.get("cat") != "cpu_op" or e.get("name") != "record_param_comms":
            continue
        args = e.get("args", {})
        if "reduce_scatter" not in args.get("Collective name", "").lower():
            continue
        pg = args.get("Process Group Description", "")
        if "mesh_tp" not in pg.lower() and "default_pg" not in pg.lower():
            continue
        bpe = BYTES_PER_DTYPE.get(args.get("dtype"))
        if bpe is None:
            continue
        nelems = max(args.get("In msg nelems", 0), args.get("Out msg nelems", 0))
        max_bytes = max(max_bytes, nelems * bpe)
    return max_bytes / 1e9

def gather_llama70b_combo_stats():
    with open(COMBO_DATA_FP, "rb") as f:
        payload = pickle.load(f)
    kernel_data = payload["data"]
    params = dict(payload["params"])

    seen = set()
    for arch in ARCHS:
        fp = PATH_TRACE_TEMPLATE.format(workload=WORKLOAD, arch=arch)
        with open(fp, "rb") as f:
            path_traces = pickle.load(f)
        for iter_data in path_traces.values():
            ranks = list(iter_data["comp"].keys())
            num_events = len(iter_data["comp"][ranks[0]])
            for ev_idx in range(num_events):
                rec = iter_data["comp"][ranks[0]][ev_idx]
                if len(rec) == 0:
                    continue
                combo = tuple(name for name, _ in rec)
                if any(sub in name for name in combo for sub in IGNORE):
                    continue
                key = (combo, arch)
                if key in kernel_data:
                    seen.add(key)

    durs = np.array([kernel_data[k]["mean"] for k in seen], dtype=float)
    stds = np.array([kernel_data[k]["std"] for k in seen], dtype=float)
    covs = stds / durs
    return durs, covs, params


if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 16,
        "font.family": "Fira Code",
        "axes.prop_cycle": cycler(color=list(COLORS2HEX.values())),
    })

    durs, covs, gumbel_params = gather_llama70b_combo_stats()
    MU = float(gumbel_params["loc"])
    SIGMA = float(gumbel_params["scale"])
    KAPPA = float(gumbel_params["shape"])

    with open(COMBO_DATA_FP_GENERAL, "rb") as f:
        general_params = pickle.load(f)["params"]
    MU_G = float(general_params["loc"])
    SIGMA_G = float(general_params["scale"])
    KAPPA_G = float(general_params["shape"])

    S = get_max_reducescatter_gb(TRACE_SAMPLE_TEMPLATE.format(
        workload=WORKLOAD, arch=ARCHS[0]))
    print(f"Empirical max ReduceScatter buffer: S = {S:.4f} GB")

    MEAN_DUR = float(np.mean(durs))
    MEDIAN_DUR = float(np.median(durs))
    COV_MEAN = float(np.mean(covs))
    COV_MEDIAN = float(np.median(covs))

    # Use the mean of compute block durations / COVs for the simulation
    DUR = MEAN_DUR
    COV = COV_MEAN
    ALPHA = ALPHA_US / 1e3  # ms

    fig, axes = plt.subplots(1, len(BASELINES), figsize=(6 * len(BASELINES), 4.2), squeeze=False)
    handles = None
    for j, alg in enumerate(BASELINES):
        ax = axes[0][j]
        letter = chr(ord("a") + j)

        for e in ELASTICITY:
            n_vals_base, bw_base = [], []
            n_vals_ours, bw_ours = [], []
            for n in N_ALL:
                if alg == "bucket" and not is_perfect_cube(n):
                    continue
                p, q = calc_p_q_for_alg(alg, n)
                bw_b = calc_optimal_bw_baseline(p, q, ALPHA, S, e)
                bw_o = calc_optimal_bw_ours(p, q, ALPHA, S, MU, SIGMA,
                                            n / M, DUR, COV, e, KAPPA)
                # bw currently in GB/ms; convert to GB/s
                n_vals_base.append(n); bw_base.append(bw_b * 1e3)
                n_vals_ours.append(n); bw_ours.append(bw_o * 1e3)

            linestyle = "-" if e == -0.5 else "--"
            ax.plot(n_vals_base, bw_base,
                    color=COLORS2HEX["petrol"], linestyle=linestyle,
                    marker="o", markersize=5, linewidth=3.5,
                    label=fr"Baseline, $\epsilon$={e}")
            ax.plot(n_vals_ours, bw_ours,
                    color=COLORS2HEX["terracotta"], linestyle=linestyle,
                    marker="s", markersize=5, linewidth=4,
                    label=fr"w/ sync tax, $\epsilon$={e}")
            
            print(f"Baseline bandwidth requirements for {alg} (epsilon={e}):")
            print("n", "baseline BW (GB/s)", "sync tax BW (GB/s)", "% diff", sep="\t")
            for n, b, o in zip(n_vals_base, bw_base, bw_ours):
                print(n, f"{b:.1f}", f"{o:.1f}", f"{(o - b) / b * 100:.3f}%", sep="\t")
        ax.axhline(450, color=COLORS2HEX["gold"], linestyle="--",
                   linewidth=2, label="NVLink (450 GB/s)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (# GPUs)", fontsize=18)
        ax.set_ylabel("Optimal bandwidth (GB/s)", fontsize=16)
        ax.set_title(f"({letter}) {BASELINE_TITLES[alg]}", fontsize=19)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

        if handles is None:
            handles = ax.get_legend_handles_labels()

    if handles is not None:
        fig.legend(*handles, loc="lower center", ncol=len(handles[0]),
                   bbox_to_anchor=(0.5, -0.02), fontsize=16, frameon=False, handlelength=2.5)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()