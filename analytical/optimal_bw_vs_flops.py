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

WORKLOAD = "n8-fsdp-tp-llama70b"
ARCHS = ["h200"]
COMBO_DATA_FP = "<COMBO_DATA_FP>" # e.g., "../validation-data/agg_combo_data_gumbel.pkl"
COMBO_DATA_FP_GENERAL = "<COMBO_DATA_FP_GENERAL>" # e.g., "../validation-data/agg_combo_data_general.pkl"
PATH_TRACE_TEMPLATE = "<PATH_TRACE_TEMPLATE>" # e.g., "../validation-data/path-traces/{workload}-{arch}_v4.pkl"
TRACE_SAMPLE_TEMPLATE = "<TRACE_SAMPLE_TEMPLATE>"
IGNORE = {"CPU", "MemEffAttention", "sdpa"}

BYTES_PER_DTYPE = {
    "Float": 4, "Double": 8, "Half": 2, "BFloat16": 2,
    "Long": 8, "Int": 4, "Byte": 1, "Char": 1, "Bool": 1,
}

ALPHA_US = 5.0
M = 4
PFLOPS_POINTS = [1, 5, 10, 15, 20]
N_PANELS = [256, 4096]
ALG_FLOPS = "sharp"


def calc_expected_value_gumbel(mu, sigma, k=1):
    gamma = 0.5772156649
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
    return - (q * s) / (bw * (p * a + dur) + q * s)


def calc_elasticity_sync(p, q, a, s, bw, dur, sync_delay):
    return - (q * s) / (bw * (p * a + dur + sync_delay) + q * s)


def calc_optimal_bw_baseline(p, q, a, s, elasticity, e2e=False, dur=None):
    assert -1 <= elasticity <= 0
    if not e2e:
        return -(1 + elasticity) * q * s / (elasticity * p * a)
    else:
        assert dur is not None
        return -(1 + elasticity) * q * s / (elasticity * (p * a + dur))


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
    COV_MEAN = float(np.mean(covs))

    DUR = MEAN_DUR
    COV = COV_MEAN
    ALPHA = ALPHA_US / 1e3  # ms

    BASELINE_BW = 450  # GB/s
    BASELINE_TFLOPS = 989
    N_ANCHOR = 8
    p_anchor, q_anchor = calc_p_q_for_alg(ALG_FLOPS, N_ANCHOR)

    k_anchor = N_ANCHOR / M
    z_anchor = calc_expected_value_frechet(MU, SIGMA, KAPPA, k_anchor) if KAPPA != 0 \
               else calc_expected_value_gumbel(MU, SIGMA, k_anchor)
    # sync_delay_anchor = COV * DUR * z_anchor
    BASELINE_ELASTICITY = calc_elasticity(
        p_anchor, q_anchor, ALPHA, S, BASELINE_BW, DUR)
    print(f"Sync-tax-calibrated elasticity at N={N_ANCHOR}: {BASELINE_ELASTICITY:.8f}")
    print(f"  (|e| = {abs(BASELINE_ELASTICITY) * 100:.1f}% of e2e time is comm at anchor)")

    fig, axes = plt.subplots(1, len(N_PANELS), figsize=(6.5 * len(N_PANELS), 5))
    width = 0.27
    panel_letters = ["a", "b", "c", "d"]

    for j, N_PANEL in enumerate(N_PANELS):
        p_fixed, q_fixed = calc_p_q_for_alg(ALG_FLOPS, N_PANEL)
        bw_base_flops = []
        bw_sync_flops = []
        bw_sync_general_flops = []
        for pflops in PFLOPS_POINTS:
            factor = pflops / (BASELINE_TFLOPS / 1e3)  # relative to H200 baseline
            T_comp = DUR / factor
            bw_base_flops.append(calc_optimal_bw_baseline(p_fixed, q_fixed, ALPHA, S, BASELINE_ELASTICITY, e2e=True, dur=T_comp) / 1e3)
            bw_sync_flops.append(calc_optimal_bw_ours(p_fixed, q_fixed, ALPHA, S, MU, SIGMA, N_PANEL / M, T_comp, COV, BASELINE_ELASTICITY, KAPPA, e2e=True) / 1e3)
            bw_sync_general_flops.append(calc_optimal_bw_ours(p_fixed, q_fixed, ALPHA, S, MU_G, SIGMA_G, N_PANEL / M, T_comp, COV, BASELINE_ELASTICITY, KAPPA_G, e2e=True) / 1e3)

        print(f"N={N_PANEL}")
        print(f"  PFLOPS: {PFLOPS_POINTS}")
        print(f"  Baseline:            {bw_base_flops}")
        print(f"  w/ sync tax (Gumb.): {bw_sync_flops}")
        print(f"  w/ sync tax (b.fit): {bw_sync_general_flops}")

        ax = axes[j]
        x = np.arange(len(PFLOPS_POINTS))
        ax.bar(x - width, bw_base_flops, width,
               color=COLORS2HEX["petrol"], label="Baseline")
        ax.bar(x, bw_sync_flops, width,
               color=COLORS2HEX["terracotta"], label="w/ sync tax (Gumbel)")
        ax.bar(x + width, bw_sync_general_flops, width,
               color=COLORS2HEX["gold"], label="w/ sync tax (best fit)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in PFLOPS_POINTS])
        ax.set_xlabel("PFLOPS", fontsize=19)
        ax.set_ylabel("Optimal bandwidth (TB/s)", fontsize=19)
        ax.set_title(f"({panel_letters[j]}) N={N_PANEL}", fontsize=21)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    handles = axes[0].get_legend_handles_labels()
    fig.legend(*handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0), fontsize=18, frameon=False, handlelength=2.5)
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    out_dir = "<PATH_TO_OUTPUT_DIR>"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/optimal_bw_bars.pdf"
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Saved plot to {out_path}")
