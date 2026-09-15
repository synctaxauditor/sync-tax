import os
import math
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from cycler import cycler

COLORS2HEX = {
    "petrol":     "#264653",
    "gold":       "#E9C46A",
    "teal":       "#2A9D8F",
    "sand":       "#F4A261",
    "terracotta": "#E76F51",
}

WORKLOAD = "<WORKLOAD_NAME>"  # e.g., "n8-fsdp-tp-llama70b"
ARCHS = ["<ARCH_NAME>"]  # e.g., ["h200"]
COMBO_DATA_FP = "<COMBO_DATA_FP>" # e.g., "..<DATA_PATH>/agg_combo_data_gumbel.pkl"
LOO_ERRORS_FP = "<LOO_ERRORS_FP>"
COMBO_DATA_FP_GENERAL = "<COMBO_DATA_FP_GENERAL>" # e.g., "<DATA_PATH>/agg_combo_data_general.pkl"
PATH_TRACE_TEMPLATE = "<PATH_TRACE_TEMPLATE>" # e.g., "<DATA_PATH>/{workload}-{arch}.pkl"
TRACE_SAMPLE_TEMPLATE = "<TRACE_SAMPLE_TEMPLATE>"
IGNORE = {"CPU", "MemEffAttention", "sdpa"} #Only analyze GEMMs because dimensions are fixed

BYTES_PER_DTYPE = {
    "Float": 4, "Double": 8, "Half": 2, "BFloat16": 2,
    "Long": 8, "Int": 4, "Byte": 1, "Char": 1, "Bool": 1,
}

BASELINES = ["ring", "bucket", "sharp"]
BASELINE_TITLES = {"ring": "Ring", "bucket": "3D Torus", "sharp": "Ideal Fully Connected"}
ALPHA_US = 5.0
NVLINK_PEAK = {"a100": 300.0, "h100": 450.0, "h200": 450.0}
M = 4 #Group size for n8-fsdp-tp-llama70b
ELASTICITY = [float(x) for x in os.environ.get("SIM_ELASTICITY", "-0.5").split(",")]
BAND_ELASTICITY = ELASTICITY[0]
N_ALL = sorted({8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096})
FLOPS_SCALING = [1, 5, 10, 15, 20]
N_FLOPS_LIST = [256, 4096]
ALG_FLOPS = "sharp"

def calc_expected_value_gumbel(mu, sigma, k=1):
    gamma = 0.5772156649  #Euler-Mascheroni constant
    return mu + (sigma * (gamma + np.log(k)))

def calc_expected_value_frechet(mu, sigma, kappa, k=1):
    assert kappa != 0
    kappa = -1 * kappa
    return mu + (sigma * (((k ** kappa) * math.gamma(1 - kappa)) - 1) / kappa)

def calc_p_q_for_alg(alg, n):
    if alg == "ring":
        return 2 * (n - 1), 2 * (n - 1) / n
    elif alg in ("rhd", "rh/d"):
        return 2 * np.log2(n), 2 * (n - 1) / n
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

def calc_bw_from_sync_delay(p, q, a, s, elasticity, sync_delay):
    return -(1 + elasticity) * q * s / (elasticity * (p * a + sync_delay))

def load_loo_rel_errors():
    if not os.path.exists(LOO_ERRORS_FP):
        return None
    with open(LOO_ERRORS_FP, "rb") as f:
        payload = pickle.load(f)
    rel = np.asarray(payload["rel"], dtype=float)
    meta = payload.get("meta", {})
    print(f"Loaded {len(rel)} held-out relative errors "
          f"(holdout={meta.get('holdout')}, pooled MAPE={meta.get('pooled_mape', float('nan')):.2f}%) "
          f"from {LOO_ERRORS_FP}")
    return rel


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

DTYPE_LABEL = {"Float": "fp32", "BFloat16": "bf16", "Half": "fp16", "Double": "fp64"}


def get_max_reducescatter_gb(trace_path):
    with open(trace_path, "r") as f:
        data = json.load(f)
    max_bytes = 0
    max_dtype = None
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
        if nelems * bpe > max_bytes:
            max_bytes = nelems * bpe
            max_dtype = args.get("dtype")
    return max_bytes / 1e9, max_dtype

def gather_llama70b_combo_stats(workload=None, archs=None):
    workload = WORKLOAD if workload is None else workload
    archs = ARCHS if archs is None else archs
    with open(COMBO_DATA_FP, "rb") as f:
        payload = pickle.load(f)
    kernel_data = payload["data"]
    params = dict(payload["params"])
    param_samples = np.asarray(payload["param_samples"], dtype=float)

    seen = set()
    for arch in archs:
        fp = PATH_TRACE_TEMPLATE.format(workload=workload, arch=arch)
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
    return durs, covs, params, param_samples


if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 16,
        "font.family": "Fira Code",
        "axes.prop_cycle": cycler(color=list(COLORS2HEX.values())),
    })

    durs, covs, gumbel_params, param_samples = gather_llama70b_combo_stats()
    rel_errors = load_loo_rel_errors()

    with open(COMBO_DATA_FP_GENERAL, "rb") as f:
        general_payload = pickle.load(f)
    general_params = dict(general_payload["params"])
    general_samples = np.asarray(general_payload["param_samples"], dtype=float)

    FAMILIES = [
        ("w/ sync tax (Gumbel)", "terracotta", gumbel_params, param_samples),
        ("w/ sync tax (best fit)", "gold", general_params, general_samples),
    ]
    for lbl, _, prm, smp in FAMILIES:
        print(f"{lbl}: shape={float(prm['shape']):.4f} loc={float(prm['loc']):.4f} "
              f"scale={float(prm['scale']):.4f} ({len(smp)} bootstrap draws)")

    S, S_DTYPE = get_max_reducescatter_gb(TRACE_SAMPLE_TEMPLATE.format(
        workload=WORKLOAD, arch=ARCHS[0]))
    DTYPE_TAG = DTYPE_LABEL.get(S_DTYPE, str(S_DTYPE).lower())
    print(f"Empirical max ReduceScatter buffer: S = {S:.4f} GB (dtype {S_DTYPE})")

    MEAN_DUR = float(np.mean(durs))
    MEDIAN_DUR = float(np.median(durs))
    COV_MEAN = float(np.mean(covs))
    COV_MEDIAN = float(np.median(covs))

    print(f"{WORKLOAD} compute blocks (n={len(durs)}):")
    print(f"  dur:  mean = {MEAN_DUR:.4f}, median = {MEDIAN_DUR:.4f}")
    print(f"  cov:  mean = {COV_MEAN:.6f}, median = {COV_MEDIAN:.6f}")

    # Use the mean of compute block durations / COVs for the simulation
    DUR = MEAN_DUR
    COV = COV_MEAN
    ALPHA = ALPHA_US / 1e3  # ms
    PEAK_BW = NVLINK_PEAK[ARCHS[0]]

    fig, axes = plt.subplots(1, len(BASELINES), figsize=(6 * len(BASELINES), 4.2), squeeze=False)
    handles = None
    for j, alg in enumerate(BASELINES):
        ax = axes[0][j]
        letter = chr(ord("a") + j)

        for e in ELASTICITY:
            n_vals = [n for n in N_ALL
                      if not (alg == "bucket" and not is_perfect_cube(n))]
            pqs = [calc_p_q_for_alg(alg, n) for n in n_vals]
            bw_base = [calc_optimal_bw_baseline(p, q, ALPHA, S, e) * 1e3
                       for p, q in pqs]

            linestyle = "-" if e == BAND_ELASTICITY else "--"
            ax.plot(n_vals, bw_base,
                    color=COLORS2HEX["petrol"], linestyle=linestyle,
                    marker="o", markersize=5, linewidth=3.5,
                    label="Baseline")

            for fam_i, (fam_label, fam_color, fam_params, fam_samples) in enumerate(FAMILIES):
                mu = float(fam_params["loc"])
                sigma = float(fam_params["scale"])
                kappa = float(fam_params["shape"])

                bw_ours, tot_lo, tot_hi, fit_lo, fit_hi = [], [], [], [], []
                for n, (p, q) in zip(n_vals, pqs):
                    bw_ours.append(calc_optimal_bw_ours(
                        p, q, ALPHA, S, mu, sigma, n / M, DUR, COV, e, kappa) * 1e3)

                    k = n / M
                    z_s = np.array([
                        calc_expected_value_gumbel(loc, scale, k) if shape == 0
                        else calc_expected_value_frechet(loc, scale, shape, k)
                        for shape, loc, scale in fam_samples])
                    sync_fit = COV * DUR * z_s

                    bw_fit = calc_bw_from_sync_delay(p, q, ALPHA, S, e, sync_fit) * 1e3
                    fit_lo.append(np.percentile(bw_fit, 2.5))
                    fit_hi.append(np.percentile(bw_fit, 97.5))

                    if rel_errors is None:
                        sync_tot = sync_fit
                    else:
                        sync_tot = (sync_fit[:, None] * (1.0 + rel_errors[None, :])).ravel()
                    bw_tot = calc_bw_from_sync_delay(p, q, ALPHA, S, e, sync_tot) * 1e3
                    tot_lo.append(np.percentile(bw_tot, 2.5))
                    tot_hi.append(np.percentile(bw_tot, 97.5))

                if e == BAND_ELASTICITY:
                    ax.fill_between(n_vals, tot_lo, tot_hi,
                                    color=COLORS2HEX[fam_color], alpha=0.20,
                                    linewidth=0)
                ax.plot(n_vals, bw_ours,
                        color=COLORS2HEX[fam_color], linestyle=linestyle,
                        marker="s" if fam_i == 0 else "^", markersize=5, linewidth=4,
                        label=fam_label)

                print(f"\n[{alg}] {fam_label}")
                print("n", "baseline", "sync tax", "% diff", "fit-only CI", "fit+extrap CI",
                      sep="\t")
                for n, b, c, fl, fh, tl, th in zip(n_vals, bw_base, bw_ours,
                                                   fit_lo, fit_hi, tot_lo, tot_hi):
                    print(n, f"{b:.1f}", f"{c:.1f}", f"{(c - b) / b * 100:.2f}%",
                          f"-{100*(c-fl)/c:.1f}/+{100*(fh-c)/c:.1f}%",
                          f"-{100*(c-tl)/c:.1f}/+{100*(th-c)/c:.1f}%", sep="\t")
        ax.axhline(PEAK_BW, color=COLORS2HEX["teal"], linestyle="--",
                   linewidth=2, label=f"NVLink ({PEAK_BW:.0f} GB/s)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (# GPUs)", fontsize=18)
        ax.set_ylabel(r"Required $B^{*}$ (GB/s)", fontsize=16)
        ax.set_title(f"({letter}) {BASELINE_TITLES[alg]}", fontsize=19)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

        if handles is None:
            hs, ls = ax.get_legend_handles_labels()
            pos = max(i for i, l in enumerate(ls) if l.startswith("w/ sync tax")) + 1
            hs.insert(pos, Patch(facecolor="0.72", edgecolor="none"))
            ls.insert(pos, "95% CI (fit + extrapolation)")
            handles = (hs, ls)

    if handles is not None:
        fig.legend(*handles, loc="lower center", ncol=len(handles[0]),
                   bbox_to_anchor=(0.5, -0.02), fontsize=16, frameon=False, handlelength=2.5)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_dir = "<OUTPUT_DIR>"
    os.makedirs(out_dir, exist_ok=True)
    eps_tag = "" if BAND_ELASTICITY == -0.5 else f"_eps{-BAND_ELASTICITY:g}".replace(".", "p")
    out_path = f"{out_dir}/<OUTPUT_FILE>_{DTYPE_TAG}_{ARCHS[0]}{eps_tag}.pdf"
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Saved plot to {out_path}")
