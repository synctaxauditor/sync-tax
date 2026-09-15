import os
import math
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from cycler import cycler

import bw_scaling as S
from bw_scaling import (
    COLORS2HEX, BASELINES, BASELINE_TITLES, ALPHA_US, NVLINK_PEAK, M, N_ALL,
    calc_p_q_for_alg, calc_bw_from_sync_delay, calc_optimal_bw_baseline,
    gather_llama70b_combo_stats, get_max_reducescatter_gb, is_perfect_cube,
)

DEP_PARAMS_FP = "<DEP_PARAMS_FP>" #e.g. ../data/dependence_params.pkl
COMBO_DATA_FP = "<COMBO_DATA_FP>" #e.g. ../data/agg_combo_data_gumbel_boot.pkl
OUT_DIR = "<OUT_DIR>"
GAMMA = 0.5772156649

LAMBDA_TABLE = [1.0, 0.75, 0.5, 0.25, 0.0]
TABLE_N = [64, 512, 4096]

FS_TICK = 20
FS_LABEL = 24
FS_LEGEND = 23
FS_PANEL = 22

def load_lambda(granularity, n_ranks):
    if not os.path.exists(DEP_PARAMS_FP):
        raise SystemExit(f"{DEP_PARAMS_FP} not found; run quantify_dependence.py first")
    with open(DEP_PARAMS_FP, "rb") as f:
        payload = pickle.load(f)
    res = payload[granularity]["scaling"][n_ranks]
    lo, hi = res["lambda_ci"]
    print(f"lambda = {res['lambda']:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
          f"({granularity} granularity, R={n_ranks}, {res['n_units']} units, "
          f"null={res.get('null', 'iidperm')})")
    print(f"  pooled {res.get('weight', 'unit')}-weighted", end="")
    alt = res.get("alt_weight")
    if alt:
        print(f"; {alt['weight']}-weighted sensitivity gives "
              f"lambda = {alt['fit']['lambda']:.4f}", end="")
    print()
    return float(res["lambda"]), float(lo), float(hi)

def expected_z(mu, sigma, kappa, k, lam):
    k_eff = k ** lam
    if kappa == 0:
        return mu + sigma * (GAMMA + math.log(k_eff))
    kap = -kappa
    return mu + sigma * (((k_eff ** kap) * math.gamma(1 - kap)) - 1) / kap

def bw_curve(alg, n_vals, mu, sigma, kappa, lam, dur, cov, alpha, s, e):
    out = []
    for n in n_vals:
        p, q = calc_p_q_for_alg(alg, n)
        sync = cov * dur * expected_z(mu, sigma, kappa, n / M, lam)
        out.append(calc_bw_from_sync_delay(p, q, alpha, s, e, sync) * 1e3)
    return np.array(out)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--granularity", choices=["kernel", "combo"], default="kernel")
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--elasticity", type=float, default=-0.5)
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": FS_TICK, "font.family": "Fira Code",
        "axes.prop_cycle": cycler(color=list(COLORS2HEX.values())),
    })

    lam_hat, lam_lo, lam_hi = load_lambda(args.granularity, args.ranks)

    with open(COMBO_DATA_FP, "rb") as f:
        params = dict(pickle.load(f)["params"])
    MU, SIGMA, KAPPA = (float(params["loc"]), float(params["scale"]),
                        float(params["shape"]))
    durs, covs, _, _ = gather_llama70b_combo_stats()
    DUR, COV = float(np.mean(durs)), float(np.mean(covs))
    ALPHA = ALPHA_US / 1e3
    Sbuf, dtype = get_max_reducescatter_gb(
        S.TRACE_SAMPLE_TEMPLATE.format(workload=S.WORKLOAD, arch=S.ARCHS[0]))
    E = args.elasticity
    print(f"mu={MU:.4f} sigma={SIGMA:.4f} kappa={KAPPA:.4f} | "
          f"DUR={DUR:.3f} ms COV={COV:.5f} S={Sbuf:.4f} GB ({dtype}) e={E}")

    fig, axes = plt.subplots(1, len(BASELINES), figsize=(7.2 * len(BASELINES), 5.6),
                             squeeze=False)
    handles = None
    for j, alg in enumerate(BASELINES):
        ax = axes[0][j]
        n_vals = [n for n in N_ALL if not (alg == "bucket" and not is_perfect_cube(n))]

        base = np.array([calc_optimal_bw_baseline(*calc_p_q_for_alg(alg, n),
                                                  ALPHA, Sbuf, E) * 1e3
                         for n in n_vals])
        ax.plot(n_vals, base, color=COLORS2HEX["petrol"], linestyle="-",
                marker="o", markersize=7, linewidth=3.5, label="Baseline (no sync tax)")

        kw = dict(mu=MU, sigma=SIGMA, kappa=KAPPA, dur=DUR, cov=COV,
                  alpha=ALPHA, s=Sbuf, e=E)
        indep = bw_curve(alg, n_vals, lam=1.0, **kw)
        meas = bw_curve(alg, n_vals, lam=lam_hat, **kw)
        ci_lo = bw_curve(alg, n_vals, lam=lam_hi, **kw)
        ci_hi = bw_curve(alg, n_vals, lam=lam_lo, **kw)
        worst = bw_curve(alg, n_vals, lam=0.0, **kw)

        ax.plot(n_vals, indep, color=COLORS2HEX["terracotta"], linestyle="-",
                marker="s", markersize=11, linewidth=7.0,
                label=r"w/ sync tax, $\lambda=1$ (independent)")
        ax.fill_between(n_vals, ci_lo, ci_hi, color=COLORS2HEX["gold"],
                        alpha=0.30, linewidth=0)
        ax.plot(n_vals, meas, color=COLORS2HEX["gold"], linestyle="-",
                marker="^", markersize=6, linewidth=3.0,
                label=r"measured $\lambda=%.2f$" % lam_hat)
        ax.plot(n_vals, worst, color=COLORS2HEX["teal"], linestyle="--",
                linewidth=3, label=r"$\lambda=0$ (total dependence)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel("N (# GPUs)", fontsize=FS_LABEL)
        if j == 0:
            ax.set_ylabel(r"Required $B^{*}$ (GB/s)", fontsize=FS_LABEL)
        ax.set_title(f"({chr(ord('a') + j)}) {BASELINE_TITLES[alg]}", fontsize=FS_PANEL)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        if handles is None:
            handles = ax.get_legend_handles_labels()

        print(f"\n[{alg}] % change in B* vs baseline")
        print("  N", *[f"lam={l:g}" for l in LAMBDA_TABLE], "lam_hat", sep="\t")
        for n in TABLE_N:
            if n not in n_vals:
                continue
            i = n_vals.index(n)
            cells = []
            for l in LAMBDA_TABLE:
                c = bw_curve(alg, [n], lam=l, **kw)[0]
                cells.append(f"{(c - base[i]) / base[i] * 100:.1f}%")
            cells.append(f"{(meas[i] - base[i]) / base[i] * 100:.1f}%")
            print(f"  {n}", *cells, sep="\t")

    fig.tight_layout()
    fig.legend(*handles, loc="upper center", ncol=len(handles[0]),
               bbox_to_anchor=(0.5, 0.145), fontsize=FS_LEGEND, frameon=False,
               handlelength=2.2, columnspacing=1.6, handletextpad=0.5)
    fig.subplots_adjust(bottom=0.245)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/<OUTPUT_FILE>"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
