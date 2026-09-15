import os
import pickle
import numpy as np
from scipy import stats
from gevd_fit import collect_by_combo

B = 1000
SEED = 0
JOBS = [
    "<DATA_PATHS>"
]

def build_z_clusters():
    by_combo, _ = collect_by_combo()
    clusters = []
    for info in by_combo.values():
        if info["pooled_std"] == 0:
            continue
        zc = [(m - sm) / info["pooled_std"]
              for m, sm in zip(info["maxes"], info["sample_means"])]
        if zc:
            clusters.append(np.asarray(zc, dtype=float))
    return clusters

def cluster_bootstrap(clusters, rng):
    n = len(clusters)
    idx = np.arange(n)
    gen = np.empty((B, 3))
    gum = np.empty((B, 3))
    for b in range(B):
        pick = rng.choice(idx, size=n, replace=True)
        zb = np.concatenate([clusters[i] for i in pick])
        c, loc, scale = stats.genextreme.fit(zb)
        gen[b] = (c, loc, scale)
        gl, gs = stats.gumbel_r.fit(zb)
        gum[b] = (0.0, gl, gs)
    return gen, gum

def main():
    clusters = build_z_clusters()
    total = sum(len(c) for c in clusters)
    print(f"n clusters = {len(clusters)}, total z-scores = {total}")

    rng = np.random.default_rng(SEED)
    gen_samples, gum_samples = cluster_bootstrap(clusters, rng)
    ensembles = {"gumbel": gum_samples, "general": gen_samples}

    for name, ens in ensembles.items():
        med = np.median(ens, axis=0)
        lo, hi = np.percentile(ens, [2.5, 97.5], axis=0)
        print(f"[{name}] bootstrap median (shape, loc, scale) = "
              f"({med[0]:.4f}, {med[1]:.4f}, {med[2]:.4f})")
        print(f"          95% CI shape={lo[0]:.4f}..{hi[0]:.4f} "
              f"loc={lo[1]:.4f}..{hi[1]:.4f} scale={lo[2]:.4f}..{hi[2]:.4f}")

    meta = {"method": "cluster_bootstrap", "unit": "(combo, arch)",
            "B": B, "seed": SEED, "n_clusters": len(clusters)}
    for src_fp, dst_fp, family in JOBS:
        with open(src_fp, "rb") as f:
            payload = pickle.load(f)
        p = payload["params"]
        med = np.median(ensembles[family], axis=0)
        print(f"[{family}] stored params vs bootstrap median: "
              f"shape {float(p['shape']):.4f}/{med[0]:.4f} "
              f"loc {float(p['loc']):.4f}/{med[1]:.4f} "
              f"scale {float(p['scale']):.4f}/{med[2]:.4f}")
        payload = dict(payload)
        payload["param_samples"] = ensembles[family]
        payload["bootstrap_meta"] = meta
        assert not os.path.exists(dst_fp) or dst_fp.endswith("_boot.pkl")
        with open(dst_fp, "wb") as f:
            pickle.dump(payload, f)
        print(f"Wrote {dst_fp} (added param_samples: {ensembles[family].shape})")

if __name__ == "__main__":
    main()
