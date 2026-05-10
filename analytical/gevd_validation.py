import os
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from cycler import cycler

COLORS2HEX = {
    "petrol":     "#264653",
    "gold":       "#E9C46A",
    "teal":       "#2A9D8F",
    "sand":       "#F4A261",
    "terracotta": "#E76F51",
}

def calc_expected_value_gumbel(mu, sigma, k=1):
    gamma = 0.5772156649  # Euler-Mascheroni constant
    return mu + (sigma * (gamma + np.log(k)))

def calc_expected_value_frechet(mu, sigma, kappa, k=1):
    assert kappa != 0
    kappa = -1 * kappa # because scipy does the opposite sign
    return mu + (sigma * (((k ** kappa) * math.gamma(1 - kappa)) - 1) / kappa)

def _combo_stats(combo, arch, kernel_data, per_kernel):
    if per_kernel:
        if any((k, arch) not in kernel_data for k in combo):
            return None
        mean = float(sum(kernel_data[(k, arch)]["mean"] for k in combo))
        std = float(math.sqrt(sum(kernel_data[(k, arch)]["std"] ** 2 for k in combo)))
    else:
        if (combo, arch) not in kernel_data:
            return None
        mean = float(kernel_data[(combo, arch)]["mean"])
        std = float(kernel_data[(combo, arch)]["std"])
    return mean, std

def compute_workload(workload_prefix, arch, kernel_data, workload_spec, gevd_params, per_kernel):
    path_tracing_fp = f"<PATH_TO_VALIDATION_DATA>/path-traces/{workload_prefix}-{arch}.pkl" # e.g., ../validation-data/path-traces/{workload_prefix}-{arch}.pkl
    with open(path_tracing_fp, 'rb') as f:
        path_traces = pickle.load(f)

    IGNORE = {"CPU", "MemEffAttention", "sdpa"}
    predicted_delays = []
    empirical_delays = []
    kernel_combos = []
    group_total, group_size = workload_spec
    if gevd_params['shape'] == 0:
        z = calc_expected_value_gumbel(gevd_params['loc'], gevd_params['scale'], group_size / gevd_params['m'])
    else:
        z = calc_expected_value_frechet(gevd_params['loc'], gevd_params['scale'], gevd_params['shape'], group_size / gevd_params['m'])

    for iter, iter_data in path_traces.items():
        for start_rank in range(0, group_total, group_size):
            end_rank = start_rank + group_size
            group = range(start_rank, end_rank)
            comm_lens = {r: len(iter_data['comm'][r]) for r in group}
            comp_lens = {r: len(iter_data['comp'][r]) for r in group}
            assert len(set(comm_lens.values())) == 1, \
                f"{iter} group [{start_rank},{end_rank}): comm lengths differ: {comm_lens}"
            assert len(set(comp_lens.values())) == 1, \
                f"{iter} group [{start_rank},{end_rank}): comp lengths differ: {comp_lens}"

            num_events = comp_lens[start_rank]
            for ev_idx in range(num_events):
                op_names = {r: tuple(iter_data['comp'][r][ev_idx]) for r in group}
                if any(len(op_names[r]) == 0 for r in op_names): continue
                op_names = {k: tuple(x[0] for x in v) for k, v in op_names.items()}
                if len(set(op_names.values())) != 1: continue
                if any(sub in name for r in op_names for name in op_names[r] for sub in IGNORE): continue
                op_name = list(op_names.values())[0]
                stats = _combo_stats(op_name, arch, kernel_data, per_kernel)
                if stats is None: continue
                _, comp_dur = stats
                assert comp_dur > 0
                straggler_time = min(iter_data['comm'][r][ev_idx] for r in group)
                empirical_delay = np.mean([iter_data['comm'][r][ev_idx] - straggler_time for r in group])
                empirical_delays.append(empirical_delay / 1000.0)
                kernel_combos.append(op_name)
                predicted_delays.append(comp_dur * z)

    predicted = np.array(predicted_delays)
    empirical = np.array(empirical_delays)

    levels = np.unique(predicted)
    xs, means, sems = [], [], []
    for lvl in levels:
        vals = empirical[predicted == lvl]
        if len(vals) < 2:
            continue
        xs.append(lvl)
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(len(vals)))

    combo_to_idx = {}
    for combo in kernel_combos:
        if combo not in combo_to_idx:
            combo_to_idx[combo] = len(combo_to_idx)
    combo_idx_arr = np.array([combo_to_idx[c] for c in kernel_combos])

    bar_combos, bar_pred, bar_emp_mean, bar_emp_sem = [], [], [], []
    for combo, idx in combo_to_idx.items():
        mask = combo_idx_arr == idx
        vals = empirical[mask]
        if len(vals) < 2:
            continue
        combo_mean, combo_std = _combo_stats(combo, arch, kernel_data, per_kernel)
        bar_combos.append({
            'name': combo,
            'mean': combo_mean,
            'std': combo_std,
            'cov': combo_std / combo_mean,
        })
        bar_pred.append(predicted[mask][0])
        bar_emp_mean.append(vals.mean())
        bar_emp_sem.append(vals.std(ddof=1) / np.sqrt(len(vals)))

    order = np.argsort(bar_pred)
    bar_pred = np.array(bar_pred)[order] if len(bar_pred) else np.array([])
    bar_emp_mean = np.array(bar_emp_mean)[order] if len(bar_emp_mean) else np.array([])
    bar_emp_sem = np.array(bar_emp_sem)[order] if len(bar_emp_sem) else np.array([])
    bar_combos = [bar_combos[i] for i in order]

    return {
        'n': len(empirical),
        'xs': np.array(xs),
        'means': np.array(means),
        'sems': np.array(sems),
        'bar_pred': bar_pred,
        'bar_emp_mean': bar_emp_mean,
        'bar_emp_sem': bar_emp_sem,
        'bar_combos': bar_combos,
    }

if __name__ == "__main__":
    WORKLOADS = {"n8-tp-qwen32b": (8, 8), "n8-fsdp-tp-llama70b": (8, 4)}
    TITLES = {"n8-tp-qwen32b": r"Qwen-3$\,$32B, TP=8", "n8-fsdp-tp-llama70b": r"Llama-3$\,$70B, TP=4$\,$x$\,$FSDP=2"}
    ARCHS = ["a100", "h100", "h200"]
    M = 4
    MODES = [
        ("per_kernel", "<PER_KERNEL_DATA_PKL_FILE>", True),
        ("per_combo", "<PER_COMPUTE_BLOCK_DATA_PKL_FILE>", False),
    ]
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'Fira Code',
        'axes.prop_cycle': cycler(color=list(COLORS2HEX.values())),
    })

    for mode_name, data_fp, per_kernel in MODES:
        with open(data_fp, 'rb') as f:
            payload = pickle.load(f)
        kernel_data = payload["data"]
        gevd_params = dict(payload["params"])
        gevd_params["m"] = M
        print(f"[{mode_name}] GEVD params: {gevd_params}")

        results = {}
        for workload_prefix, spec in WORKLOADS.items():
            for arch in ARCHS:
                key = (workload_prefix, arch)
                results[key] = compute_workload(workload_prefix, arch, kernel_data, spec, gevd_params, per_kernel=per_kernel)
                print(f"[{mode_name}] {workload_prefix}-{arch}: n = {results[key]['n']}")

        out_dir = "<PATH_TO_OUTPUT_DIR>"
        os.makedirs(out_dir, exist_ok=True)

        nrows = len(WORKLOADS)
        ncols = len(ARCHS)

        # Scatter / errorbar validation figure
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows), squeeze=False)
        scatter_handles = None
        for i, (workload_prefix, _) in enumerate(WORKLOADS.items()):
            for j, arch in enumerate(ARCHS):
                ax = axes[i][j]
                letter = chr(ord('a') + i * ncols + j)
                title = f"({letter}) {TITLES[workload_prefix]}, {arch.upper()}"
                r = results[(workload_prefix, arch)]
                xs, means, sems = r['xs'] * 1000.0, r['means'] * 1000.0, r['sems'] * 1000.0
                if len(xs) == 0:
                    ax.set_title(f"{title} (no data)")
                    continue
                ax.errorbar(xs, means, yerr=sems, fmt='o', capsize=4, label='Mean empirical ± SEM')
                lim = max(xs.max(), (means + sems).max()) * 1.05
                ax.plot([0, lim], [0, lim], 'r--', linewidth=1, label='y = x')
                ax.set_xlabel("Predicted mean delay (µs)", fontsize=16)
                ax.set_ylabel("Mean empirical delay (µs)", fontsize=16)
                ax.set_title(title, fontsize=16)
                if scatter_handles is None:
                    scatter_handles = ax.get_legend_handles_labels()
        if scatter_handles is not None:
            fig.legend(*scatter_handles, loc='lower center', ncol=len(scatter_handles[0]),
                       bbox_to_anchor=(0.5, -0.02), fontsize=16, frameon=False)
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.show()
   
        plt.clf()
        # Bar chart figure
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
        bar_handles = None
        for i, (workload_prefix, _) in enumerate(WORKLOADS.items()):
            for j, arch in enumerate(ARCHS):
                ax = axes[i][j]
                letter = chr(ord('a') + i * ncols + j)
                title = f"({letter}) {TITLES[workload_prefix]}, {arch.upper()}"
                r = results[(workload_prefix, arch)]
                bar_pred = r['bar_pred'] * 1000.0
                bar_emp_mean = r['bar_emp_mean'] * 1000.0
                bar_emp_sem = r['bar_emp_sem'] * 1000.0
                n = len(bar_pred)
                if n == 0:
                    ax.set_title(f"{title} (no data)")
                    continue
                x = np.arange(n)
                width = 0.4
                ax.bar(x - width / 2, bar_emp_mean, width, yerr=bar_emp_sem, capsize=3,
                       label='Mean empirical ± SEM')
                ax.bar(x + width / 2, bar_pred, width, label='Predicted')
                ax.set_xticks(x)
                ax.set_xticklabels([str(k) for k in range(n)])
                ax.set_xlabel("Unique compute block", fontsize=22)
                ax.set_ylabel("Sync delay (µs)", fontsize=22)
                ax.set_title(title, fontsize=21)
                if bar_handles is None:
                    bar_handles = ax.get_legend_handles_labels()
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.show()
