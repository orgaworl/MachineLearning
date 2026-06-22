import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACT_ORDER = ["relu", "leaky_relu", "swish", "gelu"]


def load_data() -> dict[str, dict[str, dict]]:
    """Returns {dataset: {activation: {epoch_key: np.ndarray}}}."""
    datasets: dict[str, dict] = {}
    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        metrics_f = exp_dir / "metrics.json"
        dist_f = exp_dir / "activation_dist.npz"
        if not metrics_f.exists() or not dist_f.exists():
            continue
        m = json.loads(metrics_f.read_text())
        ds, act = m["config"]["dataset"], m["config"]["activation"]
        npz = np.load(dist_f)
        datasets.setdefault(ds, {})[act] = {k: npz[k] for k in npz.files}
    return datasets


def main():
    datasets = load_data()
    if not datasets:
        print("No results found. Run experiments/run_all.py first.")
        return

    for dataset, act_data in datasets.items():
        all_keys = sorted(
            set().union(*[set(d.keys()) for d in act_data.values()]),
            key=lambda k: int(k.split("_")[1]),
        )
        acts_present = [a for a in ACT_ORDER if a in act_data]
        n_rows, n_cols = len(acts_present), len(all_keys)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        fig.suptitle(f"Activation Distribution — {dataset.upper()}", fontsize=13)

        for row_i, act_name in enumerate(acts_present):
            for col_j, epoch_key in enumerate(all_keys):
                ax = axes[row_i, col_j]
                if epoch_key in act_data[act_name]:
                    vals = act_data[act_name][epoch_key]
                    ax.hist(vals, bins=60, density=True, alpha=0.75,
                            color="steelblue", edgecolor="none")
                    ax.axvline(0, color="red", linewidth=0.8, linestyle="--")
                if row_i == 0:
                    ax.set_title(epoch_key.replace("epoch_", "Ep."), fontsize=9)
                if col_j == 0:
                    ax.set_ylabel(act_name, fontsize=9)
                ax.tick_params(labelsize=7)
                ax.set_xlabel("")

        plt.tight_layout()
        out = OUTPUT_DIR / f"activation_dist_{dataset}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
