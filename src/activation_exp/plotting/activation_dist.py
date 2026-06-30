import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = "PingFang SC"
matplotlib.rcParams["axes.unicode_minus"] = False

_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = _ROOT / "results"
OUTPUT_DIR = _ROOT / "plots"

ACT_ORDER = ["relu", "leaky_relu", "swish", "gelu"]

DATASET_NAMES = {
    "mnist": "MNIST 手写数字",
    "cifar10": "CIFAR-10（浅层+BN）",
    "cifar10_lite": "CIFAR-10 轻量配置",
    "cifar10_deep": "CIFAR-10（深层，无BN）",
}


def load_data() -> dict[str, dict[str, dict]]:
    """Returns {dataset_group: {activation: {epoch_key: np.ndarray}}}."""
    datasets: dict[str, dict] = {}
    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        metrics_f = exp_dir / "metrics.json"
        dist_f = exp_dir / "activation_dist.npz"
        if not metrics_f.exists() or not dist_f.exists():
            continue
        m = json.loads(metrics_f.read_text())
        ds = m["config"]["dataset"]
        model_type = m["config"].get("model_type", "cnn")
        group = f"{ds}_deep" if model_type == "deep_cnn" else ds
        act = m["config"]["activation"]
        npz = np.load(dist_f)
        datasets.setdefault(group, {})[act] = {k: npz[k] for k in npz.files}
    return datasets


def main():
    datasets = load_data()
    if not datasets:
        print("No results found. Run: uv run train")
        return

    for dataset, act_data in datasets.items():
        all_keys = sorted(
            set().union(*[set(d.keys()) for d in act_data.values()]),
            key=lambda k: int(k.split("_")[1]),
        )
        n_keep = 4
        if len(all_keys) > n_keep:
            step = (len(all_keys) - 1) / (n_keep - 1)
            all_keys = [all_keys[round(i * step)] for i in range(n_keep)]
        acts_present = [a for a in ACT_ORDER if a in act_data]
        n_rows, n_cols = len(acts_present), len(all_keys)

        all_vals_flat = np.concatenate([
            act_data[a][k]
            for a in acts_present
            for k in all_keys
            if k in act_data[a]
        ])
        x_lo = float(np.percentile(all_vals_flat, 0.5))
        x_hi = float(np.percentile(all_vals_flat, 99.5))

        y_hi = 0.0
        for a in acts_present:
            for k in all_keys:
                if k not in act_data[a]:
                    continue
                counts, _ = np.histogram(
                    act_data[a][k], bins=60, range=(x_lo, x_hi), density=True
                )
                y_hi = max(y_hi, float(counts.max()))
        y_hi *= 1.15

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        ds_label = DATASET_NAMES.get(dataset, dataset.upper())
        fig.suptitle(
            f"第2卷积层激活值分布 — {ds_label}\n"
            f"（行=激活函数，列=训练轮次；红色虚线为 x=0；分布越宽说明激活值越分散）",
            fontsize=12,
        )

        for row_i, act_name in enumerate(acts_present):
            for col_j, epoch_key in enumerate(all_keys):
                ax = axes[row_i, col_j]
                if epoch_key in act_data[act_name]:
                    vals = act_data[act_name][epoch_key]
                    ax.hist(vals, bins=60, range=(x_lo, x_hi), density=True,
                            alpha=0.75, color="steelblue", edgecolor="none")
                    ax.axvline(0, color="red", linewidth=0.8, linestyle="--")
                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(0, y_hi)
                if row_i == 0:
                    epoch_num = epoch_key.replace("epoch_", "第") + "轮"
                    ax.set_title(epoch_num, fontsize=9)
                if col_j == 0:
                    ax.set_ylabel(act_name, fontsize=9)
                ax.tick_params(labelsize=7)
                ax.set_xlabel("")

        plt.tight_layout()
        stem = OUTPUT_DIR / f"activation_dist_{dataset}"
        plt.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close()
        print(f"Saved: {stem}.png / .pdf")


if __name__ == "__main__":
    main()
