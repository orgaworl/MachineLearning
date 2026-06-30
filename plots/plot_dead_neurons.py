import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "PingFang SC"
matplotlib.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"relu": "#1f77b4", "leaky_relu": "#ff7f0e", "swish": "#2ca02c", "gelu": "#d62728"}

DATASET_NAMES = {
    "mnist": "MNIST 手写数字",
    "cifar10": "CIFAR-10（浅层+BN）",
    "cifar10_lite": "CIFAR-10 轻量配置",
    "cifar10_deep": "CIFAR-10（深层，无BN）",
}


def load_results() -> dict[str, dict[str, dict]]:
    datasets: dict[str, dict] = {}
    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        f = exp_dir / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        ds = m["config"]["dataset"]
        model_type = m["config"].get("model_type", "cnn")
        group = f"{ds}_deep" if model_type == "deep_cnn" else ds
        datasets.setdefault(group, {})[m["config"]["activation"]] = m
    return datasets


def main():
    datasets = load_results()
    if not datasets:
        print("No results found. Run experiments/run_all.py first.")
        return

    n = len(datasets)

    # Figure 1: dynamic dead ratio per epoch
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("各激活函数死亡神经元比例随训练变化\n（死亡神经元：在整个验证集上输出始终 ≤ 0 的神经元）", fontsize=13)

    for ax, (dataset, activations) in zip(axes, datasets.items()):
        for act_name, m in activations.items():
            pct = [v * 100 for v in m["dead_ratio"]]
            ax.plot(m["epoch"], pct,
                    color=COLORS.get(act_name, "black"), label=act_name, linewidth=2)
        all_vals = [v * 100 for m in activations.values() for v in m["dead_ratio"]]
        y_max = max(all_vals) if all_vals else 100.0
        ax.set_ylim(0, max(y_max * 1.25, 2.0))
        ax.set_title(DATASET_NAMES.get(dataset, dataset.upper()))
        ax.set_xlabel("训练轮次（Epoch）")
        ax.set_ylabel("死亡神经元比例（%）")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    stem1 = OUTPUT_DIR / "dead_neurons_dynamic"
    plt.savefig(stem1.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(stem1.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {stem1}.png / .pdf")

    # Figure 2: final dead ratio bar chart
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("训练结束时各激活函数的死亡神经元比例\n（柱顶数字为具体比例值）", fontsize=13)

    for ax, (dataset, activations) in zip(axes, datasets.items()):
        act_names = list(activations.keys())
        final_pcts = [activations[a]["dead_ratio"][-1] * 100 for a in act_names]
        bar_colors = [COLORS.get(a, "gray") for a in act_names]
        bars = ax.bar(act_names, final_pcts, color=bar_colors, alpha=0.85)
        y_max = max(final_pcts) if final_pcts else 100.0
        ax.set_ylim(0, max(y_max * 1.3, 2.0))
        for bar, pct in zip(bars, final_pcts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.02,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9,
            )
        ax.set_title(DATASET_NAMES.get(dataset, dataset.upper()))
        ax.set_xlabel("激活函数")
        ax.set_ylabel("死亡神经元比例（%）")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    stem2 = OUTPUT_DIR / "dead_neurons_final"
    plt.savefig(stem2.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(stem2.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {stem2}.png / .pdf")


if __name__ == "__main__":
    main()
