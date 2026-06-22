import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"relu": "#1f77b4", "leaky_relu": "#ff7f0e", "swish": "#2ca02c", "gelu": "#d62728"}


def load_results() -> dict[str, dict[str, dict]]:
    datasets: dict[str, dict] = {}
    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        f = exp_dir / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        datasets.setdefault(m["config"]["dataset"], {})[m["config"]["activation"]] = m
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
    fig.suptitle("Dead Neuron Ratio Over Training", fontsize=13)

    for ax, (dataset, activations) in zip(axes, datasets.items()):
        for act_name, m in activations.items():
            ax.plot(m["epoch"], m["dead_ratio"],
                    color=COLORS.get(act_name, "black"), label=act_name, linewidth=2)
        ax.set_title(dataset.upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dead Neuron Ratio")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out1 = OUTPUT_DIR / "dead_neurons_dynamic.png"
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")

    # Figure 2: final dead ratio bar chart
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Final Dead Neuron Ratio", fontsize=13)

    for ax, (dataset, activations) in zip(axes, datasets.items()):
        act_names = list(activations.keys())
        final_ratios = [activations[a]["dead_ratio"][-1] for a in act_names]
        bar_colors = [COLORS.get(a, "gray") for a in act_names]
        bars = ax.bar(act_names, final_ratios, color=bar_colors, alpha=0.85)
        for bar, ratio in zip(bars, final_ratios):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{ratio:.3f}", ha="center", va="bottom", fontsize=9,
            )
        ax.set_title(dataset.upper())
        ax.set_xlabel("Activation")
        ax.set_ylabel("Dead Neuron Ratio")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out2 = OUTPUT_DIR / "dead_neurons_final.png"
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
