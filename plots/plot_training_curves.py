import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"relu": "#1f77b4", "leaky_relu": "#ff7f0e", "swish": "#2ca02c", "gelu": "#d62728"}


def load_results() -> dict[str, dict[str, dict]]:
    """Returns {dataset: {activation: metrics}}."""
    datasets: dict[str, dict] = {}
    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        f = exp_dir / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        ds = m["config"]["dataset"]
        act = m["config"]["activation"]
        datasets.setdefault(ds, {})[act] = m
    return datasets


def main():
    datasets = load_results()
    if not datasets:
        print("No results found in results/. Run experiments/run_all.py first.")
        return

    for dataset, activations in datasets.items():
        fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Training Curves — {dataset.upper()}", fontsize=13)

        for act_name, m in activations.items():
            color = COLORS.get(act_name, "black")
            epochs = m["epoch"]
            ax_acc.plot(epochs, m["val_acc"], color=color, label=act_name, linewidth=2)
            ax_acc.plot(epochs, m["train_acc"], color=color, linestyle="--", alpha=0.4)
            ax_loss.plot(epochs, m["val_loss"], color=color, label=act_name, linewidth=2)
            ax_loss.plot(epochs, m["train_loss"], color=color, linestyle="--", alpha=0.4)

        for ax, title, ylabel in [
            (ax_acc, "Accuracy  (solid=val, dashed=train)", "Accuracy"),
            (ax_loss, "Loss  (solid=val, dashed=train)", "Loss"),
        ]:
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = OUTPUT_DIR / f"training_curves_{dataset}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
