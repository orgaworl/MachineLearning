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
    """Returns {dataset_group: {activation: metrics}}."""
    datasets: dict[str, dict] = {}
    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        f = exp_dir / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        ds = m["config"]["dataset"]
        model_type = m["config"].get("model_type", "cnn")
        group = f"{ds}_deep" if model_type == "deep_cnn" else ds
        act = m["config"]["activation"]
        datasets.setdefault(group, {})[act] = m
    return datasets


def main():
    datasets = load_results()
    if not datasets:
        print("No results found in results/. Run experiments/run_all.py first.")
        return

    for dataset, activations in datasets.items():
        ds_label = DATASET_NAMES.get(dataset, dataset.upper())
        fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"训练曲线对比 — {ds_label}\n（实线=验证集，虚线=训练集）", fontsize=13)

        for act_name, m in activations.items():
            color = COLORS.get(act_name, "black")
            epochs = m["epoch"]
            ax_acc.plot(epochs, m["val_acc"], color=color, label=act_name, linewidth=2)
            ax_acc.plot(epochs, m["train_acc"], color=color, linestyle="--", alpha=0.4)
            ax_loss.plot(epochs, m["val_loss"], color=color, label=act_name, linewidth=2)
            ax_loss.plot(epochs, m["train_loss"], color=color, linestyle="--", alpha=0.4)

        all_acc = [v for m in activations.values() for v in m["train_acc"] + m["val_acc"]]
        all_loss = [v for m in activations.values() for v in m["train_loss"] + m["val_loss"]]
        acc_margin = (max(all_acc) - min(all_acc)) * 0.1 or 0.01
        loss_margin = (max(all_loss) - min(all_loss)) * 0.1 or 0.01
        ax_acc.set_ylim(max(0, min(all_acc) - acc_margin), min(1, max(all_acc) + acc_margin))
        ax_loss.set_ylim(max(0, min(all_loss) - loss_margin), max(all_loss) + loss_margin)

        for ax, title, ylabel in [
            (ax_acc, "准确率（Accuracy）", "准确率"),
            (ax_loss, "损失值（Loss）", "损失值"),
        ]:
            ax.set_title(title)
            ax.set_xlabel("训练轮次（Epoch）")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        stem = OUTPUT_DIR / f"training_curves_{dataset}"
        plt.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close()
        print(f"Saved: {stem}.png / .pdf")


if __name__ == "__main__":
    main()
