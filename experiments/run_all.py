import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from activation_exp.configs import EXPERIMENTS
from activation_exp.data import get_dataloaders
from activation_exp.trainer import train_one_experiment
from activation_exp.utils import get_device, set_seed

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    device = get_device()
    set_seed(42)
    print(f"Device: {device}")
    print(f"Running {len(EXPERIMENTS)} experiments\n")

    for i, config in enumerate(EXPERIMENTS, 1):
        results_dir = RESULTS_DIR / config.name
        if (results_dir / "metrics.json").exists():
            print(f"[{i}/{len(EXPERIMENTS)}] SKIP {config.name} (already done)")
            continue

        print(f"[{i}/{len(EXPERIMENTS)}] START {config.name}")
        train_loader, val_loader = get_dataloaders(config)
        train_one_experiment(config, train_loader, val_loader, results_dir, device)
        print(f"[{i}/{len(EXPERIMENTS)}] DONE  {config.name}\n")

    print("All experiments complete.")


if __name__ == "__main__":
    main()
