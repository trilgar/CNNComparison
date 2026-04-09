import os
import csv
import random
import numpy as np
import torch

from src.config import (
    DEVICE, RANDOM_SEED, MODEL_NAMES, MODELS_DIR, RESULTS_DIR, NUM_CLASSES,
)
from src.dataset import get_dataloaders
from src.models import get_model, count_parameters
from src.train import train_model
from src.evaluate import evaluate_model, plot_training_curves


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(RANDOM_SEED)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("ERROR: CUDA not available. Aborting — training on CPU is too slow.")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "confusion_matrices"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "training_logs"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "classification_reports"), exist_ok=True)

    # Load data once
    print("\n=== Loading data ===")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders()

    all_metrics = []
    all_per_class = []
    all_logs = []
    trained_names = []

    for model_name in MODEL_NAMES:
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")

        set_seed(RANDOM_SEED)

        model, param_groups = get_model(model_name)
        num_params = count_parameters(model)
        print(f"  Parameters: {num_params / 1e6:.1f}M")

        # Train
        training_log, train_time = train_model(
            model, model_name, param_groups,
            train_loader, val_loader, class_weights,
        )

        # Evaluate
        metrics, per_class = evaluate_model(model, test_loader, model_name)
        metrics["params_M"] = round(num_params / 1e6, 1)
        metrics["train_time_min"] = round(train_time / 60, 1)

        all_metrics.append(metrics)
        all_per_class.append(per_class)
        all_logs.append(training_log)
        trained_names.append(model_name)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Save comparison table
    comp_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    fields = ["model", "accuracy", "precision_macro", "recall_macro",
              "f1_macro", "f1_weighted", "params_M", "train_time_min"]
    with open(comp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: m[k] for k in fields})
    print(f"\nComparison table saved: {comp_path}")

    # Save per-class F1 table
    pc_path = os.path.join(RESULTS_DIR, "per_class_f1.csv")
    with open(pc_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + list(all_per_class[0].keys()))
        writer.writeheader()
        for name, pc in zip(trained_names, all_per_class):
            row = {"model": name}
            row.update(pc)
            writer.writerow(row)
    print(f"Per-class F1 table saved: {pc_path}")

    # Plot training curves
    plot_training_curves(all_logs, trained_names)
    print(f"Training curves saved: {os.path.join(RESULTS_DIR, 'training_curves.png')}")

    print("\n=== All experiments complete ===")


if __name__ == "__main__":
    main()
