import os
import numpy as np
import torch
from torch.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from src.config import DEVICE, CLASS_NAMES, NUM_CLASSES, RESULTS_DIR


def evaluate_model(model, test_loader, model_name):
    """Evaluate model on test set. Returns dict of metrics."""
    model = model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type):
                outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Overall metrics
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_metrics = {CLASS_NAMES[i]: float(per_class_f1[i]) for i in range(NUM_CLASSES)}

    # Save classification report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    report_path = os.path.join(
        RESULTS_DIR, "classification_reports",
        f"{model_name.replace('/', '-')}.txt",
    )
    with open(report_path, "w") as f:
        f.write(f"Classification Report: {model_name}\n")
        f.write("=" * 60 + "\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    _save_confusion_matrix(cm, model_name)

    print(f"  {model_name} — Accuracy: {metrics['accuracy']:.4f}, "
          f"F1 macro: {metrics['f1_macro']:.4f}, F1 weighted: {metrics['f1_weighted']:.4f}")

    return metrics, per_class_metrics


def _save_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(
        RESULTS_DIR, "confusion_matrices",
        f"{model_name.replace('/', '-')}.png",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(all_logs, model_names):
    """Plot training curves for all models on one figure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Loss curves
    for name, log in zip(model_names, all_logs):
        epochs = [e["epoch"] for e in log]
        axes[0].plot(epochs, [e["val_loss"] for e in log], label=name)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Validation Loss by Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    for name, log in zip(model_names, all_logs):
        epochs = [e["epoch"] for e in log]
        axes[1].plot(epochs, [e["val_acc"] for e in log], label=name)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy")
    axes[1].set_title("Validation Accuracy by Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)
