import os
import time
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.config import (
    DEVICE, NUM_EPOCHS, FREEZE_EPOCHS, HYBRID_FREEZE_EPOCHS, LR_HEAD, LR_BACKBONE,
    WEIGHT_DECAY, EARLY_STOPPING_PATIENCE, MODELS_DIR, RESULTS_DIR,
    LABEL_SMOOTHING, GRAD_CLIP_MAX_NORM,
)
from src.models import freeze_backbone, unfreeze_all


def _gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


def train_model(model, model_name, param_groups, train_loader, val_loader, class_weights):
    """Train a model with freeze/unfreeze strategy, mixed precision, early stopping.

    Returns: training_log (list of dicts), total_time (seconds).
    """
    model = model.to(DEVICE)
    class_weights = class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    # Determine freeze duration per model
    freeze_epochs = HYBRID_FREEZE_EPOCHS if model_name == "Hybrid CNN-Transformer" else FREEZE_EPOCHS

    # Phase 1: frozen backbone — train head only
    freeze_backbone(model, model_name)
    optimizer = AdamW(
        [{"params": [p for p in param_groups[1] if p.requires_grad], "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler()

    training_log = []
    best_val_loss = float("inf")
    patience_counter = 0
    save_path = os.path.join(MODELS_DIR, f"{model_name.replace('/', '-')}_best.pth")

    start_time = time.time()

    # Epoch-level progress bar
    epoch_bar = tqdm(range(1, NUM_EPOCHS + 1), desc=f"[{model_name}]", unit="epoch")

    for epoch in epoch_bar:
        # Unfreeze after freeze_epochs
        if epoch == freeze_epochs + 1:
            unfreeze_all(model)
            optimizer = AdamW([
                {"params": param_groups[0], "lr": LR_BACKBONE},
                {"params": param_groups[1], "lr": LR_HEAD},
            ], weight_decay=WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - freeze_epochs)
            tqdm.write(f"  >>> Epoch {epoch}: backbone unfrozen, differential LR active")

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        batch_bar = tqdm(
            train_loader,
            desc=f"  Train ep{epoch}",
            leave=False,
            bar_format="{l_bar}{bar:20}{r_bar}",
        )
        for images, labels in batch_bar:
            images = images.to(DEVICE, non_blocking=True)
            labels = torch.as_tensor(labels, dtype=torch.long, device=DEVICE)

            optimizer.zero_grad()
            with autocast(device_type=DEVICE.type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            train_loss += batch_loss * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

            # Live stats on batch bar
            running_acc = train_correct / train_total
            batch_bar.set_postfix(
                loss=f"{batch_loss:.3f}",
                acc=f"{running_acc:.3f}",
                gpu=f"{_gpu_mem_mb():.0f}MB",
            )

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validate ---
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion)

        # Check if best
        is_best = val_loss < best_val_loss
        best_marker = " *" if is_best else ""

        # Update epoch bar postfix
        elapsed = time.time() - start_time
        epoch_bar.set_postfix(
            tl=f"{train_loss:.3f}",
            ta=f"{train_acc:.3f}",
            vl=f"{val_loss:.3f}",
            va=f"{val_acc:.3f}",
            best=f"{best_val_loss:.3f}",
            pat=f"{patience_counter}/{EARLY_STOPPING_PATIENCE}",
        )

        # Print summary line
        phase = "frozen" if epoch <= freeze_epochs else "full"
        tqdm.write(
            f"  Epoch {epoch:2d}/{NUM_EPOCHS} [{phase:6s}] "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={optimizer.param_groups[-1]['lr']:.1e}  "
            f"[{elapsed/60:.1f}min]{best_marker}"
        )

        # Log
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[-1]["lr"],
        }
        training_log.append(log_entry)

        # Best model checkpoint
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                tqdm.write(f"  >>> Early stopping at epoch {epoch} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    total_time = time.time() - start_time
    tqdm.write(f"  >>> {model_name} done in {total_time/60:.1f} min | best val_loss={best_val_loss:.4f}\n")

    # Load best weights
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))

    # Save training log
    log_path = os.path.join(RESULTS_DIR, "training_logs", f"{model_name.replace('/', '-')}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        writer.writeheader()
        writer.writerows(training_log)

    return training_log, total_time


def evaluate_epoch(model, loader, criterion):
    """Run validation for one epoch, return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = torch.as_tensor(labels, dtype=torch.long, device=DEVICE)

            with autocast(device_type=DEVICE.type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total
