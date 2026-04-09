import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import (
    GROUND_TRUTH_CSV, IMAGES_DIR, CLASS_NAMES, NUM_CLASSES,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED, BATCH_SIZE,
    IMAGE_SIZE, RESIZE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    NUM_WORKERS, PIN_MEMORY, DATA_ROOT,
)


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class HAM10000Dataset(Dataset):
    def __init__(self, image_ids, labels, transform=None):
        self.image_ids = image_ids
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


def load_and_split_data():
    """Load GroundTruth.csv + metadata, split by lesion_id."""
    gt = pd.read_csv(GROUND_TRUTH_CSV)
    meta_path = os.path.join(DATA_ROOT, "HAM10000_metadata.csv")
    meta = pd.read_csv(meta_path)

    # Merge to get lesion_id for each image
    merged = gt.merge(meta[["image_id", "lesion_id"]], left_on="image", right_on="image_id")

    # Convert one-hot to class index
    label_cols = CLASS_NAMES
    merged["label"] = merged[label_cols].values.argmax(axis=1)

    # Get unique lesion_ids with their class (majority class for lesion)
    lesion_labels = merged.groupby("lesion_id")["label"].first().reset_index()

    # Split lesion_ids: 80% train+val, 20% test
    train_val_lesions, test_lesions = train_test_split(
        lesion_labels["lesion_id"],
        test_size=1 - TRAIN_RATIO,
        stratify=lesion_labels["label"],
        random_state=RANDOM_SEED,
    )

    # Split train_val into train and val
    # val = VAL_RATIO of total, so val_fraction = VAL_RATIO / TRAIN_RATIO
    train_val_label_df = lesion_labels[lesion_labels["lesion_id"].isin(train_val_lesions)]
    val_fraction = VAL_RATIO / TRAIN_RATIO
    train_lesions, val_lesions = train_test_split(
        train_val_label_df["lesion_id"],
        test_size=val_fraction,
        stratify=train_val_label_df["label"],
        random_state=RANDOM_SEED,
    )

    train_lesions = set(train_lesions)
    val_lesions = set(val_lesions)
    test_lesions = set(test_lesions)

    # Assign images to splits
    train_mask = merged["lesion_id"].isin(train_lesions)
    val_mask = merged["lesion_id"].isin(val_lesions)
    test_mask = merged["lesion_id"].isin(test_lesions)

    splits = {}
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        subset = merged[mask]
        splits[name] = {
            "image_ids": subset["image"].tolist(),
            "labels": subset["label"].tolist(),
        }

    print(f"Split sizes — Train: {len(splits['train']['labels'])}, "
          f"Val: {len(splits['val']['labels'])}, "
          f"Test: {len(splits['test']['labels'])}")

    # Print class distribution per split
    for name in ["train", "val", "test"]:
        counts = np.bincount(splits[name]["labels"], minlength=NUM_CLASSES)
        print(f"  {name}: {dict(zip(CLASS_NAMES, counts))}")

    return splits


def compute_class_weights(labels):
    """Compute inverse-frequency class weights."""
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = len(labels)
    weights = total / (NUM_CLASSES * counts.astype(float))
    return torch.FloatTensor(weights)


def get_dataloaders():
    """Create train/val/test dataloaders."""
    splits = load_and_split_data()

    train_dataset = HAM10000Dataset(
        splits["train"]["image_ids"],
        splits["train"]["labels"],
        transform=get_train_transforms(),
    )
    val_dataset = HAM10000Dataset(
        splits["val"]["image_ids"],
        splits["val"]["labels"],
        transform=get_test_transforms(),
    )
    test_dataset = HAM10000Dataset(
        splits["test"]["image_ids"],
        splits["test"]["labels"],
        transform=get_test_transforms(),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    class_weights = compute_class_weights(splits["train"]["labels"])

    return train_loader, val_loader, test_loader, class_weights
