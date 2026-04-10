import os
import torch

# Paths
DATA_ROOT = "F:/datasets/SkinCancer"
GROUND_TRUTH_CSV = os.path.join(DATA_ROOT, "GroundTruth.csv")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Class info
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_NAMES_FULL = [
    "Melanoma", "Melanocytic nevus", "Basal cell carcinoma",
    "Actinic keratosis", "Benign keratosis", "Dermatofibroma", "Vascular lesion"
]
NUM_CLASSES = 7

# Data split
TRAIN_RATIO = 0.8  # 80% train+val, 20% test (at lesion_id level)
VAL_RATIO = 0.1    # 10% of total as validation
RANDOM_SEED = 42

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 40
FREEZE_EPOCHS = 5
HYBRID_FREEZE_EPOCHS = 10
LR_HEAD = 7e-4
LR_BACKBONE = 5e-5
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 12
LABEL_SMOOTHING = 0.05
GRAD_CLIP_MAX_NORM = 1.0
HEAD_DROPOUT = 0.2

# Input
IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Num workers
NUM_WORKERS = 4
PIN_MEMORY = True

# Model names for iteration
MODEL_NAMES = [
    "ResNet-50",
    "DenseNet-121",
    "EfficientNet-B0",
    "ViT-B/16",
    "Hybrid CNN-Transformer",
]
