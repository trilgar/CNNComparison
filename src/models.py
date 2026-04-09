import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet50_Weights, DenseNet121_Weights,
    EfficientNet_B0_Weights, ViT_B_16_Weights,
)

from src.config import NUM_CLASSES


def get_model(name):
    """Return (model, param_groups) for the given architecture name.

    param_groups is a list of dicts for the optimizer:
      [{"params": backbone_params}, {"params": head_params}]
    """
    if name == "ResNet-50":
        return _build_resnet50()
    elif name == "DenseNet-121":
        return _build_densenet121()
    elif name == "EfficientNet-B0":
        return _build_efficientnet_b0()
    elif name == "ViT-B/16":
        return _build_vit_b16()
    elif name == "Hybrid CNN-Transformer":
        return _build_hybrid()
    else:
        raise ValueError(f"Unknown model: {name}")


def _build_resnet50():
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    head_params = list(model.fc.parameters())
    return model, [backbone_params, head_params]


def _build_densenet121():
    model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, NUM_CLASSES)

    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    head_params = list(model.classifier.parameters())
    return model, [backbone_params, head_params]


def _build_efficientnet_b0():
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    head_params = list(model.classifier.parameters())
    return model, [backbone_params, head_params]


def _build_vit_b16():
    model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, NUM_CLASSES)

    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("heads.")]
    head_params = list(model.heads.parameters())
    return model, [backbone_params, head_params]


class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # B x 2048 x 7 x 7

        self.proj = nn.Linear(2048, d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, 50, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        features = self.backbone(x)  # B x 2048 x 7 x 7
        B, C, H, W = features.shape

        patches = features.flatten(2).transpose(1, 2)  # B x 49 x 2048
        patches = self.proj(patches)  # B x 49 x d_model

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)  # B x 50 x d_model
        tokens = tokens + self.pos_embed

        out = self.transformer(tokens)
        return self.head(out[:, 0])


def _build_hybrid():
    model = HybridCNNTransformer()

    backbone_params = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    return model, [backbone_params, head_params]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def freeze_backbone(model, model_name):
    """Freeze backbone parameters (for initial head-only training)."""
    if model_name == "ResNet-50":
        for n, p in model.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = False
    elif model_name == "DenseNet-121":
        for n, p in model.named_parameters():
            if not n.startswith("classifier."):
                p.requires_grad = False
    elif model_name == "EfficientNet-B0":
        for n, p in model.named_parameters():
            if not n.startswith("classifier."):
                p.requires_grad = False
    elif model_name == "ViT-B/16":
        for n, p in model.named_parameters():
            if not n.startswith("heads."):
                p.requires_grad = False
    elif model_name == "Hybrid CNN-Transformer":
        for n, p in model.named_parameters():
            if n.startswith("backbone."):
                p.requires_grad = False


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for p in model.parameters():
        p.requires_grad = True
