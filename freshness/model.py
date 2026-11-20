"""MobileNetV3-Small construction and safe, self-describing checkpoints."""
from __future__ import annotations
import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .common import CLASSES, IMAGE_SIZE, MEAN, STD, SCHEMA_VERSION


def build_model(initialization: str = "none",
                initial_weights: Path | None = None) -> nn.Module:
    if initial_weights is not None:
        model = mobilenet_v3_small(weights=None)
        state = torch.load(initial_weights, map_location="cpu", weights_only=True)
        # Accept an original ImageNet state_dict, not a trained project checkpoint.
        model.load_state_dict(state, strict=True)
    elif initialization == "imagenet":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    elif initialization == "none":
        model = mobilenet_v3_small(weights=None)
    else:
        raise ValueError("initialization: imagenet или none.")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    return model


def save_checkpoint(path: Path, model: nn.Module, metadata: dict) -> None:
    content = {
        **metadata,
        "schema_version": SCHEMA_VERSION, "architecture": "mobilenet_v3_small",
        "classes": list(CLASSES),
        "preprocessing": {"image_size": IMAGE_SIZE, "resize": "letterbox",
                          "mean": list(MEAN), "std": list(STD)},
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(content, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(
            f"Нет обученной модели: {path}. Сначала подготовьте данные и выполните train. "
            "Обычные веса ImageNet не являются моделью свежести."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Некорректный формат модели.")
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Неизвестная версия формата модели.")
    if checkpoint.get("architecture") != "mobilenet_v3_small":
        raise ValueError("Неподдерживаемая архитектура.")
    if tuple(checkpoint.get("classes", [])) != CLASSES:
        raise ValueError("Порядок классов в модели не соответствует проекту.")
    expected = {"image_size": IMAGE_SIZE, "resize": "letterbox",
                "mean": list(MEAN), "std": list(STD)}
    if checkpoint.get("preprocessing") != expected:
        raise ValueError("Настройки обработки фото в модели не поддерживаются.")
    model = build_model()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    metadata["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return model, metadata


def cache_imagenet(output: Path) -> None:
    state = MobileNet_V3_Small_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
