"""Transfer learning: warm-up classifier, then fine-tune the whole backbone."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from .common import CLASSES, assert_new_directory, configure, now_iso, resolve_device, write_csv, write_json
from .data import ProduceDataset, audit
from .metrics import classification_metrics
from .model import build_model, save_checkpoint


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 16
    lr: float = 0.001
    backbone_lr: float = 0.0001
    weight_decay: float = 0.0001
    warmup_epochs: int = 3
    patience: int = 0  # 0: run all 25 epochs, as requested by the assignment.
    seed: int = 42
    threads: int = 4
    device: str = "cpu"
    initialization: str = "imagenet"
    initial_weights: str | None = None
    augmentations: str = "albumentations"
    freeze_batchnorm: bool = True
    smoke_test: bool = False


def train(data: Path, output: Path, config: TrainConfig) -> dict:
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs и batch_size должны быть >= 1.")
    if config.lr <= 0 or config.backbone_lr <= 0 or config.weight_decay < 0:
        raise ValueError("Некорректные параметры оптимизатора.")
    if config.warmup_epochs < 0 or config.patience < 0:
        raise ValueError("warmup_epochs и patience должны быть >= 0.")
    configure(config.seed, config.threads)
    device = resolve_device(config.device)
    checked = audit(data)
    if min(checked["counts"]["train"].values()) < 10:
        print("ВНИМАНИЕ: в train меньше 10 фото хотя бы одного класса. "
              "Это недостаточно для убедительной оценки качества.", flush=True)
    if config.initialization == "none" and not config.initial_weights:
        print("ВНИМАНИЕ: случайная инициализация. Это не transfer learning из ImageNet.",
              flush=True)
    if config.smoke_test:
        print("SMOKE TEST: искусственные изображения; результаты не подтверждают НИР.",
              flush=True)

    training = ProduceDataset(data, "train", True, config.augmentations, config.seed)
    validation = ProduceDataset(data, "val", False)
    # num_workers=0 avoids multiprocessing/spawn problems on Windows and
    # makes the single Albumentations random generator reproducible.
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(training, batch_size=config.batch_size, shuffle=True,
                        num_workers=0, generator=generator)
    val_loader = DataLoader(validation, batch_size=config.batch_size,
                            shuffle=False, num_workers=0)
    model = build_model(config.initialization,
                        Path(config.initial_weights) if config.initial_weights else None).to(device)
    output = assert_new_directory(output)
    write_json(output / "config.json", asdict(config))
    write_json(output / "dataset_audit.json",
               {key: value for key, value in checked.items() if key != "manifest"})
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": config.backbone_lr},
        {"params": model.classifier.parameters(), "lr": config.lr},
    ], weight_decay=config.weight_decay)
    weights = torch.tensor(
        [len(training) / (len(CLASSES) * checked["counts"]["train"][c]) for c in CLASSES],
        dtype=torch.float32, device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    def run_epoch(batches, training_mode):
        total_loss, loss_weight = 0.0, 0.0
        truth, guessed = [], []
        with torch.set_grad_enabled(training_mode):
            for images, targets in batches:
                images, targets = images.to(device), targets.to(device)
                if training_mode:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Получен NaN/Inf loss. Проверьте фото и learning rate.")
                if training_mode:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
                    optimizer.step()
                denominator = float(weights[targets].sum().item())
                total_loss += loss.item() * denominator
                loss_weight += denominator
                truth.extend(targets.detach().cpu().tolist())
                guessed.extend(logits.argmax(1).detach().cpu().tolist())
        return total_loss / loss_weight, classification_metrics(truth, guessed)

    history, best_key, no_improvement = [], None, 0
    best_summary = {}
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        warmup = epoch <= config.warmup_epochs
        for parameter in model.features.parameters():
            parameter.requires_grad_(not warmup)
        model.train()
        if warmup:
            model.features.eval()
        elif config.freeze_batchnorm:
            # More stable when CPU memory limits us to small minibatches.
            for layer in model.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
        train_loss, train_metrics = run_epoch(loader, True)
        model.eval()
        val_loss, val_metrics = run_epoch(val_loader, False)
        row = {
            "epoch": epoch, "stage": "head" if warmup else "fine_tune",
            "train_loss": train_loss, "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        # Only validation metrics are used for model selection. Test is untouched.
        key = (val_metrics["macro_f1"], val_metrics["accuracy"], -val_loss)
        if best_key is None or key > best_key:
            best_key, no_improvement = key, 0
            best_summary = {
                "created_at": now_iso(), "best_epoch": epoch,
                "validation_metrics": val_metrics, "validation_loss": val_loss,
                "dataset_fingerprint": checked["fingerprint"],
                "training_config": asdict(config), "smoke_test": config.smoke_test,
                "parameters": sum(p.numel() for p in model.parameters()),
                "versions": {"torch": str(torch.__version__),
                             "torchvision": str(torchvision.__version__)},
            }
            save_checkpoint(output / "best.pt", model, best_summary)
        else:
            no_improvement += 1
        write_csv(output / "history.csv", history, list(row))
        write_json(output / "history.json", history)
        print(
            f"Epoch {epoch:02d}/{config.epochs} | {row['stage']} | "
            f"loss {train_loss:.4f}/{val_loss:.4f} | "
            f"accuracy {train_metrics['accuracy']:.3f}/{val_metrics['accuracy']:.3f} | "
            f"val F1 {val_metrics['macro_f1']:.3f} | {row['epoch_seconds']:.1f} s",
            flush=True,
        )
        factor = 0.1 + 0.9 * (1 + math.cos(math.pi * epoch / config.epochs)) / 2
        optimizer.param_groups[0]["lr"] = config.backbone_lr * factor
        optimizer.param_groups[1]["lr"] = config.lr * factor
        if config.patience and not warmup and no_improvement >= config.patience:
            print("Early stopping: улучшений на validation больше нет.", flush=True)
            break
    best_summary["checkpoint"] = str(output / "best.pt")
    best_summary["checkpoint_bytes"] = (output / "best.pt").stat().st_size
    best_summary["epochs_completed"] = len(history)
    write_json(output / "training_summary.json", best_summary)
    from .reports import training_report
    training_report(output, history, best_summary)
    return best_summary
