"""Synthetic integration test. This is NOT a trained produce classifier."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import CLASSES, assert_new_directory, now_iso, write_json
from .data import prepare
from .evaluate import benchmark, evaluate
from .inference import Predictor
from .train import TrainConfig, train


def make_synthetic_data(output: Path, groups_per_class: int = 6, views: int = 2):
    rng = np.random.default_rng(1234)
    for class_id, label in enumerate(CLASSES):
        for group in range(groups_per_class):
            for view in range(views):
                array = rng.integers(0, 45, (128, 176, 3), dtype=np.uint8)
                array[:, :, class_id] += 130
                image = Image.fromarray(array)
                draw = ImageDraw.Draw(image)
                x = 20 + group * 5 + view * 3
                draw.ellipse((x, 20, x + 65, 95), outline="white", width=3)
                draw.text((8, 105), "SYNTHETIC TEST", fill="white")
                path = output / label / f"{label.lower()}_{group:03d}__view_{view}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)


def run_self_test(output: Path, augmentations: str = "torchvision") -> dict:
    output = assert_new_directory(output)
    print("SMOKE TEST: проверка программы, не точности на фруктах и овощах.", flush=True)
    raw, prepared, trained = output / "raw", output / "prepared", output / "model"
    make_synthetic_data(raw)
    split = prepare(raw, prepared)
    training = train(prepared, trained, TrainConfig(
        epochs=2, batch_size=4, warmup_epochs=1, threads=2,
        initialization="none", augmentations=augmentations, smoke_test=True,
    ))
    checkpoint = trained / "best.pt"
    test = evaluate(checkpoint, prepared, output / "evaluation", threads=2)
    speed = benchmark(checkpoint, prepared / "test", output / "benchmark.json",
                      threads=2, repeats=12, warmup=3)
    predictor = Predictor(checkpoint, threads=2)
    example = next((prepared / "test" / "FRESH").glob("*.png"))
    prediction, _ = predictor.predict_path(example)
    assert len(prediction["probabilities"]) == 3
    assert abs(sum(prediction["probabilities"].values()) - 1.0) < 1e-5
    assert 0 <= prediction["confidence"] <= 1
    assert prediction["smoke_test_model"] is True
    assert test["acceptance"]["overall_pass"] is False
    required_files = [
        trained / "best.pt", trained / "training.html",
        output / "evaluation/predictions.json", output / "evaluation/predictions.csv",
        output / "evaluation/confusion_matrix.csv", output / "evaluation/report.html",
        output / "benchmark.json",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required_files)
    assert list((output / "evaluation/annotated").glob("*.jpg"))
    report = {
        "created_at": now_iso(), "status": "PASSED",
        "scope": "Synthetic integration test only; not evidence of freshness classification accuracy.",
        "augmentations": augmentations, "split": split,
        "train_epochs": training["epochs_completed"],
        "checkpoint_bytes": training["checkpoint_bytes"],
        "pipeline_mean_ms": speed["pipeline"]["mean_ms"],
        "pipeline_p95_ms": speed["pipeline"]["p95_ms"],
        "research_acceptance_forced_false": not test["acceptance"]["overall_pass"],
        "artifacts": [str(path) for path in required_files],
    }
    write_json(output / "self_test_result.json", report)
    return report
