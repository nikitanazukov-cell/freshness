"""CPU-first local inference. No weights downloads or network calls here."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .common import (
    CLASSES, LABELS_RU, configure, now_iso, read_rgb, resolve_device, short_id, write_json,
)
from .data import Transform
from .model import load_checkpoint


class Predictor:
    def __init__(self, checkpoint: Path, device: str = "cpu",
                 threads: int = 4, threshold: float = 0.65, warmup: int = 3):
        if not 0 <= threshold <= 1:
            raise ValueError("Порог уверенности должен быть от 0 до 1.")
        if warmup < 0:
            raise ValueError("warmup должен быть >= 0.")
        configure(threads=threads)
        self.device = resolve_device(device)
        self.threshold, self.threads = threshold, threads
        self.checkpoint = Path(checkpoint)
        begin = time.perf_counter()
        self.model, self.metadata = load_checkpoint(self.checkpoint, self.device)
        self.model_load_ms = (time.perf_counter() - begin) * 1000
        self.transform = Transform(False)
        begin = time.perf_counter()
        with torch.inference_mode():
            for _ in range(warmup):
                self.model(torch.zeros(1, 3, 224, 224, device=self.device))
        self._sync()
        self.warmup_ms = (time.perf_counter() - begin) * 1000

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def predict_rgb(self, rgb: np.ndarray) -> dict:
        self._sync()
        started = time.perf_counter()
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        self._sync()
        prepared = time.perf_counter()
        logits = self.model(tensor)
        self._sync()
        forwarded = time.perf_counter()
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Модель вернула NaN/Inf.")
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        result = {
            "class": CLASSES[index],
            "class_ru": LABELS_RU[CLASSES[index]],
            "confidence": confidence,
            "probabilities": dict(zip(CLASSES, map(float, probabilities))),
            "needs_review": confidence < self.threshold,
            "review_threshold": self.threshold,
            "smoke_test_model": bool(self.metadata.get("smoke_test", False)),
            "created_at": now_iso(),
            "device": str(self.device),
            "checkpoint_sha256": self.metadata["checkpoint_sha256"],
            "confidence_note": "Uncalibrated softmax score; not a food safety guarantee.",
            "timing": {
                "read_ms": 0.0,
                "preprocess_ms": (prepared - started) * 1000,
                "model_ms": (forwarded - prepared) * 1000,
                "pipeline_ms": (time.perf_counter() - started) * 1000,
            },
        }
        return result

    def predict_path(self, path: Path) -> tuple[dict, np.ndarray]:
        started = time.perf_counter()
        rgb = read_rgb(path)
        read_ms = (time.perf_counter() - started) * 1000
        result = self.predict_rgb(rgb)
        result["file"] = str(path)
        result["timing"]["read_ms"] = read_ms
        result["timing"]["pipeline_ms"] = (time.perf_counter() - started) * 1000
        return result, rgb


def annotate(rgb: np.ndarray, result: dict, output: Path) -> None:
    image = Image.fromarray(rgb)
    image.thumbnail((1280, 960), Image.Resampling.LANCZOS)
    width = max(680, image.width)
    # ASCII captions work on any machine without bundling font files.
    canvas = Image.new("RGB", (width, image.height + 100), (24, 30, 38))
    canvas.paste(image, ((width - image.width) // 2, 100))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=17)
    title = f"{result['class']} | confidence: {result['confidence']:.1%}"
    draw.text((20, 16), title, fill="white", font=font)
    note = "VISUAL ESTIMATE ONLY"
    if result["smoke_test_model"]:
        note = "SMOKE TEST MODEL - NOT FOR REAL CLASSIFICATION"
    elif result["needs_review"]:
        note = "LOW CONFIDENCE - MANUAL REVIEW REQUIRED"
    draw.text((20, 58), note, fill=(255, 215, 120), font=small)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def save_prediction(result: dict, rgb: np.ndarray, source: Path, output: Path,
                    include_image: bool = True) -> dict:
    stem = source.stem[:65] + "_" + short_id(source)
    image_path = output / "annotated" / f"{stem}.jpg"
    json_path = output / "json" / f"{stem}.json"
    saved = dict(result)
    saved["annotated_image"] = str(image_path) if include_image else None
    saved["result_json"] = str(json_path)
    if include_image:
        annotate(rgb, saved, image_path)
    write_json(json_path, saved)
    return saved
