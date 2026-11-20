"""Shared constants, deterministic setup, image IO and JSON/CSV helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

CLASSES = ("FRESH", "MINOR_DEFECT", "SPOILED")
LABELS_RU = {
    "FRESH": "Свежий",
    "MINOR_DEFECT": "Небольшой дефект",
    "SPOILED": "Испорчен",
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
SCHEMA_VERSION = 1
MAX_PIXELS = 40_000_000


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure(seed: int = 42, threads: int = 4) -> None:
    if threads < 1:
        raise ValueError("threads должен быть >= 1.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(name: str = "cpu") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA недоступна. Используйте --device cpu.")
    return torch.device(name)


def read_rgb(path: Path | str) -> np.ndarray:
    """Pillow handles Cyrillic paths and EXIF; processing is performed with OpenCV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")
    try:
        with Image.open(path) as image:
            if image.width * image.height > MAX_PIXELS:
                raise ValueError(f"Слишком большое фото: максимум {MAX_PIXELS:,} пикселей.")
            image = ImageOps.exif_transpose(image)
            # Composite transparency against a neutral background, not black.
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                rgba = image.convert("RGBA")
                bg = Image.new("RGBA", rgba.size, (114, 114, 114, 255))
                image = Image.alpha_composite(bg, rgba)
            return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Невозможно прочитать изображение {path}: {exc}") from exc


def letterbox(rgb: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize without cropping defect areas; retain the image's aspect ratio."""
    h, w = rgb.shape[:2]
    if h < 1 or w < 1:
        raise ValueError("Пустое изображение.")
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(
        rgb, (nw, nh),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    result = np.full((size, size, 3), 114, dtype=np.uint8)
    x, y = (size - nw) // 2, (size - nh) // 2
    result[y:y + nh, x:x + nw] = resized
    return result


def tensor_from_rgb(rgb: np.ndarray) -> torch.Tensor:
    array = rgb.astype(np.float32) / 255.0
    array = (array - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def image_hash(path: Path) -> str:
    """Hash decoded, oriented RGB pixels to detect exact visual duplicates."""
    rgb = read_rgb(path)
    digest = hashlib.sha256()
    digest.update(str(rgb.shape).encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def images_under(path: Path | str) -> list[Path]:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Неподдерживаемый формат: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Папка не найдена: {path}")
    return sorted(
        (p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=lambda p: p.as_posix().casefold(),
    )


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path | str, rows: Iterable[dict], fields: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 BOM for Excel on Windows; comma separator is standard CSV.
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def short_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:10]


def assert_new_directory(path: Path | str) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(
            f"Папка уже содержит файлы: {path}. Выберите новую папку; "
            "автоматического удаления данных нет."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path
