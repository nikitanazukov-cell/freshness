"""Group-aware split, leakage audit and train/evaluation datasets."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .common import (
    CLASSES, IMAGE_SIZE, assert_new_directory, image_hash, images_under,
    letterbox, now_iso, read_rgb, tensor_from_rgb, write_csv, write_json,
)

SPLITS = ("train", "val", "test")


def group_for(path: Path, class_dir: Path) -> str:
    relative = path.relative_to(class_dir)
    # All photos inside <class>/<specimen_id>/ belong to one specimen.
    if len(relative.parts) > 1:
        return relative.parts[0]
    # Alternative: apple001__front.jpg, apple001__back.jpg.
    if "__" in path.stem:
        return path.stem.split("__", 1)[0]
    # Ungrouped images are independent. Prefix labels to avoid merging IMG_001
    # from unrelated folders; exact duplicates are merged by image_hash below.
    return f"single:{class_dir.name}/{path.name}"


def scan_labeled(root: Path) -> list[dict]:
    if not root.is_dir():
        raise ValueError(f"Нет папки с данными: {root}")
    unknown = [p.name for p in root.iterdir()
               if p.is_dir() and p.name not in CLASSES and not p.name.startswith(".")]
    if unknown:
        raise ValueError(
            f"Неизвестные папки классов: {unknown}. Нужны ровно {list(CLASSES)}. "
            "Не переименовывайте fresh/rotten автоматически в три класса."
        )
    rows = []
    for label in CLASSES:
        directory = root / label
        if not directory.is_dir():
            raise ValueError(f"Нет обязательной папки класса: {directory}")
        paths = images_under(directory)
        if not paths:
            raise ValueError(f"Класс {label} пуст. Добавьте реальные размеченные фото.")
        for path in paths:
            rows.append({
                "source": path,
                "class": label,
                "group_id": group_for(path, directory),
                "sha256": image_hash(path),
            })
    return rows


def prepare(
    source: Path, output: Path, seed: int = 42,
    val_fraction: float = 0.15, test_fraction: float = 0.15,
) -> dict:
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1
            and val_fraction + test_fraction < 1):
        raise ValueError("Доли val и test должны быть > 0 и в сумме < 1.")
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Исходная и выходная папки не должны быть вложены друг в друга.")
    rows = scan_labeled(source)
    # Union groups connected by a duplicate, preventing leakage from the other
    # views of a specimen even if the duplicate was given another group name.
    parents = {r["group_id"]: r["group_id"] for r in rows}

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    seen, unique = {}, []
    duplicate_count = 0
    for row in rows:
        old = seen.get(row["sha256"])
        if old is not None:
            if old["class"] != row["class"]:
                raise ValueError(
                    "Одинаковое фото имеет разные метки: "
                    f'{old["source"]} и {row["source"]}. Исправьте разметку.'
                )
            parents[find(row["group_id"])] = find(old["group_id"])
            duplicate_count += 1
        else:
            seen[row["sha256"]] = row
            unique.append(row)
    groups = defaultdict(list)
    for row in unique:
        row["group_id"] = find(row["group_id"])
        groups[row["group_id"]].append(row)
    for label in CLASSES:
        independent = sum(any(r["class"] == label for r in group)
                          for group in groups.values())
        if independent < 3:
            raise ValueError(
                f"{label}: нужно минимум 3 независимые группы для train/val/test, "
                f"получено {independent}. Желательно существенно больше."
            )
    keys = sorted(groups)
    counts = np.array([[sum(r["class"] == label for r in groups[key])
                        for label in CLASSES] for key in keys], dtype=np.int64)
    fractions = np.array([1 - val_fraction - test_fraction,
                          val_fraction, test_fraction])
    targets = fractions[:, None] * counts.sum(axis=0)[None, :]
    rng = np.random.default_rng(seed)
    best, best_score = None, float("inf")
    # Whole specimens are assigned, not individual augmented images.
    # Choose a reproducible candidate nearest to the requested class proportions.
    homogeneous = bool(np.all((counts > 0).sum(axis=1) == 1))
    for attempt in range(1024):
        if homogeneous:
            # Stratify pure-label specimen groups, guaranteeing all three parts
            # even in the minimum case of just 3 independent groups per class.
            assignment = np.zeros(len(keys), dtype=np.int64)
            for class_index in range(len(CLASSES)):
                indices = np.flatnonzero(counts[:, class_index] > 0)
                indices = rng.permutation(indices)
                n_val = max(1, round(len(indices) * val_fraction))
                n_test = max(1, round(len(indices) * test_fraction))
                while n_val + n_test >= len(indices):
                    if n_val >= n_test and n_val > 1:
                        n_val -= 1
                    elif n_test > 1:
                        n_test -= 1
                    else:
                        break
                assignment[indices[:n_val]] = 1
                assignment[indices[n_val:n_val + n_test]] = 2
        else:
            # A specimen may appear under different labels on different days.
            # Never separate these stages into different subsets.
            probabilities = (np.ones(3) / 3
                             if len(keys) < 30 and attempt < 512 else fractions)
            assignment = rng.choice(3, len(keys), p=probabilities)
        actual = np.array([counts[assignment == i].sum(axis=0)
                           for i in range(3)])
        if np.any(actual == 0):
            continue
        score = float(np.mean(((actual - targets) / np.maximum(targets, 1)) ** 2))
        if score < best_score:
            best, best_score = assignment.copy(), score
    if best is None:
        raise ValueError(
            "Не удалось разделить группы с тремя классами в каждой части. "
            "Добавьте независимые образцы, особенно редкого класса."
        )
    output = assert_new_directory(output)
    for split in SPLITS:
        for label in CLASSES:
            (output / split / label).mkdir(parents=True, exist_ok=True)
    manifest = []
    for key, subset in zip(keys, best.tolist()):
        split = SPLITS[subset]
        for row in groups[key]:
            src = row["source"]
            relative = Path(split) / row["class"] / (row["sha256"] + src.suffix.lower())
            shutil.copy2(src, output / relative)
            manifest.append({
                "path": relative.as_posix(), "split": split, "class": row["class"],
                "group_id": key, "sha256": row["sha256"],
                "source_path": src.relative_to(source).as_posix(),
            })
    manifest.sort(key=lambda row: row["path"])
    write_csv(output / "manifest.csv", manifest,
              ["path", "split", "class", "group_id", "sha256", "source_path"])
    summary = {
        "created_at": now_iso(), "seed": seed,
        "requested_fractions": dict(zip(SPLITS, fractions.tolist())),
        "counts": {split: dict(Counter(r["class"] for r in manifest if r["split"] == split))
                   for split in SPLITS},
        "groups": len(groups), "unique_images": len(unique),
        "removed_exact_duplicates": duplicate_count,
        "note": "Exact duplicates detected. Near-duplicates require correct specimen grouping.",
    }
    write_json(output / "split_summary.json", summary)
    return summary


def audit(data: Path) -> dict:
    """Reject missing labels, changed manifests and cross-split duplicate/group leakage."""
    data = data.resolve()
    manifest_path = data / "manifest.csv"
    if not manifest_path.is_file():
        raise ValueError(
            f"Нет {manifest_path}. Сначала выполните prepare. "
            "Обучение использует только подготовленный и проверяемый набор."
        )
    with manifest_path.open(encoding="utf-8-sig", newline="") as file:
        manifest = list(csv.DictReader(file))
    if not manifest:
        raise ValueError("Пустой manifest.csv.")
    by_hash, by_group, counts = {}, {}, Counter()
    expected = set()
    for row in manifest:
        rel = Path(row["path"])
        path = (data / rel).resolve()
        if data not in path.parents or len(rel.parts) != 3:
            raise ValueError("Некорректный или небезопасный путь в manifest.csv.")
        if row["split"] not in SPLITS or row["class"] not in CLASSES:
            raise ValueError("Некорректные split/class в manifest.csv.")
        if rel.parts[:2] != (row["split"], row["class"]):
            raise ValueError(f"Путь не соответствует разметке: {rel}")
        actual_hash = image_hash(path)
        if actual_hash != row["sha256"]:
            raise ValueError(f"Фото изменено после разделения: {rel}. Повторите prepare.")
        split = row["split"]
        for mapping, key in ((by_hash, actual_hash), (by_group, row["group_id"])):
            if key in mapping and mapping[key] != split:
                raise ValueError(f"Утечка между выборками: {rel}")
            mapping[key] = split
        if rel.as_posix() in expected:
            raise ValueError(f"Дублированная строка manifest: {rel}")
        expected.add(rel.as_posix())
        counts[(split, row["class"])] += 1
    actual = set()
    for split in SPLITS:
        actual.update(p.relative_to(data).as_posix() for p in images_under(data / split))
        for label in CLASSES:
            paths = images_under(data / split / label)
            if not paths:
                raise ValueError(f"Пустой класс: {split}/{label}")
            actual.update(p.relative_to(data).as_posix() for p in paths)
    if actual != expected:
        raise ValueError("Файлы не совпадают с manifest.csv. Повторите prepare в новую папку.")
    fingerprint = hashlib.sha256(
        json.dumps(sorted((r["path"], r["sha256"], r["class"], r["group_id"])
                          for r in manifest), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        "fingerprint": fingerprint,
        "counts": {s: {c: counts[(s, c)] for c in CLASSES} for s in SPLITS},
        "images": len(manifest),
        "groups": len(by_group),
        "manifest": manifest,
    }


class Transform:
    """Identical normalization/letterboxing for train, validation and inference."""
    def __init__(self, train: bool = False, backend: str = "albumentations", seed: int = 42):
        self.train = train
        self.backend = backend
        self.augment = None
        if not train:
            return
        if backend == "albumentations":
            try:
                import albumentations as A
            except ImportError as exc:
                raise RuntimeError(
                    "Albumentations не установлена. Установите requirements.txt "
                    "или явно укажите --augmentations torchvision."
                ) from exc
            self.augment = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.12, contrast_limit=0.12, p=0.4),
                A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            ], seed=seed)
        elif backend == "torchvision":
            self.augment = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomApply([T.RandomRotation(15, fill=(114, 114, 114))], p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.12, contrast=0.12)], p=0.4),
                T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.1),
            ])
        else:
            raise ValueError(f"Неизвестный движок аугментаций: {backend}")

    def __call__(self, rgb: np.ndarray) -> torch.Tensor:
        rgb = letterbox(rgb, IMAGE_SIZE)
        if self.train:
            if self.backend == "albumentations":
                rgb = self.augment(image=rgb)["image"]
            else:
                rgb = np.asarray(self.augment(Image.fromarray(rgb)))
        return tensor_from_rgb(rgb)


class ProduceDataset(Dataset):
    def __init__(self, data: Path, split: str, train: bool = False,
                 backend: str = "albumentations", seed: int = 42):
        self.samples = [(path, CLASSES.index(label))
                        for label in CLASSES for path in images_under(data / split / label)]
        self.transform = Transform(train=train, backend=backend, seed=seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(read_rgb(path)), label
