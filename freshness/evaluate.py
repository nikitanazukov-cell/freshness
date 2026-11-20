"""Batch predictions, independent test evaluation and repeatable CPU benchmarking."""
from __future__ import annotations

import platform
import time
from pathlib import Path

import torch

from .common import CLASSES, assert_new_directory, images_under, now_iso, write_json
from .data import audit
from .inference import Predictor, save_prediction
from .metrics import classification_metrics, timing_stats
from .reports import batch_report


def run_batch(predictor: Predictor, source: Path, output: Path, labeled: bool = False,
              limit: int = 0, annotate_limit: int = 0) -> dict:
    source, output = source.resolve(), output.resolve()
    if limit < 0 or annotate_limit < 0:
        raise ValueError("limit и annotate_limit должны быть >= 0.")
    if source == output or (source.is_dir() and source in output.parents):
        raise ValueError("Папка результатов не должна находиться внутри папки входных фото.")
    paths = images_under(source)
    if not paths:
        raise ValueError("Изображения не найдены.")
    if labeled:
        if not source.is_dir():
            raise ValueError("--labeled требует папку с подкаталогами классов.")
        for path in paths:
            if path.relative_to(source).parts[0] not in CLASSES:
                raise ValueError(f"Нет метки класса у {path}. Нужна папка FRESH/MINOR_DEFECT/SPOILED.")
        # A small demonstration should contain more than the first alphabetical class.
        buckets = [[p for p in paths if p.relative_to(source).parts[0] == c] for c in CLASSES]
        paths = [bucket[i] for i in range(max(map(len, buckets)))
                 for bucket in buckets if i < len(bucket)]
    if limit:
        paths = paths[:limit]
    output = assert_new_directory(output)
    results, errors = [], []
    for i, path in enumerate(paths, 1):
        try:
            prediction, rgb = predictor.predict_path(path)
            if labeled:
                prediction["true_class"] = path.relative_to(source).parts[0]
                prediction["correct"] = prediction["class"] == prediction["true_class"]
            save_image = not annotate_limit or len(results) < annotate_limit
            saved = save_prediction(prediction, rgb, path, output, include_image=save_image)
            results.append(saved)
        except (ValueError, OSError, RuntimeError) as exc:
            errors.append({"file": str(path), "error": str(exc)})
        if i % 20 == 0 or i == len(paths):
            print(f"Обработано {i}/{len(paths)}; ошибок {len(errors)}", flush=True)
    if not results:
        write_json(output / "errors.json", errors)
        raise ValueError(f"Ни одно изображение не обработано. См. {output / 'errors.json'}")
    metrics = None
    if labeled:
        metrics = classification_metrics(
            [CLASSES.index(r["true_class"]) for r in results],
            [CLASSES.index(r["class"]) for r in results],
        )
    summary = {
        "created_at": now_iso(), "source": str(source), "requested": len(paths),
        "processed": len(results), "errors": errors, "metrics": metrics,
        "smoke_test_model": bool(predictor.metadata.get("smoke_test", False)),
        "device": str(predictor.device), "threads": predictor.threads,
        "model_load_ms": predictor.model_load_ms,
        "warmup_ms": predictor.warmup_ms,
        "checkpoint_sha256": predictor.metadata["checkpoint_sha256"],
        "timing": {
            "model": timing_stats([r["timing"]["model_ms"] for r in results]),
            "pipeline": timing_stats([r["timing"]["pipeline_ms"] for r in results]),
        },
        "timing_scope": "Batch=1; warmed-up model; pipeline=decode+preprocess+forward+softmax; excludes saving.",
    }
    write_json(output / "predictions.json", results)
    write_json(output / "summary.json", summary)
    batch_report(output, results, summary)
    return summary


def evaluate(checkpoint: Path, data: Path, output: Path, threads: int = 4,
             threshold: float = 0.65) -> dict:
    checked = audit(data)
    predictor = Predictor(checkpoint, "cpu", threads, threshold)
    if predictor.metadata.get("dataset_fingerprint") != checked["fingerprint"]:
        raise ValueError(
            "Набор данных не совпадает с набором обучения модели. "
            "Для дополнительных независимых фото используйте predict --labeled; "
            "для итогового test используйте исходную папку prepared."
        )
    summary = run_batch(predictor, data / "test", output, labeled=True, annotate_limit=15)
    expected = sum(checked["counts"]["test"].values())
    eligible = (
        not summary["smoke_test_model"]
        and not summary["errors"]
        and summary["processed"] == expected
    )
    acc_pass = summary["metrics"]["accuracy"] >= 0.75
    speed_pass = summary["timing"]["pipeline"]["p95_ms"] <= 200
    summary["dataset_fingerprint"] = checked["fingerprint"]
    summary["acceptance"] = {
        "accuracy_at_least_75_percent": acc_pass,
        "cpu_pipeline_p95_at_most_200_ms": speed_pass,
        "eligible_real_test": eligible,
        "overall_pass": bool(acc_pass and speed_pass and eligible),
        "note": "Technical thresholds only. Smoke tests never pass research acceptance.",
    }
    summary["environment"] = environment(threads)
    write_json(output / "summary.json", summary)
    import json
    results = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    batch_report(output, results, summary)
    return summary


def environment(threads: int) -> dict:
    return {
        "platform": platform.platform(), "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torch_threads": threads, "batch_size": 1,
    }


def benchmark(checkpoint: Path, source: Path, output: Path, threads: int = 4,
              repeats: int = 50, warmup: int = 10) -> dict:
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats >= 1; warmup >= 0.")
    paths = images_under(source)
    if not paths:
        raise ValueError("Нет фото для измерения.")
    predictor = Predictor(checkpoint, "cpu", threads, warmup=warmup)
    timings = []
    for i in range(repeats):
        result, _ = predictor.predict_path(paths[i % len(paths)])
        timings.append(result["timing"])
    model_stats = timing_stats([r["model_ms"] for r in timings])
    pipeline_stats = timing_stats([r["pipeline_ms"] for r in timings])
    report = {
        "created_at": now_iso(), "environment": environment(threads),
        "model_load_ms": predictor.model_load_ms, "warmup_ms": predictor.warmup_ms,
        "warmup_runs": warmup, "repeats": repeats,
        "distinct_photos_used": min(len(paths), repeats),
        "model": model_stats, "pipeline": pipeline_stats,
        "cpu_pipeline_p95_at_most_200_ms": pipeline_stats["p95_ms"] <= 200,
        "checkpoint_sha256": predictor.metadata["checkpoint_sha256"],
        "smoke_test_model": bool(predictor.metadata.get("smoke_test", False)),
        "scope": "Single images; includes disk read/decode/preprocess/forward/softmax, excludes result saving and GUI. Repeated reads may use the OS file cache.",
    }
    write_json(output, report)
    return report
