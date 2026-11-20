"""Command-line entry point. All heavy imports are delayed until needed."""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path

DEFAULT_MODEL = Path("models/freshness/best.pt")


def print_result(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Freshness NIR: обучение и локальная оценка свежести по фото.")
    p.add_argument("--debug", action="store_true", help="Показывать полный traceback.")
    commands = p.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="Проверить Python, зависимости и выход модели.")
    cache = commands.add_parser("cache-weights", help="Один раз скачать веса ImageNet.")
    cache.add_argument("--output", type=Path, default=Path("models/imagenet_mobilenet_v3_small.pth"))

    prepare = commands.add_parser("prepare", help="Разделить данные по группам на train/val/test.")
    prepare.add_argument("--source", type=Path, default=Path("data/raw"))
    prepare.add_argument("--output", type=Path, default=Path("data/prepared"))
    prepare.add_argument("--val", type=float, default=.15)
    prepare.add_argument("--test", type=float, default=.15)
    prepare.add_argument("--seed", type=int, default=42)

    audit = commands.add_parser("audit", help="Проверить данные и отсутствие точных утечек.")
    audit.add_argument("--data", type=Path, default=Path("data/prepared"))

    train = commands.add_parser("train", help="Дообучить MobileNetV3-Small на трёх классах.")
    train.add_argument("--data", type=Path, default=Path("data/prepared"))
    train.add_argument("--output", type=Path, default=Path("models/freshness"))
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lr", type=float, default=.001)
    train.add_argument("--backbone-lr", type=float, default=.0001)
    train.add_argument("--weight-decay", type=float, default=.0001)
    train.add_argument("--warmup-epochs", type=int, default=3)
    train.add_argument("--patience", type=int, default=0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    train.add_argument("--init", dest="initialization", choices=("imagenet", "none"), default="imagenet")
    train.add_argument("--initial-weights", type=Path, help="Локальный оригинальный state_dict ImageNet.")
    train.add_argument("--augmentations", choices=("albumentations", "torchvision"), default="albumentations")
    train.add_argument("--train-batchnorm", action="store_true", help="Обновлять статистики BatchNorm после прогрева.")

    predict = commands.add_parser("predict", help="Фото или папка → JSON, CSV, HTML и подписи.")
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, default=Path("outputs/predictions"))
    predict.add_argument("--labeled", action="store_true", help="Внутри input есть папки классов с ручной разметкой.")
    predict.add_argument("--limit", type=int, default=0, help="0 = все фото.")
    predict.add_argument("--threshold", type=float, default=.65)
    predict.add_argument("--threads", type=int, default=4)

    evaluate = commands.add_parser("evaluate", help="Итоговые метрики на отложенном test, CPU.")
    evaluate.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    evaluate.add_argument("--data", type=Path, default=Path("data/prepared"))
    evaluate.add_argument("--output", type=Path, default=Path("outputs/test"))
    evaluate.add_argument("--threads", type=int, default=4)
    evaluate.add_argument("--threshold", type=float, default=.65)

    benchmark = commands.add_parser("benchmark", help="Измерить скорость batch=1 на CPU.")
    benchmark.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    benchmark.add_argument("--input", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, default=Path("outputs/benchmark.json"))
    benchmark.add_argument("--threads", type=int, default=4)
    benchmark.add_argument("--repeats", type=int, default=50)
    benchmark.add_argument("--warmup", type=int, default=10)

    gui = commands.add_parser("gui", help="Открыть оконный интерфейс.")
    gui.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    label = commands.add_parser("label", help="Вручную разметить исходные фотографии.")
    label.add_argument("--input", type=Path, required=True)
    label.add_argument("--output", type=Path, default=Path("data/raw"))

    serve = commands.add_parser("serve", help="Необязательный HTTP API на 127.0.0.1.")
    serve.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--threads", type=int, default=4)

    smoke = commands.add_parser("self-test", help="Сквозной технический тест на искусственных картинках.")
    smoke.add_argument("--output", type=Path, default=Path("outputs/self_test"))
    smoke.add_argument("--augmentations", choices=("albumentations", "torchvision"), default="torchvision")
    return p


def doctor() -> int:
    result = {"python": platform.python_version(), "platform": platform.platform(), "modules": {}}
    required = ("torch", "torchvision", "numpy", "cv2", "PIL")
    failed = False
    for name in (*required, "albumentations", "tkinter"):
        try:
            module = importlib.import_module(name)
            result["modules"][name] = str(getattr(module, "__version__", "OK"))
        except Exception as exc:
            result["modules"][name] = f"ERROR: {exc}"
            if name in required:
                failed = True
    if not failed:
        from .common import configure
        from .model import build_model
        import torch
        configure(threads=2)
        with torch.inference_mode():
            model = build_model().eval()
            result["model_output_shape"] = list(model(torch.zeros(1, 3, 224, 224)).shape)
        result["cuda_available"] = torch.cuda.is_available()
        result["parameters_after_replacing_head"] = sum(p.numel() for p in model.parameters())
    result["note"] = (
        "doctor проверяет программную среду, не качество распознавания. "
        "Для стандартного train нужна Albumentations; альтернатива: --augmentations torchvision. "
        "Для GUI нужен tkinter и графический рабочий стол."
    )
    print_result(result)
    return 1 if failed else 0


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor()
        if args.command == "cache-weights":
            from .model import cache_imagenet
            cache_imagenet(args.output)
            print(f"Веса ImageNet: {args.output}")
        elif args.command == "prepare":
            from .data import prepare
            print_result(prepare(args.source, args.output, args.seed, args.val, args.test))
        elif args.command == "audit":
            from .data import audit
            print_result({k: v for k, v in audit(args.data).items() if k != "manifest"})
        elif args.command == "train":
            from .train import TrainConfig, train
            config = TrainConfig(
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                backbone_lr=args.backbone_lr, weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs, patience=args.patience,
                seed=args.seed, threads=args.threads, device=args.device,
                initialization=args.initialization,
                initial_weights=str(args.initial_weights) if args.initial_weights else None,
                augmentations=args.augmentations, freeze_batchnorm=not args.train_batchnorm,
            )
            print_result(train(args.data, args.output, config))
        elif args.command == "predict":
            from .inference import Predictor
            from .evaluate import run_batch
            predictor = Predictor(args.model, threads=args.threads, threshold=args.threshold)
            print_result(run_batch(predictor, args.input, args.output, args.labeled, args.limit))
        elif args.command == "evaluate":
            from .evaluate import evaluate
            print_result(evaluate(args.model, args.data, args.output, args.threads, args.threshold))
        elif args.command == "benchmark":
            from .evaluate import benchmark
            print_result(benchmark(args.model, args.input, args.output,
                                   args.threads, args.repeats, args.warmup))
        elif args.command == "gui":
            from .gui import launch
            launch(args.model)
        elif args.command == "label":
            from .gui import launch_labeler
            launch_labeler(args.input, args.output)
        elif args.command == "serve":
            if not 1 <= args.port <= 65535:
                raise ValueError("port должен быть от 1 до 65535.")
            from .api import create_app
            import uvicorn
            uvicorn.run(create_app(args.model, args.threads), host="127.0.0.1", port=args.port)
        elif args.command == "self-test":
            from .smoke import run_self_test
            print_result(run_self_test(args.output, args.augmentations))
        return 0
    except KeyboardInterrupt:
        print("\nОперация остановлена. Уже сохранённые результаты оставлены.", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.debug:
            raise
        print(f"Ошибка: {exc}\nПодробнее: python main.py --debug {args.command} ...",
              file=sys.stderr)
        return 1
