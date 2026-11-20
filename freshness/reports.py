"""Portable offline HTML reports; no CDN, JavaScript, or external fonts."""
from __future__ import annotations

from html import escape
from pathlib import Path
from urllib.parse import quote

from .common import CLASSES, write_csv


STYLE = """
body{font:15px/1.6 system-ui,Arial,sans-serif;margin:0;background:#f2f5f8;color:#172638}
main{max-width:1250px;margin:32px auto;padding:28px;background:white;border-radius:12px}
h1{line-height:1.2;margin:0 0 14px}h2{margin-top:32px}
table{border-collapse:collapse;width:100%;margin:16px 0;font-variant-numeric:tabular-nums}
td,th{text-align:left;padding:10px;border-bottom:1px solid #dce3ea;vertical-align:top}
th{background:#eef3f8}img{width:190px;height:auto;border-radius:6px}
.notice{border-left:4px solid #b76c15;background:#fff7e8;padding:12px 16px}
.good{color:#176936}.bad{color:#ad2626}.muted{color:#637488}.scroll{overflow:auto}
code{background:#eef3f8;padding:2px 5px}small{color:#637488}
"""
NOTICE = (
    "Оцениваются только видимые признаки. Softmax confidence не является гарантией "
    "безопасности продукта или достоверной вероятностью правильного ответа."
)


def page(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><style>{STYLE}</style></head>"
        f"<body><main><h1>{escape(title)}</h1>{body}</main></body></html>"
    )


def metrics_html(metrics: dict) -> str:
    lower, upper = metrics["accuracy_wilson_95"]
    text = (
        f'<p><b>Accuracy: {metrics["accuracy"]:.2%}</b> · '
        f'Macro-F1: {metrics["macro_f1"]:.4f} · '
        f'Правильно: {metrics["correct"]}/{metrics["n"]}</p>'
        f'<p class="muted">95% интервал Уилсона для accuracy: '
        f'{lower:.1%}–{upper:.1%}. Не учитывает ошибки разметки.</p>'
    )
    text += "<table><tr><th>Класс</th><th>Precision</th><th>Recall</th><th>F1</th><th>Фото</th></tr>"
    for label, values in metrics["per_class"].items():
        text += (
            f"<tr><td>{escape(label)}</td><td>{values['precision']:.4f}</td>"
            f"<td>{values['recall']:.4f}</td><td>{values['f1']:.4f}</td>"
            f"<td>{values['support']}</td></tr>"
        )
    text += "</table><h2>Матрица ошибок</h2><p>Строка — истинный класс; столбец — предсказание.</p><table><tr><th>Истинный / прогноз</th>"
    text += "".join(f"<th>{label}</th>" for label in CLASSES) + "</tr>"
    for label, counts in zip(CLASSES, metrics["confusion_matrix"]):
        text += f"<tr><th>{label}</th>" + "".join(f"<td>{n}</td>" for n in counts) + "</tr>"
    return text + "</table>"


def batch_report(output: Path, results: list[dict], summary: dict) -> None:
    rows = []
    for r in results:
        rows.append({
            "photo": r["file"], "true_class": r.get("true_class", ""),
            "predicted_class": r["class"], "confidence": r["confidence"],
            "correct": r.get("correct", ""),
            "needs_review": r["needs_review"],
            "model_ms": r["timing"]["model_ms"],
            "pipeline_ms": r["timing"]["pipeline_ms"],
            **{f"p_{c}": r["probabilities"][c] for c in CLASSES},
            "annotated_image": r.get("annotated_image", ""),
            "result_json": r.get("result_json", ""),
            "smoke_test_model": r["smoke_test_model"],
        })
    write_csv(output / "predictions.csv", rows, list(rows[0]))
    body = f'<p class="notice">{NOTICE}</p>'
    if summary.get("smoke_test_model"):
        body += '<p class="notice"><b>ТЕХНИЧЕСКИЙ ТЕСТ НА ИСКУССТВЕННЫХ ДАННЫХ. Не результат исследования.</b></p>'
    body += (
        f'<p>Успешно обработано: {len(results)}. Ошибок чтения/обработки: '
        f'{len(summary.get("errors", []))}.</p>'
        '<p>Время pipeline включает чтение фото, подготовку, модель и получение ответа. '
        'Загрузка модели, прогрев и сохранение результатов считаются отдельно.</p>'
    )
    if summary.get("metrics"):
        body += metrics_html(summary["metrics"])
        matrix = [{"true_class": label, **dict(zip(CLASSES, counts))}
                  for label, counts in zip(CLASSES, summary["metrics"]["confusion_matrix"])]
        write_csv(output / "confusion_matrix.csv", matrix, ["true_class", *CLASSES])
    timing = summary.get("timing", {}).get("pipeline", {})
    if timing:
        body += (
            f'<p><b>Pipeline:</b> среднее {timing["mean_ms"]:.2f} мс; '
            f'медиана {timing["median_ms"]:.2f} мс; P95 {timing["p95_ms"]:.2f} мс.</p>'
        )
    acceptance = summary.get("acceptance")
    if acceptance:
        status = "Критерии выполнены" if acceptance["overall_pass"] else "Критерии пока не подтверждены"
        body += (
            f'<h2>{status}</h2><p>Accuracy ≥ 75%: '
            f'{acceptance["accuracy_at_least_75_percent"]}; '
            f'P95 pipeline на CPU ≤ 200 мс: {acceptance["cpu_pipeline_p95_at_most_200_ms"]}; '
            f'реальный, полный тест: {acceptance["eligible_real_test"]}.</p>'
        )
    body += '<h2>Результаты по фотографиям</h2><div class="scroll"><table><tr><th>Фото</th><th>Файл / истинный класс</th><th>Прогноз</th><th>Confidence</th><th>Проверка</th><th>Pipeline</th></tr>'
    for r in results:
        image = r.get("annotated_image")
        image_cell = "—"
        if image:
            relative = Path(image).resolve().relative_to(output.resolve()).as_posix()
            image_cell = f'<img loading="lazy" src="{escape(quote(relative))}" alt="Фото с подписью">'
        correct = r.get("correct")
        mark = "Не размечено" if correct is None else ("Правильно" if correct else "Ошибка")
        css = "" if correct is None else ("good" if correct else "bad")
        review = "<br><small>Нужна ручная проверка</small>" if r["needs_review"] else ""
        body += (
            f"<tr><td>{image_cell}</td><td>{escape(Path(r['file']).name)}<br>"
            f"<small>{escape(r.get('true_class') or 'Нет разметки')}</small></td>"
            f"<td>{escape(r['class'])}{review}</td><td>{r['confidence']:.2%}</td>"
            f'<td class="{css}">{mark}</td><td>{r["timing"]["pipeline_ms"]:.2f} мс</td></tr>'
        )
    body += "</table></div>"
    if summary.get("errors"):
        body += "<h2>Ошибки обработки</h2><pre>" + escape(
            "\n".join(f"{e['file']}: {e['error']}" for e in summary["errors"])
        ) + "</pre>"
    (output / "report.html").write_text(page("Оценка свежести по фото", body), encoding="utf-8")


def training_report(output: Path, history: list[dict], summary: dict) -> None:
    body = (
        "<p>Лучшая модель выбрана только по validation: сначала macro-F1, "
        "затем accuracy, затем loss. Test не использовался для выбора.</p>"
        f"<p>Лучшая эпоха: <b>{summary['best_epoch']}</b>. "
        f"Файл модели: <code>best.pt</code>; размер: "
        f"{summary['checkpoint_bytes'] / 1024**2:.2f} MiB.</p>"
    )
    if summary.get("smoke_test"):
        body += '<p class="notice">SMOKE TEST — искусственные изображения, не оценка качества на продуктах.</p>'
    body += metrics_html(summary["validation_metrics"])
    # Inline SVG curves keep the training report completely offline.
    body += '<h2>Accuracy по эпохам</h2><svg viewBox="0 0 800 240" role="img" aria-label="Accuracy train и validation" style="width:100%;max-width:800px">'
    body += '<rect x="0" y="0" width="800" height="240" fill="white"/>'
    for value in (0, .25, .5, .75, 1):
        y = 205 - value * 175
        body += f'<line x1="50" y1="{y}" x2="770" y2="{y}" stroke="#dce3ea"/><text x="7" y="{y+4}" font-size="12">{value:.2f}</text>'
    for key, color in (("train_accuracy", "#3465b4"), ("val_accuracy", "#c36a17")):
        points = " ".join(f"{50+i*720/max(1,len(history)-1):.2f},{205-r[key]*175:.2f}"
                          for i, r in enumerate(history))
        body += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
    body += '<text x="50" y="230" font-size="13">Синий: train · оранжевый: validation · ось X: эпоха</text></svg>'
    body += '<h2>История обучения</h2><table><tr><th>Эпоха</th><th>Этап</th><th>Train loss</th><th>Val loss</th><th>Train accuracy</th><th>Val accuracy</th><th>Val F1</th></tr>'
    for row in history:
        body += (
            f"<tr><td>{row['epoch']}</td><td>{row['stage']}</td>"
            f"<td>{row['train_loss']:.4f}</td><td>{row['val_loss']:.4f}</td>"
            f"<td>{row['train_accuracy']:.2%}</td><td>{row['val_accuracy']:.2%}</td>"
            f"<td>{row['val_macro_f1']:.4f}</td></tr>"
        )
    body += "</table>"
    (output / "training.html").write_text(page("Обучение MobileNetV3-Small", body), encoding="utf-8")
