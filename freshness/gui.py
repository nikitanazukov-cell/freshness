"""Offline desktop prediction and a small manual labeling utility."""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import threading
import tkinter as tk
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .common import CLASSES, LABELS_RU, image_hash, images_under, read_rgb, short_id

IMAGE_TYPES = [("Изображения", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
               ("Все файлы", "*.*")]


class FreshnessApp(tk.Tk):
    def __init__(self, checkpoint: Path):
        super().__init__()
        self.title("Freshness NIR — свежесть продуктов")
        self.geometry("1120x800")
        self.minsize(850, 650)
        self.messages = queue.Queue()
        self.busy = False
        self.predictor = None
        self.predictor_key = None
        self.last_output = None
        self.preview_image = None
        self.model_path = tk.StringVar(value=str(checkpoint))
        self.photo_path = tk.StringVar()
        self.status = tk.StringVar(value="Выберите обученную модель и фотографию.")
        self.result_title = tk.StringVar(value="Результат ещё не получен")
        self.batch_labeled = tk.BooleanVar(value=False)

        outer = ttk.Frame(self, padding=20)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Оценка свежести по фотографии",
                  font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(outer, text="MobileNetV3-Small · локальная обработка на CPU").pack(anchor="w", pady=(2, 16))
        model_row = ttk.Frame(outer)
        model_row.pack(fill="x", pady=3)
        ttk.Label(model_row, text="Модель .pt:", width=15).pack(side="left")
        ttk.Entry(model_row, textvariable=self.model_path).pack(side="left", fill="x", expand=True)
        ttk.Button(model_row, text="Выбрать", command=self.choose_model).pack(side="left", padx=(8, 0))
        photo_row = ttk.Frame(outer)
        photo_row.pack(fill="x", pady=3)
        ttk.Label(photo_row, text="Фотография:", width=15).pack(side="left")
        ttk.Entry(photo_row, textvariable=self.photo_path).pack(side="left", fill="x", expand=True)
        ttk.Button(photo_row, text="Выбрать", command=self.choose_photo).pack(side="left", padx=(8, 0))
        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=12)
        self.single_button = ttk.Button(actions, text="Распознать фото", command=self.predict_single)
        self.single_button.pack(side="left")
        self.batch_button = ttk.Button(actions, text="Обработать папку", command=self.predict_folder)
        self.batch_button.pack(side="left", padx=8)
        ttk.Checkbutton(actions, text="Папка размечена по классам",
                        variable=self.batch_labeled).pack(side="left")
        ttk.Button(actions, text="Открыть результат", command=self.open_output).pack(side="right")
        pane = ttk.Panedwindow(outer, orient="horizontal")
        pane.pack(fill="both", expand=True)
        left, right = ttk.Frame(pane), ttk.Frame(pane, padding=(15, 0, 0, 0))
        pane.add(left, weight=3)
        pane.add(right, weight=2)
        self.preview = ttk.Label(left, text="Здесь появится фотография", anchor="center")
        self.preview.pack(fill="both", expand=True)
        ttk.Label(right, textvariable=self.result_title, font=("Segoe UI", 14, "bold"),
                  wraplength=370).pack(anchor="w", pady=(0, 10))
        self.details = tk.Text(right, width=43, height=17, wrap="word", font=("Consolas", 10))
        self.details.pack(fill="both", expand=True)
        self.details.insert("1.0", "Без файла обученной модели распознавание недоступно.\n\n"
                            "Порядок работы:\n1. Подготовьте фото трёх классов.\n"
                            "2. Выполните prepare и train.\n3. Выберите models/freshness/best.pt.\n\n"
                            "Команды находятся в README.md.")
        self.details.config(state="disabled")
        self.progress = ttk.Progressbar(outer, mode="indeterminate")
        self.progress.pack(fill="x", pady=(14, 5))
        ttk.Label(outer, textvariable=self.status, wraplength=1020).pack(anchor="w")
        ttk.Label(outer, text="Оцениваются только видимые признаки. Результат не гарантирует безопасность продукта.",
                  wraplength=1020).pack(anchor="w", pady=(10, 0))
        self._poll_id = self.after(100, self.poll)

    def destroy(self):
        # Cancel scheduled callbacks before destroying Tcl widgets.
        callback = getattr(self, "_poll_id", None)
        if callback is not None:
            try:
                self.after_cancel(callback)
            except tk.TclError:
                pass
        super().destroy()

    def choose_model(self):
        value = filedialog.askopenfilename(filetypes=[("Модель проекта", "*.pt")])
        if value:
            self.model_path.set(value)

    def choose_photo(self):
        value = filedialog.askopenfilename(filetypes=IMAGE_TYPES)
        if value:
            self.photo_path.set(value)
            try:
                self.show_image(Path(value))
            except (ValueError, OSError) as exc:
                messagebox.showerror("Ошибка", str(exc))

    def show_image(self, path: Path):
        image = Image.fromarray(read_rgb(path))
        image.thumbnail((600, 460))
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview.config(image=self.preview_image, text="")

    def get_predictor(self, checkpoint: Path):
        from .inference import Predictor
        key = (str(checkpoint.resolve()),
               checkpoint.stat().st_mtime_ns if checkpoint.exists() else None)
        if self.predictor is None or self.predictor_key != key:
            self.predictor = Predictor(checkpoint)
            self.predictor_key = key
        return self.predictor

    def start_work(self, function):
        if self.busy:
            return
        self.busy = True
        self.single_button.config(state="disabled")
        self.batch_button.config(state="disabled")
        self.progress.start(12)
        self.status.set("Выполняется обработка. Изображения не отправляются в интернет.")
        def worker():
            try:
                self.messages.put(("ok", function()))
            except Exception as exc:
                self.messages.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def predict_single(self):
        path = Path(self.photo_path.get())
        if not path.is_file():
            messagebox.showwarning("Выберите фото", "Укажите существующий файл изображения.")
            return
        checkpoint = Path(self.model_path.get())
        output = self.new_output()
        def work():
            from .evaluate import run_batch
            summary = run_batch(self.get_predictor(checkpoint), path, output)
            result = json.loads((output / "predictions.json").read_text(encoding="utf-8"))[0]
            return output, summary, result
        self.start_work(work)

    def predict_folder(self):
        value = filedialog.askdirectory(title="Папка с фотографиями")
        if not value:
            return
        path, checkpoint, labeled = Path(value), Path(self.model_path.get()), self.batch_labeled.get()
        output = self.new_output()
        def work():
            from .evaluate import run_batch
            summary = run_batch(self.get_predictor(checkpoint), path, output, labeled=labeled)
            return output, summary, None
        self.start_work(work)

    def new_output(self) -> Path:
        return Path("outputs/gui") / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])

    def poll(self):
        try:
            kind, payload = self.messages.get_nowait()
        except queue.Empty:
            self._poll_id = self.after(100, self.poll)
            return
        self.busy = False
        self.single_button.config(state="normal")
        self.batch_button.config(state="normal")
        self.progress.stop()
        if kind == "error":
            self.status.set("Ошибка. Проверьте путь к модели, фото и установленные библиотеки.")
            messagebox.showerror("Ошибка обработки", payload)
        else:
            output, summary, result = payload
            self.last_output = output
            if result:
                title = f"{result['class_ru']}\nУверенность: {result['confidence']:.1%}"
                if result["smoke_test_model"]:
                    title = "ТЕХНИЧЕСКИЙ ТЕСТ\n" + title
                elif result["needs_review"]:
                    title += "\nНужна ручная проверка"
                self.result_title.set(title)
                self.show_image(Path(result["annotated_image"]))
            else:
                self.result_title.set(f"Обработано фото: {summary['processed']}")
            self.details.config(state="normal")
            self.details.delete("1.0", "end")
            self.details.insert("1.0", json.dumps(result or summary, ensure_ascii=False, indent=2))
            self.details.config(state="disabled")
            self.status.set(f"Сохранено: {output.resolve()}. Кнопка «Открыть результат» откроет HTML-отчёт.")
        self._poll_id = self.after(100, self.poll)

    def open_output(self):
        if self.last_output is None:
            messagebox.showinfo("Результат", "Сначала обработайте фото или папку.")
        else:
            webbrowser.open((self.last_output / "report.html").resolve().as_uri())


class LabelApp(tk.Tk):
    def __init__(self, source: Path, output: Path):
        super().__init__()
        self.title("Разметка фото — FRESH / MINOR_DEFECT / SPOILED")
        self.geometry("950x720")
        self.paths = images_under(source)
        if not self.paths:
            self.destroy()
            raise ValueError("Нет изображений для разметки.")
        if source.resolve() == output.resolve() or source.resolve() in output.resolve().parents:
            self.destroy()
            raise ValueError("Папка разметки должна быть отдельно от исходных изображений.")
        self.output, self.index = output, 0
        self.status = tk.StringVar()
        self.group = tk.StringVar()
        body = ttk.Frame(self, padding=20)
        body.pack(fill="both", expand=True)
        ttk.Label(body, textvariable=self.status, wraplength=900).pack(anchor="w")
        self.preview = ttk.Label(body, anchor="center")
        self.preview.pack(fill="both", expand=True, pady=10)
        row = ttk.Frame(body)
        row.pack(fill="x")
        ttk.Label(row, text="ID плода/серии:").pack(side="left")
        ttk.Entry(row, textvariable=self.group, width=40).pack(side="left", padx=8)
        ttk.Label(body, text="Для всех ракурсов одного плода указывайте одинаковый ID; пусто = отдельный образец.",
                  wraplength=900).pack(anchor="w", pady=5)
        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=10)
        for index, label in enumerate(CLASSES, 1):
            ttk.Button(buttons, text=f"{index}. {LABELS_RU[label]}",
                       command=lambda c=label: self.assign(c)).pack(side="left", padx=4)
        ttk.Button(buttons, text="Пропустить", command=self.skip).pack(side="left", padx=10)
        ttk.Label(body, text="Метки выбирает человек. Не создавайте MINOR_DEFECT из размытия или случайной части rotten.",
                  wraplength=900).pack(anchor="w")
        self.show_current()

    def show_current(self):
        if self.index >= len(self.paths):
            messagebox.showinfo("Разметка", "Все изображения просмотрены. Теперь выполните prepare.")
            self.destroy()
            return
        path = self.paths[self.index]
        self.status.set(f"{self.index + 1}/{len(self.paths)} · {path}")
        try:
            image = Image.fromarray(read_rgb(path))
            image.thumbnail((860, 470))
            self.image = ImageTk.PhotoImage(image)
            self.preview.config(image=self.image)
        except (ValueError, OSError) as exc:
            messagebox.showwarning("Фото пропущено", str(exc))
            self.index += 1
            self.show_current()

    def assign(self, label: str):
        path = self.paths[self.index]
        group = self.group.get().strip() or f"specimen_{short_id(path)}"
        if not re.fullmatch(r"[\w-]{1,80}", group):
            messagebox.showwarning("ID серии", "Используйте буквы, цифры, _ или -, без пробелов и слешей.")
            return
        try:
            digest = image_hash(path)
            # Avoid quietly assigning one photo to conflicting labels.
            for other in CLASSES:
                if other != label and (self.output / other).exists():
                    if list((self.output / other).rglob(digest + ".*")):
                        raise ValueError("Это фото уже размечено другим классом. Исправьте прежнюю копию вручную.")
            destination = self.output / label / group / (digest + path.suffix.lower())
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            self.index += 1
            self.show_current()
        except (ValueError, OSError) as exc:
            messagebox.showerror("Не удалось сохранить", str(exc))

    def skip(self):
        self.index += 1
        self.show_current()


def launch(checkpoint: Path):
    FreshnessApp(checkpoint).mainloop()


def launch_labeler(source: Path, output: Path):
    LabelApp(source, output).mainloop()
