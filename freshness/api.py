"""Optional localhost-only FastAPI application."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from .common import MAX_PIXELS
from .inference import Predictor

MAX_UPLOAD = 20 * 1024 * 1024
FORM = """<!doctype html><html lang="ru"><meta charset="utf-8">
<title>Freshness NIR API</title><body style="font:16px system-ui;max-width:850px;margin:40px auto">
<h1>Оценка свежести — локальный API</h1>
<p>Фото обрабатывается на этом компьютере. Оценка не гарантирует безопасность продукта.</p>
<form id="form"><input type="file" name="file" accept="image/*" required>
<button>Распознать</button></form><pre id="out"></pre>
<script>
document.getElementById('form').onsubmit=async(e)=>{
e.preventDefault();const out=document.getElementById('out');out.textContent='Обработка...';
try{const r=await fetch('/predict',{method:'POST',body:new FormData(e.target)});
out.textContent=JSON.stringify(await r.json(),null,2);}
catch(err){out.textContent=String(err);}
};
</script></body></html>"""


def create_app(checkpoint: Path, threads: int = 4) -> FastAPI:
    predictor = Predictor(checkpoint, threads=threads)
    lock = Lock()
    # Default Swagger UI requires external CDN files; use our offline HTML instead.
    app = FastAPI(title="Freshness NIR", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return FORM

    @app.get("/health")
    def health():
        return {"status": "ok", "device": "cpu",
                "smoke_test_model": bool(predictor.metadata.get("smoke_test", False))}

    @app.post("/predict")
    def predict(file: UploadFile = File(...)):
        data = file.file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(status_code=413, detail="Максимум 20 MiB на фото.")
        try:
            with Image.open(BytesIO(data)) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Слишком большое изображение.")
                image = ImageOps.exif_transpose(image)
                if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    image = Image.alpha_composite(Image.new("RGBA", rgba.size, (114,114,114,255)), rgba)
                rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise HTTPException(status_code=400, detail=f"Неверное изображение: {exc}") from exc
        with lock:
            result = predictor.predict_rgb(rgb)
        result["file"] = file.filename
        result["timing_scope"] = "API: tensor preprocessing+model+softmax; excludes HTTP and image decoding."
        return result

    return app
