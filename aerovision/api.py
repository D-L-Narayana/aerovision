"""FastAPI: POST /inspect (multipart RGB image + optional .npy thermal frame) → InspectionReport JSON."""
from __future__ import annotations
import io
from functools import lru_cache
from pathlib import Path
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from . import __version__
from .pipeline import AeroVision

app = FastAPI(title="AeroVision", version=__version__)


@lru_cache(maxsize=1)
def engine() -> AeroVision:
    return AeroVision(probe_json=Path(__file__).resolve().parent.parent / "web" / "probe_head.json")


@app.get("/health")
def health():
    e = engine(); return {"ok": True, "model": e.enc.model_id, "device": e.enc.device, "probe": e.probe_json is not None}


@app.post("/inspect")
async def inspect(rgb: UploadFile | None = File(None), thermal: UploadFile | None = File(None)):
    img = Image.open(io.BytesIO(await rgb.read())) if rgb else None
    frame = np.load(io.BytesIO(await thermal.read())) if thermal else None
    return engine().inspect(img, frame).model_dump()
