"""CLIP (Hugging Face transformers, PyTorch) image/text encoders with prompt-ensemble zero-shot classification."""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .labels import CLASSES, PROMPT_NAMES, TEMPLATES

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


def _tensor(x):
    """transformers>=5 returns a BaseModelOutputWithPooling (projected features in pooler_output); <5 returns a tensor."""
    return x if torch.is_tensor(x) else x.pooler_output


class ClipEncoder:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None, local_dir: Path | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        src = str(local_dir) if local_dir and Path(local_dir).exists() else model_id
        self.model = CLIPModel.from_pretrained(src).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(src)
        self.model_id = model_id

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image], batch: int = 32) -> np.ndarray:
        out = []
        for s in range(0, len(images), batch):
            px = self.proc(images=[im.convert("RGB") for im in images[s:s + batch]], return_tensors="pt")["pixel_values"].to(self.device)
            f = _tensor(self.model.get_image_features(pixel_values=px))
            out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
        return np.vstack(out).astype(np.float32)

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        t = self.proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        f = _tensor(self.model.get_text_features(**t))
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy().astype(np.float32)

    def class_text_embeddings(self, classes: list[str] = CLASSES, templates: list[str] = TEMPLATES) -> np.ndarray:
        """Prompt ensemble: average the normalised embeddings of every template for each class, re-normalise."""
        embs = []
        for c in classes:
            e = self.encode_texts([t.format(PROMPT_NAMES.get(c, c)) for t in templates]).mean(0)
            embs.append(e / np.linalg.norm(e))
        return np.stack(embs).astype(np.float32)

    def zero_shot(self, image_embs: np.ndarray, text_embs: np.ndarray, temperature: float = 100.0) -> np.ndarray:
        logits = temperature * image_embs @ text_embs.T
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(1, keepdims=True)

    def time_per_image_ms(self, image: Image.Image, n: int = 10) -> float:
        self.encode_images([image])
        t0 = time.perf_counter()
        for _ in range(n):
            self.encode_images([image])
        return (time.perf_counter() - t0) / n * 1000
