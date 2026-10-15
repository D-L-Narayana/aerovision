"""Synthetic radiometric frames (°C) with ground-truth hotspots — 200 frames, seeded. Saved as .npz + PNG previews."""
import json, random
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "thermal"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(11)
H, W = 96, 128
truth = []
frames = []
for i in range(200):
    base = 38 + rng.normal(0, 1.2, (H, W)) + np.linspace(-1.5, 1.5, W)[None, :]      # ambient module temp + gradient
    spots = []
    for _ in range(rng.integers(0, 4)):
        d = float(rng.choice([6, 8, 12, 15, 18, 22, 28, 35]))
        w, h = int(rng.integers(4, 14)), int(rng.integers(4, 14)); x, y = int(rng.integers(0, W - w)), int(rng.integers(0, H - h))
        yy, xx = np.mgrid[0:h, 0:w]; blob = np.exp(-(((yy - h / 2) / (h / 2.2)) ** 2 + ((xx - w / 2) / (w / 2.2)) ** 2))
        base[y:y + h, x:x + w] += d * blob
        spots.append({"x": x, "y": y, "w": w, "h": h, "delta_c": d, "severity": "high" if d >= 20 else "medium" if d >= 10 else "low"})
    hot_string = None
    if rng.random() < 0.15:
        s = int(rng.integers(0, 8)); band = np.array_split(np.arange(H), 8)[s]; d = float(rng.choice([6, 9, 12]))
        base[band] += d; hot_string = {"string": s, "delta_c": d}
    frames.append(base.astype(np.float32)); truth.append({"id": i, "hotspots": spots, "hot_string": hot_string})
    if i < 6:
        v = np.clip((base - 30) / 40, 0, 1); rgb = np.stack([v ** 0.5, v ** 2, 1 - v], -1)
        Image.fromarray((rgb * 255).astype(np.uint8)).resize((W * 4, H * 4), Image.NEAREST).save(OUT / f"preview_{i}.png")
np.savez_compressed(OUT / "frames.npz", frames=np.stack(frames))
(OUT / "truth.json").write_text(json.dumps(truth))
print(len(frames), "frames,", sum(len(t["hotspots"]) for t in truth), "hotspots")
