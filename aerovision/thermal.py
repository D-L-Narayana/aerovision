"""Deterministic radiometric hotspot detector (NumPy). Thresholds follow the thermal-inspection SOP:
cell hotspot ≥ 10 °C above median → medium, ≥ 20 °C → high; whole-string ≥ 5 °C → hot string."""
from __future__ import annotations
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Hotspot:
    x: int; y: int; w: int; h: int
    delta_c: float
    severity: str  # low | medium | high

    def to_dict(self):
        return asdict(self)


def _label_components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """4-connected components via iterative flood fill; returns bounding boxes (x, y, w, h)."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]; seen[y0, x0] = True
        miny = maxy = y0; minx = maxx = x0; n = 0
        while stack:
            y, x = stack.pop(); n += 1
            miny, maxy, minx, maxx = min(miny, y), max(maxy, y), min(minx, x), max(maxx, x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True; stack.append((ny, nx))
        if n >= 4:
            boxes.append((int(minx), int(miny), int(maxx - minx + 1), int(maxy - miny + 1)))
    return boxes


def severity_for(delta: float) -> str:
    return "high" if delta >= 20 else "medium" if delta >= 10 else "low"


def string_reference(frame_c: np.ndarray, n_strings: int = 8) -> np.ndarray:
    """Per-string (horizontal band) median temperature, broadcast to frame shape. A cell hotspot is measured against
    its own string so that a uniformly hot string is reported once (hot_string) rather than as one giant hotspot."""
    ref = np.empty_like(frame_c, dtype=np.float32)
    start = 0
    for band in np.array_split(np.arange(frame_c.shape[0]), n_strings):
        ref[band] = np.median(frame_c[band]); start += len(band)
    return ref


def detect_hotspots(frame_c: np.ndarray, min_delta: float = 10.0, n_strings: int = 8) -> tuple[list[Hotspot], float]:
    """frame_c: 2-D array of temperatures in °C. Returns hotspots (delta vs. their string median) and the frame median."""
    median = float(np.median(frame_c))
    ref = string_reference(frame_c, n_strings)
    excess = frame_c - ref
    mask = excess >= min_delta
    spots = []
    for x, y, w, h in _label_components(mask):
        delta = float(excess[y:y + h, x:x + w].max())
        spots.append(Hotspot(x, y, w, h, round(delta, 1), severity_for(delta)))
    spots.sort(key=lambda s: -s.delta_c)
    return spots, median


def detect_hot_strings(frame_c: np.ndarray, n_strings: int = 8, min_delta: float = 5.0) -> list[dict]:
    """Treat the frame as n horizontal strings; a string whose median is ≥ min_delta above the frame median is hot."""
    median = float(np.median(frame_c))
    out = []
    for i, band in enumerate(np.array_split(frame_c, n_strings, axis=0)):
        d = float(np.median(band) - median)
        if d >= min_delta:
            out.append({"string": i, "delta_c": round(d, 1), "severity": "high" if d >= 10 else "medium"})
    return out
