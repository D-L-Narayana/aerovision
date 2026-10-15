"""End-to-end: RGB frame → CLIP scene labels (+ probe) ; radiometric frame → hotspots ; → InspectionReport."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

from .clip_model import ClipEncoder
from .labels import CLASSES
from .probe import LinearProbe
from .report import SceneLabel, ThermalAnomaly, build_report, frame_hash, InspectionReport
from .thermal import detect_hotspots, detect_hot_strings
from .vlm import describe

ROOT = Path(__file__).resolve().parent.parent


class AeroVision:
    def __init__(self, encoder: ClipEncoder | None = None, probe_json: Path | None = None):
        self.enc = encoder or ClipEncoder()
        self.text_embs = self.enc.class_text_embeddings()
        self.probe_json = probe_json if probe_json and probe_json.exists() else None

    def scene_labels(self, img: Image.Image) -> list[SceneLabel]:
        e = self.enc.encode_images([img])
        zs = self.enc.zero_shot(e, self.text_embs)[0]
        labels = [SceneLabel(label=c, probability=float(p), source="zero_shot") for c, p in zip(CLASSES, zs)]
        if self.probe_json:
            pp = LinearProbe.predict_from_json(self.probe_json, e)[0]
            labels = [SceneLabel(label=c, probability=float(p), source="linear_probe") for c, p in zip(CLASSES, pp)]
        return sorted(labels, key=lambda s: -s.probability)[:5]

    @staticmethod
    def thermal_anomalies(frame_c: np.ndarray) -> list[ThermalAnomaly]:
        spots, _ = detect_hotspots(frame_c)
        out = [ThermalAnomaly(anomaly_type="hotspot", bbox=[s.x, s.y, s.w, s.h], delta_c=s.delta_c, severity=s.severity) for s in spots]
        out += [ThermalAnomaly(anomaly_type="hot_string", string=h["string"], delta_c=h["delta_c"], severity=h["severity"]) for h in detect_hot_strings(frame_c)]
        return out

    def inspect(self, rgb: Image.Image | None = None, frame_c: np.ndarray | None = None, frame_bytes: bytes = b"") -> InspectionReport:
        scene = self.scene_labels(rgb) if rgb is not None else []
        anomalies = self.thermal_anomalies(frame_c) if frame_c is not None else []
        summary = {"n_anomalies": len(anomalies), "max_delta_c": max((a.delta_c for a in anomalies), default=0.0)}
        finding = describe(rgb or Image.new("RGB", (8, 8)), summary)
        return build_report(frame_hash(frame_bytes or (rgb.tobytes() if rgb else frame_c.tobytes())), scene, anomalies,
                            finding.confidence, finding.summary)
