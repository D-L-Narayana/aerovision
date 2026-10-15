"""Schema-validated inspection report — the contract between perception and the acting agent (e.g. AeroSentry's
analyst agent consumes this as a tool result). Pydantic enforces types, ranges and the SOP's confidence gate."""
from __future__ import annotations
import hashlib
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .labels import FLIGHT_SENSITIVITY

Severity = Literal["none", "low", "medium", "high"]


class SceneLabel(BaseModel):
    label: str
    probability: float = Field(ge=0, le=1)
    source: Literal["zero_shot", "linear_probe"]


class ThermalAnomaly(BaseModel):
    anomaly_type: Literal["hotspot", "hot_string"]
    bbox: list[int] | None = None   # x, y, w, h in frame pixels
    string: int | None = None
    delta_c: float = Field(ge=0)
    severity: Severity


class InspectionReport(BaseModel):
    frame_id: str
    scene: list[SceneLabel] = Field(default_factory=list)
    flight_sensitivity: list[str] = Field(default_factory=list)
    anomalies: list[ThermalAnomaly] = Field(default_factory=list)
    max_severity: Severity = "none"
    recommended_action: Literal["none", "log", "schedule_within_7_days", "same_day_work_order", "escalate"] = "none"
    confidence: float = Field(ge=0, le=1)
    verified: bool = Field(description="False when VLM confidence < 0.6; deterministic detector then decides severity (SOP).")
    narrative: str = ""

    @field_validator("scene")
    @classmethod
    def _sorted(cls, v):
        return sorted(v, key=lambda s: -s.probability)[:5]

    def to_tool_result(self) -> dict:
        """Compact dict handed to an agent tool loop."""
        return {"frame_id": self.frame_id, "top_scene": self.scene[0].label if self.scene else None,
                "max_severity": self.max_severity, "n_anomalies": len(self.anomalies),
                "recommended_action": self.recommended_action, "verified": self.verified,
                "flight_sensitivity": self.flight_sensitivity}


def action_for(sev: Severity) -> str:
    return {"none": "none", "low": "log", "medium": "schedule_within_7_days", "high": "same_day_work_order"}[sev]


def frame_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def build_report(frame_id: str, scene: list[SceneLabel], anomalies: list[ThermalAnomaly], vlm_confidence: float,
                 narrative: str = "") -> InspectionReport:
    sev_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    max_sev = max((a.severity for a in anomalies), key=lambda s: sev_rank[s], default="none")
    verified = vlm_confidence >= 0.6
    sens = sorted({FLIGHT_SENSITIVITY[s.label] for s in scene[:2] if s.label in FLIGHT_SENSITIVITY and s.probability >= 0.3})
    return InspectionReport(frame_id=frame_id, scene=scene, flight_sensitivity=sens, anomalies=anomalies, max_severity=max_sev,
                            recommended_action=action_for(max_sev), confidence=vlm_confidence, verified=verified,
                            narrative=narrative or ("" if verified else "VLM confidence below 0.6 — severity set by deterministic detector."))
