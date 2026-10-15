import numpy as np, pytest
from aerovision.thermal import detect_hotspots, detect_hot_strings, severity_for
from aerovision.report import build_report, SceneLabel, ThermalAnomaly, InspectionReport
from pydantic import ValidationError

def frame():
    f = np.full((60, 80), 40.0, dtype=np.float32); f[10:16, 20:28] += 25; f[40:44, 60:66] += 12; return f

def test_hotspots_detected_with_severity():
    spots, med = detect_hotspots(frame())
    assert abs(med - 40) < 0.1 and [s.severity for s in spots] == ["high", "medium"]
    assert (spots[0].x, spots[0].y) == (20, 10)

def test_below_threshold_ignored():
    f = np.full((30, 30), 40.0); f[5:9, 5:9] += 6
    assert detect_hotspots(f)[0] == [] and severity_for(6) == "low"

def test_hot_string():
    f = np.full((80, 40), 40.0); f[10:20] += 7
    hs = detect_hot_strings(f, n_strings=8); assert hs and hs[0]["string"] == 1 and hs[0]["severity"] == "medium"

def test_report_actions_and_confidence_gate():
    an = [ThermalAnomaly(anomaly_type="hotspot", bbox=[1, 1, 4, 4], delta_c=23.0, severity="high")]
    r = build_report("f1", [SceneLabel(label="storagetanks", probability=0.8, source="zero_shot")], an, 0.9)
    assert r.recommended_action == "same_day_work_order" and r.verified and "industrial" in r.flight_sensitivity[0]
    r2 = build_report("f2", [], an, 0.4); assert not r2.verified and "deterministic" in r2.narrative
    assert r.to_tool_result()["max_severity"] == "high"

def test_schema_rejects_bad_values():
    with pytest.raises(ValidationError):
        ThermalAnomaly(anomaly_type="hotspot", delta_c=-1, severity="high")
    with pytest.raises(ValidationError):
        InspectionReport(frame_id="x", confidence=1.5, verified=True)
