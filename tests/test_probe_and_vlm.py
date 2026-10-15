import numpy as np
from aerovision.probe import LinearProbe
from aerovision.vlm import describe, VLMFinding
from PIL import Image

def test_probe_roundtrip_json(tmp_path):
    rng = np.random.default_rng(0); X = rng.normal(size=(60, 8)); y = (X[:, 0] > 0).astype(int)
    p = LinearProbe().fit(X, y); p.export_json(tmp_path / "w.json", ["a", "b"])
    pj = LinearProbe.predict_from_json(tmp_path / "w.json", X)
    assert np.allclose(pj, p.predict_proba(X), atol=1e-4)

def test_mock_vlm_confidence_gate():
    f = describe(Image.new("RGB", (8, 8)), {"n_anomalies": 1, "max_delta_c": 7.0}, provider="mock")
    assert isinstance(f, VLMFinding) and f.confidence < 0.6
    assert describe(Image.new("RGB", (8, 8)), {"n_anomalies": 0, "max_delta_c": 0.0}, provider="mock").confidence >= 0.6
