"""Linear probe on frozen CLIP embeddings — the cheap, reliable way to adapt a VLM to a site's imagery with a handful
of labelled frames. Exports to JSON so the exact same head runs in the browser demo."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


class LinearProbe:
    def __init__(self, C: float = 2.0, max_iter: int = 2000):
        self.clf = LogisticRegression(C=C, max_iter=max_iter)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def export_json(self, path: Path, classes: list[str]) -> None:
        W, b = self.clf.coef_, self.clf.intercept_
        if W.shape[0] == 1:  # binary: softmax([0, z]) == sigmoid(z)
            W, b = np.vstack([np.zeros_like(W), W]), np.concatenate([[0.0], b])
        path.write_text(json.dumps({"classes": classes, "W": np.round(W, 5).tolist(), "b": np.round(b, 5).tolist()}))

    @staticmethod
    def predict_from_json(path: Path, X: np.ndarray) -> np.ndarray:
        d = json.loads(Path(path).read_text())
        z = X @ np.array(d["W"]).T + np.array(d["b"])
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)
