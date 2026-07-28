"""Shared posture model inference for live service and blind replay."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .features import WindowFeatures
from .model import ModelManifest
from .tcn import predict_tcn


class PosturePredictor:
    def __init__(self, model_directory: Path, *, topology_id: str):
        self.manifest = ModelManifest.load(model_directory / "manifest.json")
        failures = self.manifest.validate_binding(topology_id=topology_id)
        if failures:
            raise ValueError(f"model binding mismatch: {', '.join(failures)}")
        artifact_name = (
            "model.joblib"
            if self.manifest.model_kind == "logistic"
            else "model-tcn.joblib"
        )
        self.model: Any = joblib.load(model_directory / artifact_name)
        self.history: deque[np.ndarray] = deque(maxlen=8)

    def predict(self, window: WindowFeatures) -> tuple[str, float]:
        if self.manifest.model_kind == "logistic":
            probabilities = self.model.predict_proba(window.vector.reshape(1, -1))[0]
            classes = self.model.classes_
        else:
            self.history.append(window.vector)
            if len(self.history) < self.history.maxlen:
                return "unknown", 0.0
            sequence = np.stack(self.history).T[np.newaxis, :, :]
            _, probabilities = predict_tcn(self.model, sequence)
            probabilities = probabilities[0]
            classes = self.model.classes
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        if confidence < self.manifest.confidence_threshold:
            return "unknown", confidence
        return str(classes[best]), confidence
