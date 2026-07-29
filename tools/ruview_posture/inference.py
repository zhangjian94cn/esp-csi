"""Shared staged inference for the three-head CSI model bundle."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .calibration import MotionCalibration
from .features import WindowFeatures, vector_for_head
from .model import (
    FirmwareBinding,
    HEAD_CLASSES,
    ModelManifest,
    recording_sha256,
)
from .tcn import predict_tcn


@dataclass(frozen=True)
class HeadDecision:
    label: str
    confidence: float


@dataclass(frozen=True)
class Prediction:
    presence: HeadDecision
    motion: HeadDecision
    posture: HeadDecision

    @property
    def confidence(self) -> float:
        decisions = [self.presence]
        if self.presence.label == "present":
            decisions.extend((self.motion, self.posture))
        values = [
            decision.confidence
            for decision in decisions
            if decision.label != "unknown"
        ]
        return min(values) if values else 0.0


class PosturePredictor:
    def __init__(
        self,
        model_directory: Path,
        *,
        topology_id: str,
        firmware: FirmwareBinding | None = None,
    ):
        self.manifest = ModelManifest.load(model_directory / "manifest.json")
        failures = self.manifest.validate_binding(
            topology_id=topology_id, firmware=firmware
        )
        if failures:
            raise ValueError(f"model binding mismatch: {', '.join(failures)}")
        self.motion_calibration = MotionCalibration.from_dict(
            self.manifest.motion_calibration
        )
        artifact_path = model_directory / "model-bundle.joblib"
        if recording_sha256(artifact_path) != self.manifest.model_artifact_sha256:
            raise ValueError("model artifact SHA-256 does not match manifest")
        self.models: dict[str, Any] = joblib.load(artifact_path)
        if set(self.models) != set(HEAD_CLASSES):
            raise ValueError("model bundle does not contain all three heads")
        self.histories: dict[str, deque[np.ndarray]] = {
            head: deque(maxlen=8) for head in HEAD_CLASSES
        }

    def reset(self) -> None:
        for history in self.histories.values():
            history.clear()

    def _predict_head(
        self, head: str, window: WindowFeatures
    ) -> HeadDecision:
        model = self.models[head]
        vector = vector_for_head(window, head)
        if self.manifest.head_kinds[head] == "logistic":
            probabilities = model.predict_proba(vector.reshape(1, -1))[0]
            classes = model.classes_
        else:
            history = self.histories[head]
            history.append(vector)
            if len(history) < history.maxlen:
                return HeadDecision("unknown", 0.0)
            sequence = np.stack(history).T[np.newaxis, :, :]
            _, probabilities = predict_tcn(model, sequence)
            probabilities = probabilities[0]
            classes = model.classes
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        if confidence < self.manifest.confidence_thresholds[head]:
            return HeadDecision("unknown", confidence)
        return HeadDecision(str(classes[best]), confidence)

    def predict(self, window: WindowFeatures) -> Prediction:
        presence = self._predict_head("presence", window)
        if presence.label != "present":
            return Prediction(
                presence=presence,
                motion=HeadDecision("unknown", 0.0),
                posture=HeadDecision("unknown", 0.0),
            )
        return Prediction(
            presence=presence,
            motion=self._predict_head("motion", window),
            posture=self._predict_head("posture", window),
        )
