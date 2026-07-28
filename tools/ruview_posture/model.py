"""Model artifact manifest, binding checks, and shared evaluation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .features import FEATURE_SCHEMA_HASH

OFFICIAL_ESP_CSI_COMMIT = "8633d67152db2808f141cc1595970aa9cf406045"
ESP_WIFI_SENSING_VERSION = "0.1.1~2"
POSTURE_CLASSES = ("absent", "standing", "sitting", "lying")


@dataclass(frozen=True)
class ModelManifest:
    schema: str
    model_id: str
    model_kind: str
    topology_id: str
    feature_schema_hash: str
    esp_csi_commit: str
    esp_wifi_sensing_version: str
    training_recordings: tuple[str, ...]
    classes: tuple[str, ...]
    confidence_threshold: float
    motion_threshold: float
    fall_motion_threshold: float
    metrics: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "ModelManifest":
        data = json.loads(path.read_text())
        data["training_recordings"] = tuple(data["training_recordings"])
        data["classes"] = tuple(data["classes"])
        return cls(**data)

    def save(self, path: Path) -> None:
        encoded = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(path)

    def validate_binding(self, *, topology_id: str) -> list[str]:
        failures: list[str] = []
        if self.topology_id != topology_id:
            failures.append("topology_id")
        if self.feature_schema_hash != FEATURE_SCHEMA_HASH:
            failures.append("feature_schema_hash")
        if self.esp_csi_commit != OFFICIAL_ESP_CSI_COMMIT:
            failures.append("esp_csi_commit")
        if self.esp_wifi_sensing_version != ESP_WIFI_SENSING_VERSION:
            failures.append("esp_wifi_sensing_version")
        return failures


def recording_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_model_id(
    *,
    topology_id: str,
    model_kind: str,
    recording_hashes: Iterable[str],
) -> str:
    identity = {
        "schema": "rvp-posture-model-v1",
        "topology_id": topology_id,
        "model_kind": model_kind,
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "esp_csi_commit": OFFICIAL_ESP_CSI_COMMIT,
        "esp_wifi_sensing_version": ESP_WIFI_SENSING_VERSION,
        "recordings": sorted(recording_hashes),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def macro_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
    classes: Iterable[str] = POSTURE_CLASSES,
) -> dict[str, Any]:
    labels = list(classes)
    recalls: dict[str, float] = {}
    f1s: list[float] = []
    for label in labels:
        true_positive = int(np.sum((expected == label) & (predicted == label)))
        false_negative = int(np.sum((expected == label) & (predicted != label)))
        false_positive = int(np.sum((expected != label) & (predicted == label)))
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        recalls[label] = recall
        f1s.append(f1)
    return {
        "macro_f1": float(np.mean(f1s)),
        "recall": recalls,
        "samples": int(len(expected)),
    }
