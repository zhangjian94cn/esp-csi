"""Three-head model manifest, binding checks, and evaluation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .calibration import MotionCalibration
from .features import FEATURE_SCHEMA_HASH
from .protocol import PROTOCOL_VERSION

OFFICIAL_ESP_CSI_COMMIT = "8633d67152db2808f141cc1595970aa9cf406045"
ESP_WIFI_SENSING_VERSION = "0.1.1~2"
PRESENCE_CLASSES = ("absent", "present")
MOTION_CLASSES = ("still", "moving")
POSTURE_CLASSES = ("standing", "sitting", "lying")
HEAD_CLASSES = {
    "presence": PRESENCE_CLASSES,
    "motion": MOTION_CLASSES,
    "posture": POSTURE_CLASSES,
}
REQUIRED_FIRMWARE_NODES = {"1", "2", "3"}


@dataclass(frozen=True)
class FirmwareBinding:
    fork_commit: str
    protocol_version: int
    probe_rate_hz: int
    build_ids: dict[str, str]
    artifact_sha256: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> "FirmwareBinding":
        data = json.loads(path.read_text())
        if data.get("schema") != "rvp-firmware-binding-v1":
            raise ValueError("unsupported firmware binding schema")
        binding = cls(
            fork_commit=str(data["fork_commit"]),
            protocol_version=int(data["protocol_version"]),
            probe_rate_hz=int(data["probe_rate_hz"]),
            build_ids={
                str(node): str(value)
                for node, value in data["build_ids"].items()
            },
            artifact_sha256={
                str(node): str(value)
                for node, value in data["artifact_sha256"].items()
            },
        )
        binding.validate()
        return binding

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("firmware protocol does not match host protocol")
        if self.probe_rate_hz not in {50, 100}:
            raise ValueError("firmware probe rate must be 50 or 100 Hz")
        if set(self.build_ids) != REQUIRED_FIRMWARE_NODES:
            raise ValueError("firmware binding requires build IDs for nodes 1/2/3")
        if set(self.artifact_sha256) != REQUIRED_FIRMWARE_NODES:
            raise ValueError(
                "firmware binding requires artifact hashes for nodes 1/2/3"
            )
        for digest in self.artifact_sha256.values():
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("firmware artifact SHA-256 is invalid")


@dataclass(frozen=True)
class ModelManifest:
    schema: str
    model_id: str
    topology_id: str
    feature_schema_hash: str
    protocol_version: int
    esp_csi_commit: str
    esp_wifi_sensing_version: str
    fork_commit: str
    probe_rate_hz: int
    firmware_build_ids: dict[str, str]
    firmware_artifact_sha256: dict[str, str]
    model_artifact_sha256: str
    training_recordings: tuple[str, ...]
    training_data_hash: str
    calibration_id: str
    motion_calibration: dict[str, Any]
    head_kinds: dict[str, str]
    classes: dict[str, tuple[str, ...]]
    confidence_thresholds: dict[str, float]
    metrics: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "ModelManifest":
        data = json.loads(path.read_text())
        data["training_recordings"] = tuple(data["training_recordings"])
        data["classes"] = {
            head: tuple(classes) for head, classes in data["classes"].items()
        }
        return cls(**data)

    def save(self, path: Path) -> None:
        encoded = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(path)

    def validate_binding(
        self,
        *,
        topology_id: str,
        firmware: FirmwareBinding | None = None,
    ) -> list[str]:
        failures: list[str] = []
        if self.schema != "rvp-model-bundle-v2":
            failures.append("schema")
        if self.topology_id != topology_id:
            failures.append("topology_id")
        if self.feature_schema_hash != FEATURE_SCHEMA_HASH:
            failures.append("feature_schema_hash")
        if self.protocol_version != PROTOCOL_VERSION:
            failures.append("protocol_version")
        if self.esp_csi_commit != OFFICIAL_ESP_CSI_COMMIT:
            failures.append("esp_csi_commit")
        if self.esp_wifi_sensing_version != ESP_WIFI_SENSING_VERSION:
            failures.append("esp_wifi_sensing_version")
        if len(self.model_artifact_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.model_artifact_sha256
        ):
            failures.append("model_artifact_sha256")
        try:
            calibration = MotionCalibration.from_dict(self.motion_calibration)
        except (KeyError, TypeError, ValueError):
            failures.append("motion_calibration")
        else:
            if calibration.calibration_id != self.calibration_id:
                failures.append("calibration_id")
            if calibration.topology_id != self.topology_id:
                failures.append("calibration_topology_id")
            if calibration.probe_rate_hz != self.probe_rate_hz:
                failures.append("calibration_probe_rate_hz")
        if set(self.head_kinds) != set(HEAD_CLASSES):
            failures.append("head_kinds")
        if self.classes != HEAD_CLASSES:
            failures.append("classes")
        if firmware is not None:
            comparisons = {
                "fork_commit": (self.fork_commit, firmware.fork_commit),
                "probe_rate_hz": (
                    self.probe_rate_hz,
                    firmware.probe_rate_hz,
                ),
                "firmware_build_ids": (
                    self.firmware_build_ids,
                    firmware.build_ids,
                ),
                "firmware_artifact_sha256": (
                    self.firmware_artifact_sha256,
                    firmware.artifact_sha256,
                ),
            }
            if firmware.protocol_version != self.protocol_version:
                failures.append("firmware_protocol_version")
            failures.extend(
                name
                for name, (actual, expected) in comparisons.items()
                if actual != expected
            )
        return failures


def recording_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_hash(values: Iterable[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            sorted(values), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def derive_model_id(
    *,
    topology_id: str,
    head_kinds: Mapping[str, str],
    recording_hashes: Iterable[str],
    calibration_id: str,
    firmware: FirmwareBinding,
    model_artifact_sha256: str,
) -> str:
    identity = {
        "schema": "rvp-model-bundle-v2",
        "topology_id": topology_id,
        "head_kinds": dict(sorted(head_kinds.items())),
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "protocol_version": PROTOCOL_VERSION,
        "esp_csi_commit": OFFICIAL_ESP_CSI_COMMIT,
        "esp_wifi_sensing_version": ESP_WIFI_SENSING_VERSION,
        "fork_commit": firmware.fork_commit,
        "probe_rate_hz": firmware.probe_rate_hz,
        "firmware_build_ids": firmware.build_ids,
        "firmware_artifact_sha256": firmware.artifact_sha256,
        "model_artifact_sha256": model_artifact_sha256,
        "calibration_id": calibration_id,
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
