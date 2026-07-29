"""Temporal fall suspicion logic for the research-only posture runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .features import FEATURE_SCHEMA_HASH

FALL_MODEL_SCHEMA = "rvp-fall-model-v1"


@dataclass(frozen=True)
class FallDecision:
    event: str
    latched_until_ms: int | None


@dataclass(frozen=True)
class FallModel:
    schema: str
    fall_model_id: str
    posture_model_id: str
    topology_id: str
    feature_schema_hash: str
    motion_calibration_id: str
    training_recordings: tuple[str, ...]
    training_data_hash: str
    motion_threshold: float
    transition_ms: int
    lying_confirmations: int
    latch_ms: int
    metrics: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        posture_model_id: str,
        topology_id: str,
        motion_calibration_id: str,
        training_recordings: Iterable[str],
        motion_threshold: float,
        transition_ms: int,
        lying_confirmations: int,
        latch_ms: int,
        metrics: dict[str, Any],
    ) -> "FallModel":
        recordings = tuple(sorted(set(training_recordings)))
        training_data_hash = hashlib.sha256(
            json.dumps(recordings, separators=(",", ":")).encode()
        ).hexdigest()
        identity = {
            "schema": FALL_MODEL_SCHEMA,
            "posture_model_id": posture_model_id,
            "topology_id": topology_id,
            "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "motion_calibration_id": motion_calibration_id,
            "training_recordings": recordings,
            "training_data_hash": training_data_hash,
            "motion_threshold": motion_threshold,
            "transition_ms": transition_ms,
            "lying_confirmations": lying_confirmations,
            "latch_ms": latch_ms,
        }
        fall_model_id = hashlib.sha256(
            json.dumps(
                identity, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        model = cls(
            fall_model_id=fall_model_id,
            metrics=metrics,
            **identity,
        )
        model.validate()
        return model

    @classmethod
    def load(cls, path: Path) -> "FallModel":
        payload = json.loads(path.read_text())
        payload["training_recordings"] = tuple(
            payload["training_recordings"]
        )
        model = cls(**payload)
        model.validate()
        expected = cls.create(
            posture_model_id=model.posture_model_id,
            topology_id=model.topology_id,
            motion_calibration_id=model.motion_calibration_id,
            training_recordings=model.training_recordings,
            motion_threshold=model.motion_threshold,
            transition_ms=model.transition_ms,
            lying_confirmations=model.lying_confirmations,
            latch_ms=model.latch_ms,
            metrics=model.metrics,
        )
        if (
            model.fall_model_id != expected.fall_model_id
            or model.feature_schema_hash != expected.feature_schema_hash
            or model.training_data_hash != expected.training_data_hash
        ):
            raise ValueError("fall model ID does not match its content")
        return model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(path)

    def validate_binding(
        self,
        *,
        posture_model_id: str,
        topology_id: str,
        motion_calibration_id: str,
    ) -> list[str]:
        failures: list[str] = []
        if self.posture_model_id != posture_model_id:
            failures.append("posture_model_id")
        if self.topology_id != topology_id:
            failures.append("topology_id")
        if self.motion_calibration_id != motion_calibration_id:
            failures.append("motion_calibration_id")
        if self.feature_schema_hash != FEATURE_SCHEMA_HASH:
            failures.append("feature_schema_hash")
        return failures

    def validate(self) -> None:
        if self.schema != FALL_MODEL_SCHEMA:
            raise ValueError("unsupported fall model schema")
        if (
            not math.isfinite(self.motion_threshold)
            or self.motion_threshold <= 0
        ):
            raise ValueError("fall motion threshold must be positive")
        if not self.training_recordings:
            raise ValueError("fall model requires training recordings")
        if self.transition_ms <= 0 or self.latch_ms <= 0:
            raise ValueError("fall model timing must be positive")
        if self.lying_confirmations <= 0:
            raise ValueError("fall model confirmations must be positive")

    def detector(self) -> "FallDetector":
        return FallDetector(
            motion_threshold=self.motion_threshold,
            transition_ms=self.transition_ms,
            lying_confirmations=self.lying_confirmations,
            latch_ms=self.latch_ms,
        )


class FallDetector:
    def __init__(
        self,
        *,
        motion_threshold: float,
        transition_ms: int = 1500,
        lying_confirmations: int = 2,
        latch_ms: int = 10_000,
    ):
        self.motion_threshold = motion_threshold
        self.transition_ms = transition_ms
        self.lying_confirmations = lying_confirmations
        self.latch_ms = latch_ms
        self.previous_posture = "unknown"
        self.transition_started_ms: int | None = None
        self.lying_count = 0
        self.latched_until_ms: int | None = None

    def update(
        self, *, now_ms: int, posture: str, motion_energy: float
    ) -> FallDecision:
        if self.latched_until_ms is not None and now_ms < self.latched_until_ms:
            self.previous_posture = posture
            return FallDecision("suspected", self.latched_until_ms)
        self.latched_until_ms = None

        if (
            self.previous_posture in {"standing", "sitting"}
            and motion_energy >= self.motion_threshold
        ):
            self.transition_started_ms = now_ms
            self.lying_count = 0

        if self.transition_started_ms is not None:
            if now_ms - self.transition_started_ms > self.transition_ms:
                self.transition_started_ms = None
                self.lying_count = 0
            elif posture == "lying":
                self.lying_count += 1
                if self.lying_count >= self.lying_confirmations:
                    self.latched_until_ms = now_ms + self.latch_ms
                    self.transition_started_ms = None
                    self.lying_count = 0
                    self.previous_posture = posture
                    return FallDecision("suspected", self.latched_until_ms)
            elif posture not in {"unknown", "lying"}:
                self.lying_count = 0

        self.previous_posture = posture
        return FallDecision("none", None)
