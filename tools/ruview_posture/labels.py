"""Independent ground-truth fields for controlled CSI experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


LABEL_SCHEMA = "rvp-labels-v2"
OCCUPANCY_VALUES = {"absent", "present"}
MOTION_VALUES = {"idle", "moving", "unknown"}
POSTURE_VALUES = {"standing", "sitting", "lying", "unknown"}
EVENT_VALUES = {
    "none",
    "enter",
    "exit",
    "fall",
    "slow_lying",
    "sit_down",
    "stand_up",
    "pick_up",
    "motion_start",
    "motion_stop",
}
ENVIRONMENT_VALUES = {"off", "on", "unknown"}
DATASET_ROLES = {"train", "blind", "benchmark", "repeat"}


@dataclass(frozen=True)
class TrialLabels:
    occupancy: str
    motion: str
    posture: str
    event: str
    zone_id: str
    fan_state: str
    curtain_state: str
    trial_id: str
    dataset_role: str

    def validate(self) -> None:
        fields = {
            "occupancy": (self.occupancy, OCCUPANCY_VALUES),
            "motion": (self.motion, MOTION_VALUES),
            "posture": (self.posture, POSTURE_VALUES),
            "event": (self.event, EVENT_VALUES),
            "fan_state": (self.fan_state, ENVIRONMENT_VALUES),
            "curtain_state": (self.curtain_state, ENVIRONMENT_VALUES),
            "dataset_role": (self.dataset_role, DATASET_ROLES),
        }
        for name, (value, allowed) in fields.items():
            if value not in allowed:
                raise ValueError(
                    f"{name} must be one of {sorted(allowed)}, got {value!r}"
                )
        if not self.trial_id.strip():
            raise ValueError("trial_id must not be empty")
        if not self.zone_id.strip():
            raise ValueError("zone_id must not be empty")
        if self.occupancy == "absent":
            if self.motion != "idle" or self.posture != "unknown":
                raise ValueError(
                    "absent trials require motion=idle and posture=unknown"
                )
            if self.event not in {"none", "exit"}:
                raise ValueError("absent trials only allow none or exit events")
        if self.posture != "unknown" and self.occupancy != "present":
            raise ValueError("a known posture requires occupancy=present")
        if self.event == "fall" and self.occupancy != "present":
            raise ValueError("fall trials require occupancy=present")

    def as_metadata(self) -> dict[str, Any]:
        self.validate()
        return {"label_schema": LABEL_SCHEMA, **asdict(self)}


def labels_from_metadata(metadata: Mapping[str, Any]) -> TrialLabels:
    if metadata.get("label_schema") == LABEL_SCHEMA:
        labels = TrialLabels(
            **{
                field: str(metadata[field])
                for field in TrialLabels.__dataclass_fields__
            }
        )
        labels.validate()
        return labels

    # Read-only compatibility for recordings created by the first PoC.
    legacy = str(metadata.get("label", ""))
    mapping = {
        "absent": ("absent", "idle", "unknown", "none"),
        "standing": ("present", "idle", "standing", "none"),
        "sitting": ("present", "idle", "sitting", "none"),
        "lying": ("present", "idle", "lying", "none"),
        "moving": ("present", "moving", "unknown", "none"),
        "fall": ("present", "moving", "lying", "fall"),
        "slow_lying": ("present", "moving", "lying", "slow_lying"),
    }
    if legacy not in mapping:
        raise ValueError("recording does not contain v2 labels")
    occupancy, motion, posture, event = mapping[legacy]
    labels = TrialLabels(
        occupancy=occupancy,
        motion=motion,
        posture=posture,
        event=event,
        zone_id=str(metadata.get("zone_id", "legacy")),
        fan_state=str(metadata.get("fan_state", "unknown")),
        curtain_state=str(metadata.get("curtain_state", "unknown")),
        trial_id=str(metadata.get("trial_id", "legacy")),
        dataset_role=str(metadata.get("dataset_role", "train")),
    )
    labels.validate()
    return labels
