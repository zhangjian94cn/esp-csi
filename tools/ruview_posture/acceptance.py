"""Load profile acceptance reports and derive honest runtime capabilities."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable


PROFILES = ("presence", "motion", "posture", "fall")


def load_acceptance(
    paths: Iterable[Path], *, model_id: str, topology_id: str
) -> dict[str, dict[str, Any]]:
    accepted: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("schema") != "rvp-acceptance-v2":
            raise ValueError(f"{path} has an unsupported acceptance schema")
        profile = str(payload.get("profile"))
        if profile not in PROFILES:
            raise ValueError(f"{path} has an unknown profile")
        if not payload.get("passed"):
            raise ValueError(f"{path} did not pass its profile gates")
        if payload.get("model_id") != model_id:
            raise ValueError(f"{path} belongs to a different model")
        if payload.get("topology_id") != topology_id:
            raise ValueError(f"{path} belongs to a different topology")
        if profile in accepted:
            raise ValueError(f"duplicate acceptance profile {profile}")
        accepted[profile] = payload
    return accepted


def capabilities(
    accepted: dict[str, dict[str, Any]]
) -> dict[str, bool]:
    presence = "presence" in accepted
    motion = presence and "motion" in accepted
    posture = motion and "posture" in accepted
    fall = posture and "fall" in accepted
    return {
        "presence": presence,
        "motion": motion,
        "posture": posture,
        "fall": fall,
        "persons": False,
        "skeleton": False,
        "vital_signs": False,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_activation(
    path: Path, *, model_id: str, topology_id: str
) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "rvp-activation-v1":
        raise ValueError("unsupported activation schema")
    if payload.get("model_id") != model_id:
        raise ValueError("activation belongs to a different model")
    if payload.get("topology_id") != topology_id:
        raise ValueError("activation belongs to a different topology")
    accepted = payload.get("acceptance")
    if not isinstance(accepted, dict):
        raise ValueError("activation does not contain acceptance reports")
    for profile, report in accepted.items():
        if profile not in PROFILES:
            raise ValueError(f"activation has unknown profile {profile}")
        if (
            report.get("schema") != "rvp-acceptance-v2"
            or report.get("profile") != profile
            or report.get("model_id") != model_id
            or report.get("topology_id") != topology_id
            or not report.get("passed")
        ):
            raise ValueError(f"activation profile {profile} is invalid")
    expected_order = list(PROFILES[: len(accepted)])
    if list(accepted) != expected_order:
        raise ValueError("activation profiles are not in dependency order")
    return accepted
