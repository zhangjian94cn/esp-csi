from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.ruview_posture.acceptance import (
    capabilities,
    load_acceptance,
    load_activation,
)


def write(path: Path, profile: str, *, passed: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "rvp-acceptance-v2",
                "profile": profile,
                "model_id": "model",
                "topology_id": "topology",
                "passed": passed,
            }
        )
    )


def test_capabilities_activate_in_order(tmp_path: Path) -> None:
    paths = []
    for profile in ("presence", "motion", "posture"):
        path = tmp_path / f"{profile}.json"
        write(path, profile)
        paths.append(path)
    accepted = load_acceptance(
        paths, model_id="model", topology_id="topology"
    )
    assert capabilities(accepted) == {
        "presence": True,
        "motion": True,
        "posture": True,
        "fall": False,
        "persons": False,
        "skeleton": False,
        "vital_signs": False,
    }


def test_failed_profile_cannot_activate(tmp_path: Path) -> None:
    path = tmp_path / "presence.json"
    write(path, "presence", passed=False)
    with pytest.raises(ValueError, match="did not pass"):
        load_acceptance([path], model_id="model", topology_id="topology")


def test_activation_rejects_out_of_order_profiles(tmp_path: Path) -> None:
    path = tmp_path / "activation.json"
    path.write_text(
        json.dumps(
            {
                "schema": "rvp-activation-v2",
                "state": "active",
                "model_id": "model",
                "topology_id": "topology",
                "acceptance": {
                    "posture": {
                        "schema": "rvp-acceptance-v2",
                        "profile": "posture",
                        "model_id": "model",
                        "topology_id": "topology",
                        "passed": True,
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="dependency order"):
        load_activation(path, model_id="model", topology_id="topology")


def test_activation_requires_matching_validation_stability(
    tmp_path: Path,
) -> None:
    report = {
        "schema": "rvp-acceptance-v2",
        "profile": "presence",
        "model_id": "model",
        "topology_id": "topology",
        "passed": True,
    }
    payload = {
        "schema": "rvp-activation-v2",
        "state": "active",
        "model_id": "model",
        "topology_id": "topology",
        "acceptance": {"presence": report},
        "stability": {
            "schema": "rvp-stability-v2",
            "passed": True,
            "model_ids": ["model"],
            "topology_ids": ["topology"],
            "sources": ["esp_csi_validation_model"],
        },
    }
    path = tmp_path / "activation.json"
    path.write_text(json.dumps(payload))
    assert set(
        load_activation(path, model_id="model", topology_id="topology")
    ) == {"presence"}

    payload["stability"]["sources"] = ["esp_csi_local_posture_model"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="stability evidence"):
        load_activation(path, model_id="model", topology_id="topology")
