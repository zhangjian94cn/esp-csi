from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.ruview_posture.topology import (
    TopologyError,
    finalize_topology,
    load_topology,
    topology_id,
)


def topology() -> dict:
    return {
        "schema": "rvp-topology-v1",
        "room": {"width_m": 4, "length_m": 5, "height_m": 2.8},
        "channel": 6,
        "bandwidth_mhz": 20,
        "probe_rate_hz": 100,
        "nodes": [
            {
                "node_id": 1,
                "role": "tx",
                "mac": "28:84:85:92:81:3c",
                "position": {"x_m": 0, "y_m": 2.5, "z_m": 1.2},
            },
            {
                "node_id": 2,
                "role": "rx",
                "mac": "e0:72:a1:fd:19:0c",
                "position": {"x_m": 4, "y_m": 2.5, "z_m": 0.8},
            },
            {
                "node_id": 3,
                "role": "rx",
                "mac": "28:84:85:45:f1:28",
                "position": {"x_m": 2, "y_m": 5, "z_m": 1.5},
            },
        ],
        "zones": ["z6", "z2", "z1", "z4", "z5", "z3"],
        "environment": {"fan": "north", "curtain": "west"},
    }


def test_topology_id_is_canonical_and_verified(tmp_path: Path) -> None:
    first = topology()
    second = topology()
    second["nodes"] = list(reversed(second["nodes"]))
    second["zones"] = list(reversed(second["zones"]))
    assert topology_id(first) == topology_id(second)

    finalized = finalize_topology(first)
    path = tmp_path / "topology.json"
    path.write_text(json.dumps(finalized))
    assert load_topology(path)["topology_id"] == finalized["topology_id"]


def test_modified_topology_invalidates_id(tmp_path: Path) -> None:
    finalized = finalize_topology(topology())
    finalized["channel"] = 11
    path = tmp_path / "topology.json"
    path.write_text(json.dumps(finalized))
    with pytest.raises(TopologyError, match="does not match"):
        load_topology(path)
