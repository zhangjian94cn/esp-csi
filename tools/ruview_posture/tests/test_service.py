from __future__ import annotations

from pathlib import Path
import time

from tools.ruview_posture.model import FirmwareBinding
from tools.ruview_posture.service import LiveState
from tools.ruview_posture.tests.fixtures import status_datagram
from tools.ruview_posture.tests.test_train import topology_payload


def test_status_packets_gate_collection_topology(tmp_path: Path) -> None:
    topology = topology_payload()
    firmware = FirmwareBinding(
        fork_commit="test-build-commit",
        protocol_version=2,
        probe_rate_hz=100,
        build_ids={"1": "test-build", "2": "test-build", "3": "test-build"},
        artifact_sha256={"1": "1" * 64, "2": "2" * 64, "3": "3" * 64},
    )
    state = LiveState(
        predictor=None,
        topology=topology,
        firmware=firmware,
        acceptance={},
        event_log=tmp_path / "events.jsonl",
        recordings_directory=tmp_path / "recordings",
    )
    for node in (2, 3):
        state.ingest(
            time.time_ns(), status_datagram(node_id=node, build_id="test-build")
        )
    assert state.topology_failures() == []

    state.ingest(
        time.time_ns(), status_datagram(node_id=2, build_id="wrong-build")
    )
    assert "node_2_build_id" in state.topology_failures()
