from __future__ import annotations

from pathlib import Path
import time

from tools.ruview_posture.model import FirmwareBinding
from tools.ruview_posture.service import LiveState
from tools.ruview_posture.tests.fixtures import csi_datagram, status_datagram
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


def test_recording_rejects_csi_outside_topology(tmp_path: Path) -> None:
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
    state.experiment.start(
        {
            "occupancy": "absent",
            "motion": "idle",
            "posture": "unknown",
            "event": "none",
            "zone_id": "zone-1",
            "fan_state": "off",
            "curtain_state": "off",
            "trial_id": "topology-gate",
            "dataset_role": "train",
            "countdown_seconds": 0,
            "duration_seconds": 30,
        }
    )
    payload = bytes((1, 1)) * 32
    state.ingest(
        time.time_ns(),
        csi_datagram(node_id=4, sequence=1, csi=payload),
    )
    state.ingest(
        time.time_ns(),
        csi_datagram(node_id=2, sequence=1, csi=payload),
    )
    snapshot = state.experiment.snapshot()
    assert snapshot["valid_frames"] == {"2": 1, "3": 0}
    state.experiment.cancel()
