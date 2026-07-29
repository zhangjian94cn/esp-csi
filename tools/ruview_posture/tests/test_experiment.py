from __future__ import annotations

from pathlib import Path

import pytest

from tools.ruview_posture.experiment import (
    ExperimentSessionManager,
    SessionConflict,
)
from tools.ruview_posture.protocol import decode_packet
from tools.ruview_posture.tests.fixtures import csi_datagram


def request() -> dict:
    return {
        "occupancy": "present",
        "motion": "idle",
        "posture": "standing",
        "event": "none",
        "zone_id": "zone-1",
        "fan_state": "off",
        "curtain_state": "off",
        "trial_id": "standing-01",
        "dataset_role": "train",
        "countdown_seconds": 0,
        "duration_seconds": 30,
    }


def test_session_start_is_idempotent_and_records_both_links(
    tmp_path: Path,
) -> None:
    manager = ExperimentSessionManager(
        output_directory=tmp_path,
        topology_id="topology",
        required_nodes=(2, 3),
    )
    first, created = manager.start(request())
    second, created_again = manager.start(request())
    assert created
    assert not created_again
    assert first["session_id"] == second["session_id"]

    for node in (2, 3):
        datagram = csi_datagram(
            node_id=node, sequence=1, csi=b"\x01\x02" * 32
        )
        manager.ingest(1, 1, datagram, decode_packet(datagram))
    completed = manager.stop()
    assert completed["state"] == "completed"
    assert Path(completed["output"]).is_file()


def test_stop_rejects_missing_link_and_cancel_removes_partial(
    tmp_path: Path,
) -> None:
    manager = ExperimentSessionManager(
        output_directory=tmp_path,
        topology_id="topology",
        required_nodes=(2, 3),
    )
    manager.start(request())
    datagram = csi_datagram(node_id=2, sequence=1, csi=b"\x01\x02" * 32)
    manager.ingest(1, 1, datagram, decode_packet(datagram))
    with pytest.raises(SessionConflict, match="node"):
        manager.stop()
    cancelled = manager.cancel()
    assert cancelled["state"] == "cancelled"
    assert not Path(cancelled["output"]).exists()
