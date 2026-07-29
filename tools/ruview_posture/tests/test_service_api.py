from __future__ import annotations

import json
from pathlib import Path
import threading
from urllib.request import Request, urlopen

from tools.ruview_posture.experiment import ExperimentSessionManager
from tools.ruview_posture.service import Handler, ThreadingHTTPServer


class FakeState:
    def __init__(self, output: Path):
        self.experiment = ExperimentSessionManager(
            output_directory=output,
            topology_id="topology",
            required_nodes=(2, 3),
        )

    def topology_failures(self) -> list[str]:
        return []


def request_payload() -> dict:
    return {
        "occupancy": "present",
        "motion": "idle",
        "posture": "sitting",
        "event": "none",
        "zone_id": "zone-2",
        "fan_state": "off",
        "curtain_state": "off",
        "trial_id": "sitting-api-01",
        "dataset_role": "train",
        "countdown_seconds": 5,
        "duration_seconds": 30,
    }


def post(url: str, payload: dict) -> tuple[int, dict]:
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return response.status, json.load(response)


def test_experiment_api_start_status_cancel_is_idempotent(
    tmp_path: Path,
) -> None:
    Handler.state = FakeState(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, first = post(
            base + "/api/v1/experiments/session/start", request_payload()
        )
        status_again, second = post(
            base + "/api/v1/experiments/session/start", request_payload()
        )
        assert status == 201
        assert status_again == 200
        assert first["session_id"] == second["session_id"]

        with urlopen(base + "/api/v1/experiments/session/status") as response:
            current = json.load(response)
        assert current["state"] == "countdown"

        cancelled_status, cancelled = post(
            base + "/api/v1/experiments/session/cancel", {}
        )
        assert cancelled_status == 200
        assert cancelled["state"] == "cancelled"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
