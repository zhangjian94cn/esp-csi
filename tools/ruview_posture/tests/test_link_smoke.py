from __future__ import annotations

from tools.ruview_posture.link_smoke import (
    baseline_report,
    transition_report,
)


def payload(*, valid: bool, fps: float = 92.0) -> dict:
    return {
        "valid": valid,
        "link_quality": [
            {
                "rx_node_id": node,
                "valid": valid,
                "fps": fps,
                "loss_rate": 0.01,
                "structure_stability": 0.99,
            }
            for node in (2, 3)
        ],
    }


def test_baseline_requires_both_healthy_links() -> None:
    samples = [(index * 0.25, payload(valid=True)) for index in range(240)]
    report = baseline_report(
        samples,
        request_failures=0,
        probe_rate_hz=100,
    )
    assert report["passed"]
    assert report["links"]["2"]["fps_p5"] == 92.0

    samples[-20:] = [
        (index * 0.25, payload(valid=False))
        for index in range(220, 240)
    ]
    report = baseline_report(
        samples,
        request_failures=0,
        probe_rate_hz=100,
    )
    assert not report["passed"]


def test_tx_off_and_reconnect_transition_limits() -> None:
    tx_off = [
        (0.0, payload(valid=True)),
        (1.5, payload(valid=False)),
    ]
    assert transition_report(
        tx_off, request_failures=0, phase="tx_off"
    )["passed"]

    reconnect = [
        (0.0, payload(valid=False)),
        (8.0, payload(valid=True)),
    ]
    assert transition_report(
        reconnect, request_failures=0, phase="reconnect"
    )["passed"]
