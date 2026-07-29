from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from tools.ruview_posture.espectre_evaluate import (
    evaluate_algorithm,
    select_baseline,
)
from tools.ruview_posture.espectre_recording import (
    EspectreRecordingWriter,
    EspectreSample,
    read_espectre_recording,
)


def _write_trial(
    path: Path,
    *,
    algorithm: str,
    expected_motion: bool,
    scenario: str,
    transition: str = "none",
    latency_samples: int = 0,
) -> None:
    metadata = {
        "algorithm": algorithm,
        "firmware_release": "2.8.0",
        "expected_motion": expected_motion,
        "scenario": scenario,
        "transition": transition,
        "warmup_seconds": 0,
    }
    with EspectreRecordingWriter(path, metadata) as writer:
        for index in range(20):
            if transition == "enter":
                motion = index >= latency_samples
            elif transition == "exit":
                motion = index < latency_samples
            else:
                motion = expected_motion
            writer.append(
                EspectreSample(
                    monotonic_ns=index * 100_000_000,
                    wall_time=datetime.now(timezone.utc).isoformat(),
                    connected=True,
                    motion=motion,
                    movement_score=2.0 if motion else 0.1,
                    threshold=1.0,
                    event="heartbeat",
                )
            )


def test_espectre_recording_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "trial.jsonl"
    _write_trial(
        path,
        algorithm="mvs",
        expected_motion=False,
        scenario="empty_fan_off",
    )
    metadata, samples = read_espectre_recording(path)
    assert metadata["algorithm"] == "mvs"
    assert len(samples) == 20
    assert not samples[0].motion


def test_evaluate_and_select_motion_baseline(tmp_path: Path) -> None:
    reports = []
    for algorithm, latency in (("mvs", 5), ("ml", 8)):
        paths: list[Path] = []
        for index in range(10):
            entry = tmp_path / f"{algorithm}-entry-{index}.jsonl"
            exit_ = tmp_path / f"{algorithm}-exit-{index}.jsonl"
            _write_trial(
                entry,
                algorithm=algorithm,
                expected_motion=True,
                scenario="enter",
                transition="enter",
                latency_samples=latency,
            )
            _write_trial(
                exit_,
                algorithm=algorithm,
                expected_motion=False,
                scenario="exit",
                transition="exit",
                latency_samples=latency,
            )
            paths.extend((entry, exit_))
        empty = tmp_path / f"{algorithm}-empty.jsonl"
        moving = tmp_path / f"{algorithm}-moving.jsonl"
        _write_trial(
            empty,
            algorithm=algorithm,
            expected_motion=False,
            scenario="empty_fan_off",
        )
        _write_trial(
            moving,
            algorithm=algorithm,
            expected_motion=True,
            scenario="moving",
        )
        paths.extend((empty, moving))
        reports.append(
            evaluate_algorithm(paths, expected_algorithm=algorithm)
        )

    assert all(report["passed"] for report in reports)
    assert select_baseline(reports) == "mvs"
