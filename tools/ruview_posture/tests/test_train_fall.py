from __future__ import annotations

from tools.ruview_posture.train_fall import (
    FallTrial,
    _select_threshold,
)


def trial(index: int, *, expected_fall: bool) -> FallTrial:
    frames = (
        (
            (0, "standing", 0.05),
            (250, "standing", 2.0),
            (500, "lying", 0.4),
            (750, "lying", 0.2),
        )
        if expected_fall
        else (
            (0, "standing", 0.05),
            (250, "sitting", 0.2),
            (2000, "lying", 0.1),
            (2250, "lying", 0.05),
        )
    )
    return FallTrial(
        recording_hash=f"{index:064x}",
        expected_fall=expected_fall,
        frames=frames,
    )


def test_fall_threshold_training_is_independent() -> None:
    trials = [
        *(trial(index, expected_fall=True) for index in range(20)),
        *(trial(index + 100, expected_fall=False) for index in range(40)),
    ]
    threshold, metrics = _select_threshold(
        trials,
        transition_ms=1500,
        lying_confirmations=2,
        latch_ms=10_000,
    )
    assert threshold > 0
    assert metrics["fall_recall"] == 1.0
    assert metrics["nonfall_false_rate"] == 0.0
