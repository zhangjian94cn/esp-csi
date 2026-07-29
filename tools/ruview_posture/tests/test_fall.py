from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from ..fall import FallDetector, FallModel


class FallDetectorTest(unittest.TestCase):
    def test_fast_transition_to_sustained_lying_is_suspected(self) -> None:
        detector = FallDetector(motion_threshold=1.0)
        self.assertEqual(
            detector.update(
                now_ms=0, posture="standing", motion_energy=0.1
            ).event,
            "none",
        )
        detector.update(now_ms=250, posture="standing", motion_energy=1.5)
        detector.update(now_ms=500, posture="lying", motion_energy=0.5)
        decision = detector.update(
            now_ms=750, posture="lying", motion_energy=0.2
        )
        self.assertEqual(decision.event, "suspected")
        self.assertEqual(
            detector.update(
                now_ms=5000, posture="lying", motion_energy=0.1
            ).event,
            "suspected",
        )

    def test_slow_lying_is_not_a_fall(self) -> None:
        detector = FallDetector(motion_threshold=1.0)
        detector.update(now_ms=0, posture="standing", motion_energy=0.1)
        detector.update(now_ms=250, posture="sitting", motion_energy=0.3)
        detector.update(now_ms=2000, posture="lying", motion_energy=0.2)
        decision = detector.update(
            now_ms=2250, posture="lying", motion_energy=0.1
        )
        self.assertEqual(decision.event, "none")

    def test_fall_model_is_hash_bound_to_posture_model(self) -> None:
        model = FallModel.create(
            posture_model_id="posture-model",
            topology_id="room-a",
            motion_calibration_id="calibration",
            training_recordings=("a" * 64, "b" * 64),
            motion_threshold=0.2,
            transition_ms=1500,
            lying_confirmations=2,
            latch_ms=10_000,
            metrics={"fall_recall": 0.9},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fall-model.json"
            model.save(path)
            loaded = FallModel.load(path)
        self.assertEqual(loaded, model)
        self.assertEqual(
            loaded.validate_binding(
                posture_model_id="posture-model",
                topology_id="room-a",
                motion_calibration_id="calibration",
            ),
            [],
        )
        self.assertEqual(
            loaded.validate_binding(
                posture_model_id="other",
                topology_id="room-a",
                motion_calibration_id="calibration",
            ),
            ["posture_model_id"],
        )


if __name__ == "__main__":
    unittest.main()
