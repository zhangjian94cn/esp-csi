from __future__ import annotations

import unittest

from ..fall import FallDetector


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


if __name__ == "__main__":
    unittest.main()
