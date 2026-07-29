from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from ..calibration import MotionCalibration, calibrate_motion
from ..recording import RecordingWriter
from .fixtures import csi_datagram


def quiet_csi(node: int, sequence: int) -> bytes:
    values = bytearray()
    for subcarrier in range(32):
        amplitude = 18 + (subcarrier % 9)
        amplitude += math.sin(sequence * 0.02 + subcarrier * 0.11) * 0.7
        phase = sequence * 0.01 + subcarrier * 0.07 + node * 0.03
        imaginary = int(amplitude * math.sin(phase))
        real = int(amplitude * math.cos(phase))
        values.extend((imaginary & 0xFF, real & 0xFF))
    return bytes(values)


class MotionCalibrationTest(unittest.TestCase):
    def test_nbvi_calibration_is_fixed_deterministic_and_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.rvp"
            with RecordingWriter(
                path,
                {
                    "schema": "rvp-recording-v2",
                    "dataset_role": "train",
                    "occupancy": "absent",
                },
            ) as writer:
                for sequence in range(420):
                    host_ns = sequence * 10_000_000
                    for node in (2, 3):
                        writer.append(
                            host_ns,
                            csi_datagram(
                                node_id=node,
                                sequence=sequence,
                                timestamp_us=host_ns // 1000,
                                csi=quiet_csi(node, sequence),
                            ),
                        )

            first = calibrate_motion(
                [path], topology_id="room-a", probe_rate_hz=100
            )
            second = calibrate_motion(
                [path], topology_id="room-a", probe_rate_hz=100
            )
            self.assertEqual(first, second)
            self.assertEqual(
                MotionCalibration.from_dict(first.as_dict()),
                first,
            )
            self.assertEqual(set(first.links), {"2", "3"})
            for link in first.links.values():
                self.assertEqual(len(link.subcarrier_indices), 12)
                self.assertTrue(
                    all(
                        right - left > 1
                        for left, right in zip(
                            link.subcarrier_indices,
                            link.subcarrier_indices[1:],
                        )
                    )
                )
                self.assertLessEqual(link.baseline_false_positive_rate, 0.05)
                self.assertGreater(link.baseline_threshold, 0)

    def test_calibration_content_tampering_is_rejected(self) -> None:
        calibration = MotionCalibration.create(
            topology_id="room-a",
            probe_rate_hz=100,
            empty_recordings=("a" * 64,),
            links={
                str(node): self._link(node)
                for node in (2, 3)
            },
        )
        payload = calibration.as_dict()
        payload["links"]["2"]["baseline_threshold"] = 9.0
        with self.assertRaisesRegex(ValueError, "ID does not match"):
            MotionCalibration.from_dict(payload)

    @staticmethod
    def _link(node: int):
        from ..calibration import LinkMotionCalibration

        return LinkMotionCalibration(
            node_id=node,
            subcarrier_indices=(
                12,
                14,
                16,
                18,
                20,
                24,
                28,
                36,
                40,
                44,
                48,
                52,
            ),
            baseline_threshold=0.01,
            baseline_false_positive_rate=0.01,
            valid_frames=500,
        )


if __name__ == "__main__":
    unittest.main()
