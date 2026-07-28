from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ..features import (
    FEATURE_SCHEMA_HASH,
    extract_recording_windows,
    feature_names,
)
from ..recording import RecordingWriter
from .fixtures import csi_datagram


def synthetic_csi(node: int, sequence: int) -> bytes:
    values = bytearray()
    for subcarrier in range(32):
        phase = sequence * 0.03 + subcarrier * 0.1 + node * 0.2
        amplitude = 20 + node + math.sin(sequence * 0.05) * 3
        imaginary = int(max(-127, min(127, amplitude * math.sin(phase))))
        real = int(max(-127, min(127, amplitude * math.cos(phase))))
        values.extend((imaginary & 0xFF, real & 0xFF))
    return bytes(values)


class FeaturesTest(unittest.TestCase):
    def test_two_link_features_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.rvp"
            with RecordingWriter(
                path,
                {
                    "dataset_role": "train",
                    "session_id": "s1",
                    "trial_id": "t1",
                    "label": "standing",
                },
            ) as writer:
                for sequence in range(130):
                    host_ns = sequence * 20_000_000
                    for node in (2, 3):
                        writer.append(
                            host_ns,
                            csi_datagram(
                                node_id=node,
                                sequence=sequence,
                                timestamp_us=host_ns // 1000,
                                csi=synthetic_csi(node, sequence),
                            ),
                        )
            first = extract_recording_windows(path)
            second = extract_recording_windows(path)
            self.assertGreater(len(first), 0)
            self.assertEqual(len(first[0].vector), len(feature_names()))
            np.testing.assert_array_equal(first[0].vector, second[0].vector)
            self.assertEqual(len(FEATURE_SCHEMA_HASH), 64)
            self.assertEqual(first[0].node_ids, (2, 3))


if __name__ == "__main__":
    unittest.main()
