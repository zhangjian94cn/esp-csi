from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from ..protocol import CsiPacket
from ..recording import RecordingWriter, iter_recording, read_metadata
from .fixtures import csi_datagram


class RecordingTest(unittest.TestCase):
    def test_atomic_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trial.rvp"
            with RecordingWriter(
                output,
                {
                    "session_id": "s1",
                    "trial_id": "t1",
                    "label": "standing",
                },
            ) as writer:
                writer.append(
                    10,
                    csi_datagram(node_id=2, sequence=1, csi=b"\x01\x02" * 32),
                )
                self.assertFalse(output.exists())
            self.assertTrue(output.exists())
            self.assertFalse(output.with_suffix(".rvp.partial").exists())
            self.assertEqual(read_metadata(output)["label"], "standing")
            records = list(iter_recording(output))
            self.assertEqual(len(records), 1)
            self.assertIsInstance(records[0].packet, CsiPacket)


if __name__ == "__main__":
    unittest.main()
