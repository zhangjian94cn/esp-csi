from __future__ import annotations

from argparse import Namespace
import math
from pathlib import Path
import tempfile
import unittest

from ..model import ModelManifest
from ..recording import RecordingWriter
from ..train import train
from .fixtures import csi_datagram


def labeled_csi(label_index: int, node: int, sequence: int) -> bytes:
    values = bytearray()
    for subcarrier in range(32):
        amplitude = 18 + ((subcarrier + label_index * 5) % 13)
        phase = sequence * 0.02 + subcarrier * (0.04 + label_index * 0.015)
        phase += node * 0.1
        imaginary = int(amplitude * math.sin(phase))
        real = int(amplitude * math.cos(phase))
        values.extend((imaginary & 0xFF, real & 0xFF))
    return bytes(values)


class TrainingTest(unittest.TestCase):
    def test_grouped_logistic_training_writes_bound_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recordings: list[Path] = []
            labels = ("absent", "standing", "sitting", "lying")
            for label_index, label in enumerate(labels):
                for trial in range(2):
                    path = root / f"{label}-{trial}.rvp"
                    with RecordingWriter(
                        path,
                        {
                            "dataset_role": "train",
                            "session_id": "session",
                            "trial_id": f"{label}-{trial}",
                            "label": label,
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
                                        csi=labeled_csi(
                                            label_index, node, sequence
                                        ),
                                    ),
                                )
                    recordings.append(path)
            output = root / "model"
            result = train(
                Namespace(
                    recordings=recordings,
                    output=output,
                    topology_id="room-test",
                    confidence_threshold=0.65,
                    enable_tcn=False,
                )
            )
            self.assertEqual(result, 0)
            manifest = ModelManifest.load(output / "manifest.json")
            self.assertEqual(manifest.topology_id, "room-test")
            self.assertEqual(manifest.model_kind, "logistic")
            self.assertTrue((output / "model.joblib").exists())


if __name__ == "__main__":
    unittest.main()
