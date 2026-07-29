from __future__ import annotations

from argparse import Namespace
import json
import math
from pathlib import Path
import tempfile
import unittest

from ..labels import TrialLabels
from ..model import ModelManifest
from ..recording import RecordingWriter
from ..topology import finalize_topology
from ..train import train
from .fixtures import csi_datagram


def labeled_csi(label_index: int, node: int, sequence: int) -> bytes:
    values = bytearray()
    for subcarrier in range(32):
        amplitude = 18 + ((subcarrier + label_index * 5) % 13)
        phase = sequence * (0.02 + label_index * 0.006)
        phase += subcarrier * (0.04 + label_index * 0.015) + node * 0.1
        imaginary = int(amplitude * math.sin(phase))
        real = int(amplitude * math.cos(phase))
        values.extend((imaginary & 0xFF, real & 0xFF))
    return bytes(values)


def topology_payload() -> dict:
    return finalize_topology(
        {
            "schema": "rvp-topology-v1",
            "room": {"width_m": 4, "length_m": 5, "height_m": 2.8},
            "channel": 6,
            "bandwidth_mhz": 20,
            "probe_rate_hz": 100,
            "nodes": [
                {
                    "node_id": 1,
                    "role": "tx",
                    "mac": "28:84:85:92:81:3c",
                    "position": {"x_m": 0, "y_m": 2.5, "z_m": 1.2},
                },
                {
                    "node_id": 2,
                    "role": "rx",
                    "mac": "e0:72:a1:fd:19:0c",
                    "position": {"x_m": 4, "y_m": 2.5, "z_m": 0.8},
                },
                {
                    "node_id": 3,
                    "role": "rx",
                    "mac": "28:84:85:45:f1:28",
                    "position": {"x_m": 2, "y_m": 5, "z_m": 1.5},
                },
            ],
            "zones": [f"z{index}" for index in range(1, 7)],
            "environment": {"fan": "north", "curtain": "west"},
        }
    )


class TrainingTest(unittest.TestCase):
    def test_grouped_logistic_training_writes_bound_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            topology = topology_payload()
            topology_path = root / "topology.json"
            topology_path.write_text(json.dumps(topology))
            firmware_path = root / "firmware.json"
            firmware_path.write_text(
                json.dumps(
                    {
                        "schema": "rvp-firmware-binding-v1",
                        "fork_commit": "test-fork",
                        "protocol_version": 2,
                        "probe_rate_hz": 100,
                        "build_ids": {
                            "1": "test-build",
                            "2": "test-build",
                            "3": "test-build",
                        },
                        "artifact_sha256": {
                            "1": "1" * 64,
                            "2": "2" * 64,
                            "3": "3" * 64,
                        },
                    }
                )
            )
            recordings: list[Path] = []
            labels = (
                ("absent", "idle", "unknown"),
                ("present", "idle", "standing"),
                ("present", "idle", "sitting"),
                ("present", "idle", "lying"),
                ("present", "moving", "unknown"),
            )
            for label_index, (occupancy, motion, posture) in enumerate(labels):
                for trial in range(2):
                    trial_id = f"{occupancy}-{motion}-{posture}-{trial}"
                    path = root / f"{trial_id}.rvp"
                    trial_labels = TrialLabels(
                        occupancy=occupancy,
                        motion=motion,
                        posture=posture,
                        event="none",
                        zone_id="z1",
                        fan_state="off",
                        curtain_state="off",
                        trial_id=trial_id,
                        dataset_role="train",
                    )
                    with RecordingWriter(
                        path,
                        {
                            "schema": "rvp-recording-v2",
                            "session_id": "session",
                            "topology_id": topology["topology_id"],
                            **trial_labels.as_metadata(),
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
                    topology=topology_path,
                    firmware_binding=firmware_path,
                    confidence_threshold=0.65,
                    enable_tcn=False,
                )
            )
            self.assertEqual(result, 0)
            manifest = ModelManifest.load(output / "manifest.json")
            self.assertEqual(manifest.topology_id, topology["topology_id"])
            self.assertEqual(
                manifest.head_kinds,
                {
                    "presence": "logistic",
                    "motion": "logistic",
                    "posture": "logistic",
                },
            )
            self.assertTrue((output / "model-bundle.joblib").exists())


if __name__ == "__main__":
    unittest.main()
