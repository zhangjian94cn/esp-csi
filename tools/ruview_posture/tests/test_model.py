from __future__ import annotations

import unittest

from ..calibration import LinkMotionCalibration, MotionCalibration
from ..features import FEATURE_SCHEMA_HASH
from ..model import (
    ESP_WIFI_SENSING_VERSION,
    HEAD_CLASSES,
    OFFICIAL_ESP_CSI_COMMIT,
    FirmwareBinding,
    ModelManifest,
)
from ..protocol import PROTOCOL_VERSION


def firmware() -> FirmwareBinding:
    return FirmwareBinding(
        fork_commit="fork-commit",
        protocol_version=PROTOCOL_VERSION,
        probe_rate_hz=100,
        build_ids={"1": "build", "2": "build", "3": "build"},
        artifact_sha256={"1": "1" * 64, "2": "2" * 64, "3": "3" * 64},
    )


def motion_calibration() -> MotionCalibration:
    links = {
        str(node): LinkMotionCalibration(
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
        for node in (2, 3)
    }
    return MotionCalibration.create(
        topology_id="room-a",
        probe_rate_hz=100,
        empty_recordings=("a" * 64,),
        links=links,
    )


class ModelBindingTest(unittest.TestCase):
    def test_exact_binding_is_required(self) -> None:
        binding = firmware()
        calibration = motion_calibration()
        manifest = ModelManifest(
            schema="rvp-model-bundle-v2",
            model_id="m1",
            topology_id="room-a",
            feature_schema_hash=FEATURE_SCHEMA_HASH,
            protocol_version=PROTOCOL_VERSION,
            esp_csi_commit=OFFICIAL_ESP_CSI_COMMIT,
            esp_wifi_sensing_version=ESP_WIFI_SENSING_VERSION,
            fork_commit=binding.fork_commit,
            probe_rate_hz=binding.probe_rate_hz,
            firmware_build_ids=binding.build_ids,
            firmware_artifact_sha256=binding.artifact_sha256,
            training_recordings=("hash",),
            training_data_hash="0" * 64,
            calibration_id=calibration.calibration_id,
            motion_calibration=calibration.as_dict(),
            head_kinds={
                "presence": "logistic",
                "motion": "logistic",
                "posture": "logistic",
            },
            classes=HEAD_CLASSES,
            confidence_thresholds={
                "presence": 0.65,
                "motion": 0.65,
                "posture": 0.65,
            },
            fall_motion_threshold=2.0,
            metrics={},
        )
        self.assertEqual(
            manifest.validate_binding(
                topology_id="room-a", firmware=binding
            ),
            [],
        )
        self.assertEqual(
            manifest.validate_binding(
                topology_id="room-b", firmware=binding
            ),
            ["topology_id"],
        )


if __name__ == "__main__":
    unittest.main()
