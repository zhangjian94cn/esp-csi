from __future__ import annotations

import unittest

from ..features import FEATURE_SCHEMA_HASH
from ..model import (
    ESP_WIFI_SENSING_VERSION,
    OFFICIAL_ESP_CSI_COMMIT,
    POSTURE_CLASSES,
    ModelManifest,
)


class ModelBindingTest(unittest.TestCase):
    def test_exact_binding_is_required(self) -> None:
        manifest = ModelManifest(
            schema="rvp-posture-model-v1",
            model_id="m1",
            model_kind="logistic",
            topology_id="room-a",
            feature_schema_hash=FEATURE_SCHEMA_HASH,
            esp_csi_commit=OFFICIAL_ESP_CSI_COMMIT,
            esp_wifi_sensing_version=ESP_WIFI_SENSING_VERSION,
            training_recordings=("hash",),
            classes=POSTURE_CLASSES,
            confidence_threshold=0.65,
            motion_threshold=1.0,
            fall_motion_threshold=2.0,
            metrics={},
        )
        self.assertEqual(
            manifest.validate_binding(topology_id="room-a"), []
        )
        self.assertEqual(
            manifest.validate_binding(topology_id="room-b"), ["topology_id"]
        )


if __name__ == "__main__":
    unittest.main()
