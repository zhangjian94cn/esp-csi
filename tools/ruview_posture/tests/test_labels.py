from __future__ import annotations

import pytest

from tools.ruview_posture.labels import TrialLabels, labels_from_metadata


def test_independent_labels_roundtrip() -> None:
    labels = TrialLabels(
        occupancy="present",
        motion="idle",
        posture="sitting",
        event="none",
        zone_id="zone-2",
        fan_state="off",
        curtain_state="on",
        trial_id="sitting-02",
        dataset_role="train",
    )
    assert labels_from_metadata(labels.as_metadata()) == labels


def test_absent_cannot_claim_posture() -> None:
    labels = TrialLabels(
        occupancy="absent",
        motion="idle",
        posture="standing",
        event="none",
        zone_id="empty",
        fan_state="off",
        curtain_state="off",
        trial_id="bad",
        dataset_role="train",
    )
    with pytest.raises(ValueError, match="posture=unknown"):
        labels.validate()


def test_legacy_label_is_read_only_compatible() -> None:
    labels = labels_from_metadata(
        {
            "label": "moving",
            "trial_id": "legacy-1",
            "dataset_role": "train",
        }
    )
    assert labels.occupancy == "present"
    assert labels.motion == "moving"
    assert labels.posture == "unknown"
