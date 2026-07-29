"""Train grouped Presence, Motion, and Posture heads without trial leakage."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import MotionCalibration, calibrate_motion
from .features import FEATURE_SCHEMA_HASH, extract_recording_windows, vector_for_head
from .labels import labels_from_metadata
from .model import (
    ESP_WIFI_SENSING_VERSION,
    HEAD_CLASSES,
    OFFICIAL_ESP_CSI_COMMIT,
    FirmwareBinding,
    ModelManifest,
    aggregate_hash,
    derive_model_id,
    macro_metrics,
    recording_sha256,
)
from .protocol import PROTOCOL_VERSION
from .recording import read_metadata
from .tcn import build_sequences, fit_tcn, predict_tcn
from .topology import load_topology


@dataclass(frozen=True)
class HeadData:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray


def _load_training_data(
    recordings: list[Path],
    calibration: MotionCalibration,
) -> tuple[dict[str, HeadData], list[str]]:
    vectors: dict[str, list[np.ndarray]] = {head: [] for head in HEAD_CLASSES}
    targets: dict[str, list[str]] = {head: [] for head in HEAD_CLASSES}
    groups: dict[str, list[str]] = {head: [] for head in HEAD_CLASSES}
    hashes: list[str] = []

    for recording in recordings:
        metadata = read_metadata(recording)
        labels = labels_from_metadata(metadata)
        if labels.dataset_role != "train":
            raise ValueError(f"{recording} is not marked as training data")
        windows = extract_recording_windows(
            recording,
            motion_subcarriers=calibration.subcarriers(),
        )
        if not windows:
            raise ValueError(f"{recording} contains no valid two-link windows")
        recording_hash = recording_sha256(recording)
        hashes.append(recording_hash)
        group = f"{metadata.get('session_id')}:{labels.trial_id}"

        for window in windows:
            vectors["presence"].append(vector_for_head(window, "presence"))
            targets["presence"].append(labels.occupancy)
            groups["presence"].append(group)

            if labels.occupancy == "present" and labels.motion != "unknown":
                vectors["motion"].append(vector_for_head(window, "motion"))
                targets["motion"].append(
                    "moving" if labels.motion == "moving" else "still"
                )
                groups["motion"].append(group)

            if labels.occupancy == "present" and labels.posture != "unknown":
                vectors["posture"].append(vector_for_head(window, "posture"))
                targets["posture"].append(labels.posture)
                groups["posture"].append(group)

    result: dict[str, HeadData] = {}
    for head in HEAD_CLASSES:
        if not vectors[head]:
            raise ValueError(f"{head} head has no training windows")
        y = np.asarray(targets[head])
        if set(y) != set(HEAD_CLASSES[head]):
            raise ValueError(
                f"{head} head requires classes {HEAD_CLASSES[head]}, got {sorted(set(y))}"
            )
        result[head] = HeadData(
            x=np.stack(vectors[head]),
            y=y,
            groups=np.asarray(groups[head]),
        )
    return result, hashes


def _logistic_candidate(
    data: HeadData, classes: tuple[str, ...]
) -> tuple[Pipeline, dict[str, Any]]:
    groups_per_class = {
        label: len(np.unique(data.groups[data.y == label]))
        for label in classes
    }
    if min(groups_per_class.values()) < 2:
        raise ValueError(
            "each class requires at least two independent trials"
        )
    folds = min(5, min(groups_per_class.values()))
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    predicted = cross_val_predict(
        model,
        data.x,
        data.y,
        cv=StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=42,
        ),
        groups=data.groups,
        method="predict",
    )
    metrics = macro_metrics(data.y, predicted, classes)
    model.fit(data.x, data.y)
    return model, metrics


def _tcn_candidate(
    data: HeadData, classes: tuple[str, ...]
) -> tuple[Any, dict[str, Any]]:
    sequences, labels, sequence_groups = build_sequences(
        data.x, data.y, data.groups
    )
    groups_per_class = {
        label: len(np.unique(sequence_groups[labels == label]))
        for label in classes
    }
    if min(groups_per_class.values()) < 2:
        raise ValueError(
            "each TCN class requires at least two independent trials"
        )
    predictions = np.empty_like(labels)
    folds = StratifiedGroupKFold(
        n_splits=min(5, min(groups_per_class.values())),
        shuffle=True,
        random_state=42,
    )
    for train_indexes, test_indexes in folds.split(
        sequences, labels, sequence_groups
    ):
        artifact = fit_tcn(
            sequences[train_indexes],
            labels[train_indexes],
            classes=classes,
        )
        predictions[test_indexes], _ = predict_tcn(
            artifact, sequences[test_indexes]
        )
    metrics = macro_metrics(labels, predictions, classes)
    artifact = fit_tcn(sequences, labels, classes=classes)
    return artifact, metrics


def train(args: argparse.Namespace) -> int:
    topology = load_topology(args.topology)
    firmware = FirmwareBinding.load(args.firmware_binding)
    if topology["probe_rate_hz"] != firmware.probe_rate_hz:
        raise ValueError("topology and firmware probe rates do not match")
    empty_recordings = [
        recording
        for recording in args.recordings
        if labels_from_metadata(read_metadata(recording)).occupancy == "absent"
    ]
    calibration = calibrate_motion(
        empty_recordings,
        topology_id=topology["topology_id"],
        probe_rate_hz=firmware.probe_rate_hz,
    )
    data, recording_hashes = _load_training_data(
        args.recordings,
        calibration,
    )

    artifacts: dict[str, Any] = {}
    head_kinds: dict[str, str] = {}
    metrics: dict[str, Any] = {}
    for head, classes in HEAD_CLASSES.items():
        logistic, logistic_metrics = _logistic_candidate(data[head], classes)
        selected = logistic
        selected_kind = "logistic"
        candidates: dict[str, Any] = {"logistic": logistic_metrics}
        if args.enable_tcn:
            tcn, tcn_metrics = _tcn_candidate(data[head], classes)
            candidates["tcn"] = tcn_metrics
            if (
                tcn_metrics["macro_f1"]
                >= logistic_metrics["macro_f1"] + 0.03
                and min(tcn_metrics["recall"].values())
                >= min(logistic_metrics["recall"].values())
            ):
                selected = tcn
                selected_kind = "tcn"
        artifacts[head] = selected
        head_kinds[head] = selected_kind
        metrics[head] = {
            "selected": selected_kind,
            "candidates": candidates,
        }

    calibration_id = calibration.calibration_id
    args.output.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output / "model-bundle.joblib"
    joblib.dump(artifacts, artifact_path)
    model_artifact_sha256 = recording_sha256(artifact_path)
    model_id = derive_model_id(
        topology_id=topology["topology_id"],
        head_kinds=head_kinds,
        recording_hashes=recording_hashes,
        calibration_id=calibration_id,
        firmware=firmware,
        model_artifact_sha256=model_artifact_sha256,
    )
    manifest = ModelManifest(
        schema="rvp-model-bundle-v2",
        model_id=model_id,
        topology_id=topology["topology_id"],
        feature_schema_hash=FEATURE_SCHEMA_HASH,
        protocol_version=PROTOCOL_VERSION,
        esp_csi_commit=OFFICIAL_ESP_CSI_COMMIT,
        esp_wifi_sensing_version=ESP_WIFI_SENSING_VERSION,
        fork_commit=firmware.fork_commit,
        probe_rate_hz=firmware.probe_rate_hz,
        firmware_build_ids=firmware.build_ids,
        firmware_artifact_sha256=firmware.artifact_sha256,
        model_artifact_sha256=model_artifact_sha256,
        training_recordings=tuple(sorted(recording_hashes)),
        training_data_hash=aggregate_hash(recording_hashes),
        calibration_id=calibration_id,
        motion_calibration=calibration.as_dict(),
        head_kinds=head_kinds,
        classes=HEAD_CLASSES,
        confidence_thresholds={
            head: args.confidence_threshold for head in HEAD_CLASSES
        },
        metrics=metrics,
    )
    manifest.save(args.output / "manifest.json")
    print(
        json.dumps(
            {
                "model_id": model_id,
                "head_kinds": head_kinds,
                "artifact": str(artifact_path),
                "calibration_id": calibration_id,
                "metrics": metrics,
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--firmware-binding", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--enable-tcn", action="store_true")
    return parser


def main() -> int:
    return train(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
