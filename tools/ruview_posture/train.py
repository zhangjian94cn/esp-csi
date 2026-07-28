"""Train grouped posture candidates without leaking adjacent windows."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_SCHEMA_HASH, extract_recording_windows
from .model import (
    ESP_WIFI_SENSING_VERSION,
    OFFICIAL_ESP_CSI_COMMIT,
    POSTURE_CLASSES,
    ModelManifest,
    derive_model_id,
    macro_metrics,
    recording_sha256,
)
from .recording import read_metadata
from .tcn import build_sequences, fit_tcn, predict_tcn


def _load_training_data(
    recordings: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    motion: list[float] = []
    hashes: list[str] = []
    for recording in recordings:
        metadata = read_metadata(recording)
        if metadata.get("dataset_role") != "train":
            raise ValueError(f"{recording} is not marked as a training recording")
        source_label = str(metadata.get("label"))
        if source_label in {"moving", "slow_lying"}:
            posture_label = "lying" if source_label == "slow_lying" else None
        elif source_label == "fall":
            posture_label = "lying"
        else:
            posture_label = source_label
        windows = extract_recording_windows(recording)
        if not windows:
            raise ValueError(f"{recording} contains no valid two-link windows")
        hashes.append(recording_sha256(recording))
        if posture_label is None:
            continue
        if posture_label not in POSTURE_CLASSES:
            raise ValueError(f"unsupported posture label {posture_label}")
        group = f"{metadata.get('session_id')}:{metadata.get('trial_id')}"
        for window in windows:
            vectors.append(window.vector)
            labels.append(posture_label)
            groups.append(group)
            motion.append(window.motion_energy)
    return (
        np.stack(vectors),
        np.asarray(labels),
        np.asarray(groups),
        np.asarray(motion),
        hashes,
    )


def _logistic_candidate(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[Pipeline, dict[str, Any]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two independent trials are required")
    folds = min(5, len(unique_groups))
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
        x,
        y,
        cv=GroupKFold(n_splits=folds),
        groups=groups,
        method="predict",
    )
    metrics = macro_metrics(y, predicted)
    model.fit(x, y)
    return model, metrics


def _tcn_candidate(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[Any, dict[str, Any]]:
    sequences, labels, sequence_groups = build_sequences(x, y, groups)
    unique_groups = np.unique(sequence_groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two sequence groups are required for TCN")
    predictions = np.empty_like(labels)
    folds = GroupKFold(n_splits=min(5, len(unique_groups)))
    for train_indexes, test_indexes in folds.split(
        sequences, labels, sequence_groups
    ):
        artifact = fit_tcn(
            sequences[train_indexes],
            labels[train_indexes],
            classes=POSTURE_CLASSES,
        )
        predictions[test_indexes], _ = predict_tcn(
            artifact, sequences[test_indexes]
        )
    metrics = macro_metrics(labels, predictions)
    artifact = fit_tcn(sequences, labels, classes=POSTURE_CLASSES)
    return artifact, metrics


def train(args: argparse.Namespace) -> int:
    x, y, groups, motion, recording_hashes = _load_training_data(args.recordings)
    logistic, logistic_metrics = _logistic_candidate(x, y, groups)
    selected_kind = "logistic"
    selected_model: Any = logistic
    metrics: dict[str, Any] = {"logistic": logistic_metrics}

    if args.enable_tcn:
        tcn, tcn_metrics = _tcn_candidate(x, y, groups)
        metrics["tcn"] = tcn_metrics
        if (
            tcn_metrics["macro_f1"] >= logistic_metrics["macro_f1"] + 0.03
            and min(tcn_metrics["recall"].values())
            >= min(logistic_metrics["recall"].values())
        ):
            selected_kind = "tcn"
            selected_model = tcn

    nonfall_motion: list[float] = []
    fall_motion: list[float] = []
    for path in args.recordings:
        label = read_metadata(path).get("label")
        values = [
            window.motion_energy for window in extract_recording_windows(path)
        ]
        if label == "fall":
            fall_motion.extend(values)
        else:
            nonfall_motion.extend(values)
    motion_threshold = float(np.percentile(motion, 75))
    nonfall_limit = (
        float(np.percentile(nonfall_motion, 99))
        if nonfall_motion
        else motion_threshold
    )
    if fall_motion:
        fall_floor = float(np.percentile(fall_motion, 20))
        fall_threshold = (nonfall_limit + fall_floor) / 2.0
    else:
        fall_threshold = nonfall_limit * 1.5

    model_id = derive_model_id(
        topology_id=args.topology_id,
        model_kind=selected_kind,
        recording_hashes=recording_hashes,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output / (
        "model.joblib" if selected_kind == "logistic" else "model-tcn.joblib"
    )
    joblib.dump(selected_model, artifact_path)
    manifest = ModelManifest(
        schema="rvp-posture-model-v1",
        model_id=model_id,
        model_kind=selected_kind,
        topology_id=args.topology_id,
        feature_schema_hash=FEATURE_SCHEMA_HASH,
        esp_csi_commit=OFFICIAL_ESP_CSI_COMMIT,
        esp_wifi_sensing_version=ESP_WIFI_SENSING_VERSION,
        training_recordings=tuple(recording_hashes),
        classes=POSTURE_CLASSES,
        confidence_threshold=args.confidence_threshold,
        motion_threshold=motion_threshold,
        fall_motion_threshold=fall_threshold,
        metrics=metrics,
    )
    manifest.save(args.output / "manifest.json")
    print(
        json.dumps(
            {
                "model_id": model_id,
                "model_kind": selected_kind,
                "artifact": str(artifact_path),
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
    parser.add_argument("--topology-id", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--enable-tcn", action="store_true")
    return parser


def main() -> int:
    return train(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
