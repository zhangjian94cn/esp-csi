"""Train the post-posture fall detector without changing the three-head model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .fall import FallDetector, FallModel
from .features import extract_recording_windows
from .inference import PosturePredictor
from .labels import labels_from_metadata
from .model import FirmwareBinding, recording_sha256
from .recording import read_metadata
from .topology import load_topology


@dataclass(frozen=True)
class FallTrial:
    recording_hash: str
    expected_fall: bool
    frames: tuple[tuple[int, str, float], ...]


def _load_trials(
    recordings: list[Path],
    *,
    predictor: PosturePredictor,
    topology_id: str,
) -> list[FallTrial]:
    trials: list[FallTrial] = []
    bands = predictor.motion_calibration.subcarriers()
    for recording in recordings:
        metadata = read_metadata(recording)
        labels = labels_from_metadata(metadata)
        if labels.dataset_role != "train":
            raise ValueError(f"{recording} is not marked as training data")
        if metadata.get("topology_id") != topology_id:
            raise ValueError(f"{recording} topology does not match")
        if labels.occupancy != "present":
            continue
        windows = extract_recording_windows(
            recording,
            motion_subcarriers=bands,
        )
        if not windows:
            raise ValueError(f"{recording} contains no valid two-link windows")
        predictor.reset()
        frames = tuple(
            (
                window.end_ns // 1_000_000,
                predictor.predict(window).posture.label,
                window.motion_energy,
            )
            for window in windows
        )
        trials.append(
            FallTrial(
                recording_hash=recording_sha256(recording),
                expected_fall=labels.event == "fall",
                frames=frames,
            )
        )
    return trials


def _trial_alerts(
    trial: FallTrial,
    *,
    threshold: float,
    transition_ms: int,
    lying_confirmations: int,
    latch_ms: int,
) -> bool:
    detector = FallDetector(
        motion_threshold=threshold,
        transition_ms=transition_ms,
        lying_confirmations=lying_confirmations,
        latch_ms=latch_ms,
    )
    return any(
        detector.update(
            now_ms=now_ms,
            posture=posture,
            motion_energy=motion_energy,
        ).event
        == "suspected"
        for now_ms, posture, motion_energy in trial.frames
    )


def _select_threshold(
    trials: list[FallTrial],
    *,
    transition_ms: int,
    lying_confirmations: int,
    latch_ms: int,
) -> tuple[float, dict]:
    positives = [trial for trial in trials if trial.expected_fall]
    negatives = [trial for trial in trials if not trial.expected_fall]
    if len(positives) < 20:
        raise ValueError("fall training requires at least 20 fall trials")
    if len(negatives) < 40:
        raise ValueError("fall training requires at least 40 non-fall trials")
    energies = np.asarray(
        [
            motion_energy
            for trial in trials
            for _, _, motion_energy in trial.frames
        ],
        dtype=np.float64,
    )
    candidates = sorted(
        {
            max(float(value), 1e-9)
            for value in np.percentile(energies, np.linspace(1, 99, 99))
        }
    )
    scored: list[tuple[float, float, float]] = []
    for threshold in candidates:
        true_positive = sum(
            _trial_alerts(
                trial,
                threshold=threshold,
                transition_ms=transition_ms,
                lying_confirmations=lying_confirmations,
                latch_ms=latch_ms,
            )
            for trial in positives
        )
        false_positive = sum(
            _trial_alerts(
                trial,
                threshold=threshold,
                transition_ms=transition_ms,
                lying_confirmations=lying_confirmations,
                latch_ms=latch_ms,
            )
            for trial in negatives
        )
        recall = true_positive / len(positives)
        false_rate = false_positive / len(negatives)
        scored.append((recall, false_rate, threshold))
    eligible = [score for score in scored if score[1] <= 0.05]
    if not eligible:
        raise ValueError("no fall threshold keeps training false alerts at 5%")
    recall, false_rate, threshold = min(
        eligible,
        key=lambda item: (-item[0], item[1], item[2]),
    )
    if recall < 0.80:
        raise ValueError(
            f"best fall training recall {recall:.3f} is below 80%"
        )
    return threshold, {
        "fall_recall": recall,
        "nonfall_false_rate": false_rate,
        "fall_trials": len(positives),
        "nonfall_trials": len(negatives),
        "candidate_thresholds": len(candidates),
    }


def train(args: argparse.Namespace) -> int:
    topology = load_topology(args.topology)
    firmware = FirmwareBinding.load(args.firmware_binding)
    predictor = PosturePredictor(
        args.model,
        topology_id=topology["topology_id"],
        firmware=firmware,
    )
    trials = _load_trials(
        args.recordings,
        predictor=predictor,
        topology_id=topology["topology_id"],
    )
    threshold, metrics = _select_threshold(
        trials,
        transition_ms=args.transition_ms,
        lying_confirmations=args.lying_confirmations,
        latch_ms=args.latch_ms,
    )
    model = FallModel.create(
        posture_model_id=predictor.manifest.model_id,
        topology_id=topology["topology_id"],
        motion_calibration_id=predictor.manifest.calibration_id,
        training_recordings=(
            trial.recording_hash for trial in trials
        ),
        motion_threshold=threshold,
        transition_ms=args.transition_ms,
        lying_confirmations=args.lying_confirmations,
        latch_ms=args.latch_ms,
        metrics=metrics,
    )
    model.save(args.output)
    print(json.dumps(model.__dict__, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--firmware-binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transition-ms", type=int, default=1500)
    parser.add_argument("--lying-confirmations", type=int, default=2)
    parser.add_argument("--latch-ms", type=int, default=10_000)
    return train(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
