"""Locked blind-test evaluation for the single-person posture model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fall import FallDetector
from .features import extract_recording_windows
from .inference import PosturePredictor
from .model import POSTURE_CLASSES, macro_metrics
from .recording import read_metadata


def evaluate(args: argparse.Namespace) -> int:
    expected: list[str] = []
    predicted: list[str] = []
    confidence_values: list[float] = []
    unknown = 0
    window_count = 0
    absent_false_positive = 0
    absent_total = 0
    present_true_positive = 0
    present_total = 0
    fall_trials = 0
    fall_detected = 0
    nonfall_trials = 0
    nonfall_false_alerts = 0
    quality_windows = 0

    for recording in args.recordings:
        metadata = read_metadata(recording)
        if metadata.get("dataset_role") != "blind":
            raise ValueError(f"{recording} is not a locked blind recording")
        source_label = str(metadata.get("label"))
        posture_label = {
            "fall": "lying",
            "slow_lying": "lying",
        }.get(source_label, source_label)
        windows = extract_recording_windows(recording)
        if not windows:
            raise ValueError(f"{recording} contains no valid two-link windows")
        predictor = PosturePredictor(args.model, topology_id=args.topology_id)
        fall = FallDetector(
            motion_threshold=predictor.manifest.fall_motion_threshold
        )
        trial_alerted = False
        for window in windows:
            label, confidence = predictor.predict(window)
            window_count += 1
            confidence_values.append(confidence)
            if label == "unknown":
                unknown += 1
            if max(window.loss_per_node) <= 0.05 and min(
                frames / 2.0 for frames in window.frames_per_node
            ) >= 40.0:
                quality_windows += 1
            if posture_label in POSTURE_CLASSES:
                expected.append(posture_label)
                predicted.append(label)
            if posture_label == "absent":
                absent_total += 1
                absent_false_positive += int(label not in {"absent", "unknown"})
            elif posture_label in POSTURE_CLASSES:
                present_total += 1
                present_true_positive += int(
                    label not in {"absent", "unknown"}
                )
            decision = fall.update(
                now_ms=window.end_ns // 1_000_000,
                posture=label,
                motion_energy=window.motion_energy,
            )
            trial_alerted |= decision.event == "suspected"

        if source_label == "fall":
            fall_trials += 1
            fall_detected += int(trial_alerted)
        else:
            nonfall_trials += 1
            nonfall_false_alerts += int(trial_alerted)

    posture_metrics = macro_metrics(
        np.asarray(expected), np.asarray(predicted), POSTURE_CLASSES
    )
    absent_fpr = (
        absent_false_positive / absent_total if absent_total else 1.0
    )
    presence_recall = (
        present_true_positive / present_total if present_total else 0.0
    )
    unknown_rate = unknown / window_count if window_count else 1.0
    fall_recall = fall_detected / fall_trials if fall_trials else 0.0
    fall_false_rate = (
        nonfall_false_alerts / nonfall_trials if nonfall_trials else 1.0
    )
    link_quality_rate = (
        quality_windows / window_count if window_count else 0.0
    )

    gates = {
        "absent_false_positive_rate": absent_fpr <= 0.05,
        "presence_recall": presence_recall >= 0.90,
        "posture_macro_f1": posture_metrics["macro_f1"] >= 0.85,
        "posture_recall": all(
            posture_metrics["recall"][label] >= 0.85
            for label in ("standing", "sitting", "lying")
        ),
        "unknown_rate": unknown_rate <= 0.10,
        "fall_recall": fall_trials >= 20 and fall_recall >= 0.90,
        "fall_false_rate": nonfall_trials >= 40 and fall_false_rate <= 0.05,
        "link_quality": link_quality_rate >= 0.95,
    }
    result = {
        "schema": "rvp-blind-evaluation-v1",
        "recordings": [str(path) for path in args.recordings],
        "metrics": {
            "absent_false_positive_rate": absent_fpr,
            "presence_recall": presence_recall,
            "unknown_rate": unknown_rate,
            "fall_recall": fall_recall,
            "fall_false_rate": fall_false_rate,
            "link_quality_rate": link_quality_rate,
            "confidence_mean": float(np.mean(confidence_values)),
            "posture": posture_metrics,
        },
        "trial_counts": {
            "fall": fall_trials,
            "nonfall": nonfall_trials,
        },
        "gates": gates,
        "latency": {
            "status": "manual_pending",
            "reason": "requires synchronized camera transition labels",
        },
        "passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--topology-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return evaluate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
