"""Locked, profile-specific blind evaluation for staged capability activation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .fall import FallDetector
from .features import extract_recording_windows
from .inference import PosturePredictor, Prediction
from .labels import labels_from_metadata
from .model import (
    FirmwareBinding,
    POSTURE_CLASSES,
    aggregate_hash,
    macro_metrics,
    recording_sha256,
)
from .recording import read_metadata
from .topology import load_topology


def _p95(values: list[float]) -> float | None:
    return float(np.percentile(values, 95)) if values else None


def _transition_target(profile: str, event: str) -> str | None:
    return {
        ("presence", "enter"): "present",
        ("presence", "exit"): "absent",
        ("motion", "motion_start"): "moving",
        ("motion", "motion_stop"): "still",
        ("posture", "sit_down"): "sitting",
        ("posture", "stand_up"): "standing",
        ("posture", "slow_lying"): "lying",
    }.get((profile, event))


def _prediction_label(profile: str, prediction: Prediction) -> str:
    if profile == "presence":
        return prediction.presence.label
    if profile == "motion":
        return prediction.motion.label
    if profile == "posture":
        return prediction.posture.label
    raise ValueError(f"no direct label for profile {profile}")


def evaluate(args: argparse.Namespace) -> int:
    topology = load_topology(args.topology)
    firmware = FirmwareBinding.load(args.firmware_binding)
    predictor = PosturePredictor(
        args.model,
        topology_id=topology["topology_id"],
        firmware=firmware,
    )

    expected_posture: list[str] = []
    predicted_posture: list[str] = []
    window_count = 0
    quality_windows = 0
    unknown_by_profile = 0
    relevant_by_profile = 0
    absent_total = 0
    absent_false_positive = 0
    present_total = 0
    present_true_positive = 0
    moving_total = 0
    moving_true_positive = 0
    still_total = 0
    still_false_motion = 0
    transition_latencies: list[float] = []
    transition_trials = 0
    missing_transition_trials: list[str] = []
    fall_trials = 0
    fall_detected = 0
    fall_latencies: list[float] = []
    nonfall_trials = 0
    nonfall_false_alerts = 0
    recording_hashes: list[str] = []

    for recording in args.recordings:
        metadata = read_metadata(recording)
        labels = labels_from_metadata(metadata)
        if labels.dataset_role != "blind":
            raise ValueError(f"{recording} is not locked blind data")
        if metadata.get("topology_id") != topology["topology_id"]:
            raise ValueError(f"{recording} topology does not match")
        recording_hashes.append(recording_sha256(recording))
        windows = extract_recording_windows(
            recording,
            motion_subcarriers=predictor.motion_calibration.subcarriers(),
        )
        if not windows:
            raise ValueError(f"{recording} contains no valid two-link windows")
        predictor.reset()
        fall = FallDetector(
            motion_threshold=predictor.manifest.fall_motion_threshold
        )
        trial_alerted = False
        first_fall_ns: int | None = None
        profile_target = _transition_target(args.profile, labels.event)
        transition_at_ns = metadata.get("transition_at_host_ns")
        profile_latency: float | None = None

        for window in windows:
            prediction = predictor.predict(window)
            window_count += 1
            minimum_fps = firmware.probe_rate_hz * 0.8
            quality = (
                max(window.loss_per_node) <= 0.05
                and min(
                    frames / 2.0 for frames in window.frames_per_node
                )
                >= minimum_fps
                and min(window.structure_stability_per_node) >= 0.95
            )
            quality_windows += int(quality)

            presence = prediction.presence.label
            if labels.occupancy == "absent":
                absent_total += 1
                absent_false_positive += int(presence == "present")
            else:
                present_total += 1
                present_true_positive += int(presence == "present")

            if labels.occupancy == "present" and labels.motion == "moving":
                moving_total += 1
                moving_true_positive += int(
                    prediction.motion.label == "moving"
                )
            elif labels.occupancy == "present" and labels.motion == "idle":
                still_total += 1
                still_false_motion += int(
                    prediction.motion.label == "moving"
                )

            if labels.posture in POSTURE_CLASSES:
                expected_posture.append(labels.posture)
                predicted_posture.append(prediction.posture.label)

            if args.profile != "fall":
                relevant = (
                    args.profile == "presence"
                    or (
                        args.profile == "motion"
                        and labels.occupancy == "present"
                        and labels.motion != "unknown"
                    )
                    or (
                        args.profile == "posture"
                        and labels.posture in POSTURE_CLASSES
                    )
                )
                if relevant:
                    relevant_by_profile += 1
                    unknown_by_profile += int(
                        _prediction_label(args.profile, prediction) == "unknown"
                    )
            if (
                profile_target is not None
                and transition_at_ns is not None
                and window.end_ns >= int(transition_at_ns)
                and profile_latency is None
                and _prediction_label(args.profile, prediction)
                == profile_target
            ):
                profile_latency = (
                    window.end_ns - int(transition_at_ns)
                ) / 1_000_000_000.0

            fall_decision = fall.update(
                now_ms=window.end_ns // 1_000_000,
                posture=prediction.posture.label,
                motion_energy=window.motion_energy,
            )
            if fall_decision.event == "suspected" and not trial_alerted:
                trial_alerted = True
                first_fall_ns = window.end_ns

        if profile_target is not None:
            transition_trials += 1
            if transition_at_ns is None or profile_latency is None:
                missing_transition_trials.append(str(recording))
            else:
                transition_latencies.append(profile_latency)

        if labels.event == "fall":
            fall_trials += 1
            fall_detected += int(trial_alerted)
            if transition_at_ns is not None and first_fall_ns is not None:
                fall_latencies.append(
                    (first_fall_ns - int(transition_at_ns))
                    / 1_000_000_000.0
                )
        elif labels.occupancy == "present":
            nonfall_trials += 1
            nonfall_false_alerts += int(trial_alerted)

    posture_metrics = macro_metrics(
        np.asarray(expected_posture),
        np.asarray(predicted_posture),
        POSTURE_CLASSES,
    )
    absent_fpr = absent_false_positive / absent_total if absent_total else 1.0
    presence_recall = (
        present_true_positive / present_total if present_total else 0.0
    )
    motion_recall = (
        moving_true_positive / moving_total if moving_total else 0.0
    )
    still_motion_fpr = (
        still_false_motion / still_total if still_total else 1.0
    )
    unknown_rate = (
        unknown_by_profile / relevant_by_profile
        if relevant_by_profile
        else 1.0
    )
    link_quality_rate = (
        quality_windows / window_count if window_count else 0.0
    )
    fall_recall = fall_detected / fall_trials if fall_trials else 0.0
    fall_false_rate = (
        nonfall_false_alerts / nonfall_trials if nonfall_trials else 1.0
    )
    transition_p95 = _p95(transition_latencies)
    fall_p95 = _p95(fall_latencies)

    common_gates = {
        "link_quality": link_quality_rate >= 0.95,
        "unknown_rate": unknown_rate <= 0.10
        if args.profile != "fall"
        else True,
    }
    if args.profile == "presence":
        profile_gates = {
            "absent_false_positive_rate": absent_total > 0
            and absent_fpr <= 0.05,
            "presence_recall": present_total > 0 and presence_recall >= 0.90,
            "transition_latency": transition_trials >= 20
            and not missing_transition_trials
            and transition_p95 is not None
            and transition_p95 <= 2.0,
        }
    elif args.profile == "motion":
        profile_gates = {
            "motion_recall": moving_total > 0 and motion_recall >= 0.90,
            "still_false_motion_rate": still_total > 0
            and still_motion_fpr <= 0.10,
            "transition_latency": transition_trials >= 20
            and not missing_transition_trials
            and transition_p95 is not None
            and transition_p95 <= 2.0,
        }
    elif args.profile == "posture":
        profile_gates = {
            "posture_macro_f1": posture_metrics["macro_f1"] >= 0.85,
            "posture_recall": all(
                posture_metrics["recall"][label] >= 0.85
                for label in POSTURE_CLASSES
            ),
            "transition_latency": transition_trials > 0
            and not missing_transition_trials
            and transition_p95 is not None
            and transition_p95 <= 2.0,
        }
    else:
        profile_gates = {
            "fall_recall": fall_trials >= 20 and fall_recall >= 0.90,
            "fall_false_rate": nonfall_trials >= 40
            and fall_false_rate <= 0.05,
            "fall_latency": len(fall_latencies) >= 20
            and fall_p95 is not None
            and fall_p95 <= 1.5,
        }
    gates = {**common_gates, **profile_gates}
    result = {
        "schema": "rvp-acceptance-v2",
        "profile": args.profile,
        "model_id": predictor.manifest.model_id,
        "topology_id": topology["topology_id"],
        "recording_hashes": sorted(recording_hashes),
        "blind_dataset_hash": aggregate_hash(recording_hashes),
        "metrics": {
            "absent_false_positive_rate": absent_fpr,
            "presence_recall": presence_recall,
            "motion_recall": motion_recall,
            "still_false_motion_rate": still_motion_fpr,
            "unknown_rate": unknown_rate,
            "link_quality_rate": link_quality_rate,
            "transition_p95_seconds": transition_p95,
            "fall_recall": fall_recall,
            "fall_false_rate": fall_false_rate,
            "fall_latency_p95_seconds": fall_p95,
            "posture": posture_metrics,
        },
        "trial_counts": {
            "transitions": transition_trials,
            "missing_transitions": missing_transition_trials,
            "fall": fall_trials,
            "nonfall": nonfall_trials,
        },
        "gates": gates,
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
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--firmware-binding", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=["presence", "motion", "posture", "fall"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return evaluate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
