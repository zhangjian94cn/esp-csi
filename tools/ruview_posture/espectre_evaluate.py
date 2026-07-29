"""Evaluate and select ESPectre MVS/ML as a motion-only benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from .espectre_recording import EspectreSample, read_espectre_recording


def _eligible_samples(
    metadata: dict[str, Any], samples: list[EspectreSample]
) -> list[EspectreSample]:
    if not samples:
        return []
    start = samples[0].monotonic_ns + int(
        float(metadata.get("warmup_seconds", 0.0)) * 1_000_000_000
    )
    return [
        sample
        for sample in samples
        if sample.monotonic_ns >= start and sample.motion is not None
    ]


def _transition_latency(
    metadata: dict[str, Any], samples: list[EspectreSample]
) -> float | None:
    transition = metadata.get("transition", "none")
    if transition == "none" or not samples:
        return None
    target = transition == "enter"
    start_ns = samples[0].monotonic_ns + int(
        float(metadata.get("warmup_seconds", 0.0)) * 1_000_000_000
    )
    for sample in samples:
        if (
            sample.monotonic_ns >= start_ns
            and sample.connected
            and sample.motion is target
        ):
            return (sample.monotonic_ns - start_ns) / 1_000_000_000.0
    return None


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def evaluate_algorithm(
    paths: Iterable[Path], *, expected_algorithm: str | None = None
) -> dict[str, Any]:
    idle_total = 0
    idle_motion = 0
    moving_total = 0
    moving_detected = 0
    disconnected_samples = 0
    sample_count = 0
    entry_latencies: list[float] = []
    exit_latencies: list[float] = []
    missing_transitions: list[str] = []
    movement_ratios: list[float] = []
    algorithm: str | None = expected_algorithm
    recordings: list[str] = []

    for path in paths:
        metadata, all_samples = read_espectre_recording(path)
        current_algorithm = str(metadata["algorithm"])
        if algorithm is None:
            algorithm = current_algorithm
        if current_algorithm != algorithm:
            raise ValueError(f"{path} belongs to {current_algorithm}, not {algorithm}")
        recordings.append(str(path))
        samples = _eligible_samples(metadata, all_samples)
        expected_motion = bool(metadata["expected_motion"])
        for sample in samples:
            sample_count += 1
            disconnected_samples += int(not sample.connected)
            if expected_motion and metadata["scenario"] == "moving":
                moving_total += 1
                moving_detected += int(bool(sample.motion))
            elif metadata["scenario"] in {
                "empty_fan_off",
                "environment_interference",
                "present_still",
            }:
                idle_total += 1
                idle_motion += int(bool(sample.motion))
            if (
                sample.movement_score is not None
                and sample.threshold is not None
                and sample.threshold > 0
            ):
                movement_ratios.append(
                    sample.movement_score / sample.threshold
                )
        latency = _transition_latency(metadata, all_samples)
        transition = metadata.get("transition", "none")
        if transition == "enter":
            if latency is None:
                missing_transitions.append(str(path))
            else:
                entry_latencies.append(latency)
        elif transition == "exit":
            if latency is None:
                missing_transitions.append(str(path))
            else:
                exit_latencies.append(latency)

    motion_recall = moving_detected / moving_total if moving_total else 0.0
    idle_false_positive = idle_motion / idle_total if idle_total else 1.0
    disconnect_rate = (
        disconnected_samples / sample_count if sample_count else 1.0
    )
    entry_p95 = _percentile(entry_latencies, 95)
    exit_p95 = _percentile(exit_latencies, 95)
    gates = {
        "motion_recall": moving_total > 0 and motion_recall >= 0.95,
        "idle_false_positive_rate": (
            idle_total > 0 and idle_false_positive <= 0.05
        ),
        "entry_latency_p95": (
            len(entry_latencies) >= 10
            and entry_p95 is not None
            and entry_p95 <= 1.0
        ),
        "exit_latency_p95": (
            len(exit_latencies) >= 10
            and exit_p95 is not None
            and exit_p95 <= 2.0
        ),
        "connection": disconnected_samples == 0,
        "transition_completeness": not missing_transitions,
    }
    return {
        "schema": "rvp-espectre-evaluation-v1",
        "algorithm": algorithm,
        "recordings": recordings,
        "metrics": {
            "motion_recall": motion_recall,
            "idle_false_positive_rate": idle_false_positive,
            "entry_latency_p95_seconds": entry_p95,
            "exit_latency_p95_seconds": exit_p95,
            "disconnected_sample_rate": disconnect_rate,
            "movement_threshold_ratio_mean": (
                mean(movement_ratios) if movement_ratios else None
            ),
        },
        "counts": {
            "samples": sample_count,
            "moving_samples": moving_total,
            "idle_samples": idle_total,
            "entry_trials": len(entry_latencies),
            "exit_trials": len(exit_latencies),
            "missing_transitions": missing_transitions,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "scope": "motion_only",
        "idle_is_not_absence": True,
    }


def select_baseline(reports: list[dict[str, Any]]) -> str | None:
    passed = [report for report in reports if report["passed"]]
    if not passed:
        return None
    return min(
        passed,
        key=lambda report: (
            report["metrics"]["idle_false_positive_rate"],
            report["metrics"]["entry_latency_p95_seconds"],
            report["metrics"]["disconnected_sample_rate"],
            report["algorithm"],
        ),
    )["algorithm"]


def evaluate(args: argparse.Namespace) -> int:
    grouped: dict[str, list[Path]] = {"mvs": [], "ml": []}
    for path in args.recordings:
        metadata, _ = read_espectre_recording(path)
        grouped[str(metadata["algorithm"])].append(path)
    reports = [
        evaluate_algorithm(paths, expected_algorithm=algorithm)
        for algorithm, paths in grouped.items()
        if paths
    ]
    selected = select_baseline(reports)
    complete_ab = {report["algorithm"] for report in reports} == {"mvs", "ml"}
    result = {
        "schema": "rvp-espectre-ab-v1",
        "reports": reports,
        "selected_baseline": selected,
        "passed": complete_ab and selected is not None,
        "complete_ab": complete_ab,
        "all_algorithms_passed": bool(reports)
        and all(report["passed"] for report in reports),
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
    parser.add_argument("--output", required=True, type=Path)
    return evaluate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
