"""Empty-room NBVI calibration for the two controlled CSI links."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .features import (
    DEFAULT_MOTION_SUBCARRIERS,
    FEATURE_SCHEMA_HASH,
    MOTION_SUBCARRIERS,
    SUBCARRIER_BINS,
    motion_variance_series,
    sanitized_frame,
)
from .protocol import CsiPacket
from .recording import iter_recording

CALIBRATION_SCHEMA = "rvp-motion-calibration-v1"
NBVI_ALPHA = 0.75
NBVI_WINDOW_FRAMES = 200
NBVI_WINDOW_STEP = 50
NBVI_QUIET_PERCENTILE = 5.0
NBVI_NOISE_GATE_PERCENTILE = 15.0
NBVI_THRESHOLD_PERCENTILE = 95.0
NBVI_THRESHOLD_FACTOR = 1.1
VALID_SUBCARRIERS = tuple(
    index for index in range(11, 53) if index != 32
)


@dataclass(frozen=True)
class LinkMotionCalibration:
    node_id: int
    subcarrier_indices: tuple[int, ...]
    baseline_threshold: float
    baseline_false_positive_rate: float
    valid_frames: int


@dataclass(frozen=True)
class MotionCalibration:
    schema: str
    calibration_id: str
    topology_id: str
    feature_schema_hash: str
    probe_rate_hz: int
    empty_recordings: tuple[str, ...]
    links: dict[str, LinkMotionCalibration]

    @classmethod
    def create(
        cls,
        *,
        topology_id: str,
        probe_rate_hz: int,
        empty_recordings: Iterable[str],
        links: dict[str, LinkMotionCalibration],
    ) -> "MotionCalibration":
        payload = {
            "schema": CALIBRATION_SCHEMA,
            "topology_id": topology_id,
            "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "probe_rate_hz": probe_rate_hz,
            "empty_recordings": sorted(set(empty_recordings)),
            "links": {
                node: asdict(link)
                for node, link in sorted(links.items())
            },
        }
        calibration_id = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        calibration = cls(
            calibration_id=calibration_id,
            empty_recordings=tuple(payload["empty_recordings"]),
            **{
                key: value
                for key, value in payload.items()
                if key not in {"empty_recordings", "links"}
            },
            links=links,
        )
        calibration.validate()
        return calibration

    @classmethod
    def from_dict(cls, value: dict) -> "MotionCalibration":
        links = {
            str(node): LinkMotionCalibration(
                node_id=int(link["node_id"]),
                subcarrier_indices=tuple(
                    int(index) for index in link["subcarrier_indices"]
                ),
                baseline_threshold=float(link["baseline_threshold"]),
                baseline_false_positive_rate=float(
                    link["baseline_false_positive_rate"]
                ),
                valid_frames=int(link["valid_frames"]),
            )
            for node, link in value["links"].items()
        }
        calibration = cls(
            schema=str(value["schema"]),
            calibration_id=str(value["calibration_id"]),
            topology_id=str(value["topology_id"]),
            feature_schema_hash=str(value["feature_schema_hash"]),
            probe_rate_hz=int(value["probe_rate_hz"]),
            empty_recordings=tuple(value["empty_recordings"]),
            links=links,
        )
        calibration.validate()
        expected = cls.create(
            topology_id=calibration.topology_id,
            probe_rate_hz=calibration.probe_rate_hz,
            empty_recordings=calibration.empty_recordings,
            links=calibration.links,
        )
        if calibration.calibration_id != expected.calibration_id:
            raise ValueError("motion calibration ID does not match its content")
        return calibration

    def as_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.schema != CALIBRATION_SCHEMA:
            raise ValueError("unsupported motion calibration schema")
        if self.feature_schema_hash != FEATURE_SCHEMA_HASH:
            raise ValueError("motion calibration feature schema mismatch")
        if self.probe_rate_hz not in {50, 100}:
            raise ValueError("motion calibration rate must be 50 or 100 Hz")
        if set(self.links) != {"2", "3"}:
            raise ValueError("motion calibration requires receiver nodes 2 and 3")
        if not self.empty_recordings:
            raise ValueError("motion calibration requires empty-room recordings")
        for node, link in self.links.items():
            if link.node_id != int(node):
                raise ValueError("motion calibration node key does not match")
            _validate_band(link.subcarrier_indices)
            if (
                not math.isfinite(link.baseline_threshold)
                or link.baseline_threshold <= 0
            ):
                raise ValueError("motion baseline threshold must be positive")
            if not 0.0 <= link.baseline_false_positive_rate <= 0.05:
                raise ValueError("motion baseline false-positive rate exceeds 5%")
            if link.valid_frames < NBVI_WINDOW_FRAMES:
                raise ValueError("motion calibration has too few valid frames")

    def subcarriers(self) -> dict[int, tuple[int, ...]]:
        return {
            int(node): link.subcarrier_indices
            for node, link in self.links.items()
        }


def _validate_band(indices: tuple[int, ...]) -> None:
    if len(indices) != MOTION_SUBCARRIERS:
        raise ValueError(
            f"motion calibration requires {MOTION_SUBCARRIERS} subcarriers"
        )
    if tuple(sorted(indices)) != indices or len(set(indices)) != len(indices):
        raise ValueError("motion subcarriers must be sorted and unique")
    if any(index not in VALID_SUBCARRIERS for index in indices):
        raise ValueError("motion subcarrier is outside the HT20 usable range")
    if any(right - left <= 1 for left, right in zip(indices, indices[1:])):
        raise ValueError("motion subcarriers must be non-consecutive")


def _recording_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _amplitudes_by_node(
    recordings: Iterable[Path],
) -> tuple[dict[int, list[np.ndarray]], list[str]]:
    by_node: dict[int, list[np.ndarray]] = {2: [], 3: []}
    hashes: list[str] = []
    for path in recordings:
        hashes.append(_recording_sha256(path))
        current: dict[int, list[np.ndarray]] = {2: [], 3: []}
        for record in iter_recording(path):
            packet = record.packet
            if not isinstance(packet, CsiPacket) or packet.node_id not in current:
                continue
            try:
                amplitude, _ = sanitized_frame(packet)
            except (ValueError, np.linalg.LinAlgError):
                continue
            current[packet.node_id].append(amplitude)
        for node in by_node:
            if current[node]:
                by_node[node].append(np.stack(current[node]))
    return by_node, hashes


def _quiet_windows(amplitudes: np.ndarray) -> list[np.ndarray]:
    if len(amplitudes) < NBVI_WINDOW_FRAMES:
        raise ValueError(
            f"NBVI requires at least {NBVI_WINDOW_FRAMES} valid frames"
        )
    windows = [
        amplitudes[start : start + NBVI_WINDOW_FRAMES]
        for start in range(
            0,
            len(amplitudes) - NBVI_WINDOW_FRAMES + 1,
            NBVI_WINDOW_STEP,
        )
    ]
    if not windows:
        windows = [amplitudes]
    variance = np.asarray(
        [float(np.mean(np.var(window, axis=0))) for window in windows]
    )
    limit = float(np.percentile(variance, NBVI_QUIET_PERCENTILE))
    selected = [
        window for window, score in zip(windows, variance) if score <= limit
    ]
    return selected or [windows[int(np.argmin(variance))]]


def _entropy(values: np.ndarray) -> float:
    counts, _ = np.histogram(values, bins=16)
    probabilities = counts[counts > 0] / max(int(np.sum(counts)), 1)
    if len(probabilities) <= 1:
        return 0.0
    return float(
        -np.sum(probabilities * np.log2(probabilities)) / math.log2(16)
    )


def _score_candidates(window: np.ndarray) -> list[np.ndarray]:
    means = np.mean(window, axis=0)
    standard = np.std(window, axis=0)
    medians = np.median(window, axis=0)
    robust = 1.4826 * np.median(np.abs(window - medians), axis=0)
    denominator = np.maximum(means, 1e-6)
    classic = NBVI_ALPHA * standard / np.square(denominator)
    classic += (1.0 - NBVI_ALPHA) * standard / denominator
    mad_score = NBVI_ALPHA * robust / np.square(denominator)
    mad_score += (1.0 - NBVI_ALPHA) * robust / denominator
    entropy = np.asarray(
        [_entropy(window[:, index]) for index in range(SUBCARRIER_BINS)]
    )
    entropy_score = classic / np.maximum(entropy, 0.5)

    valid = np.asarray(VALID_SUBCARRIERS, dtype=np.int64)
    noise_floor = float(
        np.percentile(means[valid], NBVI_NOISE_GATE_PERCENTILE)
    )
    valid = valid[means[valid] >= noise_floor]
    candidates: list[np.ndarray] = []
    for scores in (classic, mad_score, entropy_score):
        ranking = valid[np.lexsort((valid, scores[valid]))]
        band = _select_spaced(ranking)
        if band is not None:
            candidates.append(band)
    return candidates


def _select_spaced(ranking: np.ndarray) -> np.ndarray | None:
    selected: list[int] = []
    for raw_index in ranking:
        index = int(raw_index)
        if all(abs(index - existing) > 1 for existing in selected):
            selected.append(index)
            if len(selected) == MOTION_SUBCARRIERS:
                return np.asarray(sorted(selected), dtype=np.int64)
    return None


def _baseline_values(
    segments: list[np.ndarray],
    band: np.ndarray,
    probe_rate_hz: int,
) -> np.ndarray:
    values = [
        motion_variance_series(
            segment,
            tuple(int(index) for index in band),
            sample_rate_hz=probe_rate_hz,
        )
        for segment in segments
    ]
    return np.concatenate(values)


def _calibrate_link(
    node_id: int,
    segments: list[np.ndarray],
    probe_rate_hz: int,
) -> LinkMotionCalibration:
    if not segments:
        raise ValueError(f"empty-room data has no frames for receiver {node_id}")
    amplitudes = np.concatenate(segments)
    if len(amplitudes) < NBVI_WINDOW_FRAMES:
        raise ValueError(
            f"receiver {node_id} has only {len(amplitudes)} empty-room frames"
        )

    candidates: dict[tuple[int, ...], np.ndarray] = {}
    for window in _quiet_windows(amplitudes):
        for candidate in _score_candidates(window):
            candidates[tuple(int(index) for index in candidate)] = candidate
    fallback = np.asarray(DEFAULT_MOTION_SUBCARRIERS, dtype=np.int64)
    candidates[tuple(DEFAULT_MOTION_SUBCARRIERS)] = fallback

    scored: list[tuple[float, float, tuple[int, ...], np.ndarray]] = []
    for key, candidate in candidates.items():
        values = _baseline_values(segments, candidate, probe_rate_hz)
        threshold = max(
            float(np.percentile(values, NBVI_THRESHOLD_PERCENTILE))
            * NBVI_THRESHOLD_FACTOR,
            1e-9,
        )
        false_positive_rate = float(np.mean(values > threshold))
        scored.append((false_positive_rate, threshold, key, values))
    false_positive_rate, threshold, band, _ = min(
        scored, key=lambda item: (item[0] > 0.05, item[0], item[1], item[2])
    )
    if false_positive_rate > 0.05:
        raise ValueError(
            f"receiver {node_id} NBVI baseline false-positive rate "
            f"{false_positive_rate:.3f} exceeds 5%"
        )
    return LinkMotionCalibration(
        node_id=node_id,
        subcarrier_indices=band,
        baseline_threshold=threshold,
        baseline_false_positive_rate=false_positive_rate,
        valid_frames=len(amplitudes),
    )


def calibrate_motion(
    recordings: Iterable[Path],
    *,
    topology_id: str,
    probe_rate_hz: int,
) -> MotionCalibration:
    paths = list(recordings)
    if not paths:
        raise ValueError("motion calibration requires empty-room recordings")
    by_node, hashes = _amplitudes_by_node(paths)
    links = {
        str(node): _calibrate_link(node, by_node[node], probe_rate_hz)
        for node in (2, 3)
    }
    return MotionCalibration.create(
        topology_id=topology_id,
        probe_rate_hz=probe_rate_hz,
        empty_recordings=hashes,
        links=links,
    )
