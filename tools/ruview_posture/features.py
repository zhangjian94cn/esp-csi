"""Deterministic two-link CSI window feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .protocol import CsiPacket
from .recording import RecordedPacket, iter_recording

FEATURE_SCHEMA_VERSION = "rvp-features-v2"
SUBCARRIER_BINS = 64
MOTION_SUBCARRIERS = 12
MOTION_VARIANCE_WINDOW_SECONDS = 0.2
HAMPEL_WINDOW_FRAMES = 7
HAMPEL_THRESHOLD = 5.0
DEFAULT_MOTION_SUBCARRIERS = (
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
)
WINDOW_NS = 2_000_000_000
STEP_NS = 250_000_000


@dataclass(frozen=True)
class WindowFeatures:
    start_ns: int
    end_ns: int
    vector: np.ndarray
    motion_energy: float
    node_ids: tuple[int, int]
    frames_per_node: tuple[int, int]
    loss_per_node: tuple[float, float]
    structure_stability_per_node: tuple[float, float]


def feature_names() -> list[str]:
    names: list[str] = []
    for node_slot in ("link_a", "link_b"):
        for family in (
            "amp_mean",
            "amp_std",
            "phase_diff_std",
            "amp_delta_rms",
        ):
            names.extend(
                f"{node_slot}.{family}.{index}"
                for index in range(SUBCARRIER_BINS)
            )
        names.extend(
            [
                f"{node_slot}.rssi_mean",
                f"{node_slot}.rssi_std",
                f"{node_slot}.sequence_loss",
                f"{node_slot}.fps",
                f"{node_slot}.gain_mean",
                f"{node_slot}.gain_std",
            ]
        )
    names.extend(
        [
            "cross.amp_correlation",
            "cross.motion_correlation",
            "cross.fps_balance",
            "cross.csi_length_balance",
        ]
    )
    for node_slot in ("link_a", "link_b"):
        names.extend(
            f"motion.{node_slot}.mvs_segment.{index}"
            for index in range(MOTION_SUBCARRIERS)
        )
    names.extend(
        [
            "motion.mvs_mean",
            "motion.mvs_std",
            "motion.cross_correlation",
        ]
    )
    return names


FEATURE_SCHEMA_HASH = hashlib.sha256(
    json.dumps(
        {
            "version": FEATURE_SCHEMA_VERSION,
            "names": feature_names(),
            "window_ns": WINDOW_NS,
            "step_ns": STEP_NS,
            "motion_variance_window_seconds": MOTION_VARIANCE_WINDOW_SECONDS,
            "hampel_window_frames": HAMPEL_WINDOW_FRAMES,
            "hampel_threshold": HAMPEL_THRESHOLD,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def _resample(values: np.ndarray, bins: int = SUBCARRIER_BINS) -> np.ndarray:
    if len(values) == bins:
        return values.astype(np.float64, copy=False)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, bins)
    return np.interp(target, source, values).astype(np.float64)


def _complex_csi(packet: CsiPacket) -> np.ndarray:
    raw = np.frombuffer(packet.csi, dtype=np.int8)
    if packet.flags & 1 and len(raw) >= 4:
        raw = raw[4:]
    if len(raw) < 4 or len(raw) % 2:
        raise ValueError("CSI payload must contain interleaved I/Q int8 pairs")
    pairs = raw.reshape(-1, 2).astype(np.float64)
    complex_values = pairs[:, 1] + 1j * pairs[:, 0]
    return complex_values * packet.gain_compensation


def sanitized_frame(packet: CsiPacket) -> tuple[np.ndarray, np.ndarray]:
    values = _complex_csi(packet)
    amplitude = np.abs(values)
    scale = max(float(np.median(amplitude)), 1.0)
    amplitude = _resample(amplitude / scale)

    phase = np.unwrap(np.angle(values))
    x = np.linspace(-1.0, 1.0, len(phase))
    slope, offset = np.polyfit(x, phase, 1)
    residual = phase - (slope * x + offset)
    phase_difference = np.diff(residual, prepend=residual[0])
    phase_difference = _resample(phase_difference)
    return amplitude, phase_difference


def _hampel_filter(
    values: np.ndarray,
    *,
    window: int = HAMPEL_WINDOW_FRAMES,
    threshold: float = HAMPEL_THRESHOLD,
) -> np.ndarray:
    """Suppress isolated turbulence while retaining sustained movement."""

    if values.ndim != 2:
        raise ValueError("Hampel input must be frames by subcarriers")
    filtered = values.astype(np.float64, copy=True)
    radius = window // 2
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        local = values[start:end]
        median = np.median(local, axis=0)
        mad = np.median(np.abs(local - median), axis=0)
        scale = 1.4826 * mad
        outlier = np.abs(values[index] - median) > threshold * np.maximum(
            scale, 1e-6
        )
        filtered[index, outlier] = median[outlier]
    return filtered


def _validate_motion_subcarriers(indices: Sequence[int]) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    if len(selected) != MOTION_SUBCARRIERS:
        raise ValueError(
            f"motion extraction requires {MOTION_SUBCARRIERS} subcarriers"
        )
    if len(set(int(index) for index in selected)) != len(selected):
        raise ValueError("motion subcarriers must be unique")
    if np.any(selected < 0) or np.any(selected >= SUBCARRIER_BINS):
        raise ValueError("motion subcarrier index is outside the feature grid")
    return selected


def motion_variance_series(
    amplitudes: np.ndarray,
    subcarrier_indices: Sequence[int],
    *,
    sample_rate_hz: float,
) -> np.ndarray:
    """Return Hampel-filtered moving variance of spatial turbulence."""

    if amplitudes.ndim != 2 or amplitudes.shape[1] != SUBCARRIER_BINS:
        raise ValueError("motion input must be frames by 64 subcarriers")
    if len(amplitudes) < 3:
        raise ValueError("motion extraction requires at least three frames")
    selected = _validate_motion_subcarriers(subcarrier_indices)
    filtered = _hampel_filter(amplitudes)
    turbulence = np.std(filtered[:, selected], axis=1)
    turbulence = _hampel_filter(turbulence[:, np.newaxis])[:, 0]
    width = max(
        3, int(round(sample_rate_hz * MOTION_VARIANCE_WINDOW_SECONDS))
    )
    width = min(width, len(turbulence))
    cumulative = np.concatenate(([0.0], np.cumsum(turbulence)))
    cumulative_square = np.concatenate(
        ([0.0], np.cumsum(np.square(turbulence)))
    )
    result = np.empty(len(turbulence), dtype=np.float64)
    for index in range(len(turbulence)):
        start = max(0, index + 1 - width)
        count = index + 1 - start
        mean = (cumulative[index + 1] - cumulative[start]) / count
        mean_square = (
            cumulative_square[index + 1] - cumulative_square[start]
        ) / count
        result[index] = max(mean_square - mean * mean, 0.0)
    return result


def _motion_features(
    amplitudes: np.ndarray,
    subcarrier_indices: Sequence[int],
    *,
    sample_rate_hz: float,
) -> tuple[np.ndarray, float]:
    series = motion_variance_series(
        amplitudes,
        subcarrier_indices,
        sample_rate_hz=sample_rate_hz,
    )
    segments = np.array_split(series, MOTION_SUBCARRIERS)
    vector = np.asarray(
        [float(np.mean(segment)) if len(segment) else 0.0 for segment in segments]
    )
    return vector, float(np.mean(series))


def _sequence_loss(sequences: Sequence[int]) -> float:
    if len(sequences) < 2:
        return 1.0
    missing = 0
    expected = 0
    for previous, current in zip(sequences, sequences[1:]):
        delta = (current - previous) & 0xFFFFFFFF
        if delta == 0 or delta > 10_000:
            continue
        expected += delta
        missing += max(delta - 1, 0)
    return float(missing / expected) if expected else 0.0


def _link_features(
    records: Sequence[RecordedPacket],
    duration_seconds: float,
    motion_subcarriers: Sequence[int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    np.ndarray,
    float,
    float,
]:
    amplitudes: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    packets: list[CsiPacket] = []
    for record in records:
        packet = record.packet
        if not isinstance(packet, CsiPacket):
            continue
        try:
            amplitude, phase = sanitized_frame(packet)
        except (ValueError, np.linalg.LinAlgError):
            continue
        amplitudes.append(amplitude)
        phases.append(phase)
        packets.append(packet)

    if len(packets) < 3:
        raise ValueError("not enough valid CSI frames for a link window")

    amp = np.stack(amplitudes)
    phase = np.stack(phases)
    delta = np.diff(amp, axis=0)
    delta_rms = np.sqrt(np.mean(np.square(delta), axis=0))
    rssi = np.asarray([packet.rssi for packet in packets], dtype=np.float64)
    gain = np.asarray(
        [packet.gain_compensation for packet in packets], dtype=np.float64
    )
    loss = _sequence_loss([packet.sequence for packet in packets])
    fps = len(packets) / duration_seconds
    motion_vector, motion_energy = _motion_features(
        amp,
        motion_subcarriers,
        sample_rate_hz=fps,
    )

    vector = np.concatenate(
        [
            np.mean(amp, axis=0),
            np.std(amp, axis=0),
            np.std(phase, axis=0),
            delta_rms,
            np.asarray(
                [
                    np.mean(rssi),
                    np.std(rssi),
                    loss,
                    fps,
                    np.mean(gain),
                    np.std(gain),
                ]
            ),
        ]
    )
    lengths = np.asarray([len(packet.csi) for packet in packets], dtype=np.float64)
    unique_lengths, length_counts = np.unique(lengths, return_counts=True)
    del unique_lengths
    structure_stability = float(np.max(length_counts) / len(lengths))
    return (
        vector,
        np.mean(amp, axis=0),
        delta_rms,
        loss,
        float(np.mean(lengths)),
        motion_vector,
        structure_stability,
        motion_energy,
    )


def extract_window(
    records: Sequence[RecordedPacket],
    *,
    start_ns: int,
    end_ns: int,
    required_nodes: tuple[int, int] | None = None,
    min_frames_per_link: int = 10,
    motion_subcarriers: dict[int, tuple[int, ...]] | None = None,
) -> WindowFeatures:
    grouped: dict[int, list[RecordedPacket]] = {}
    for record in records:
        if not start_ns <= record.host_timestamp_ns < end_ns:
            continue
        if isinstance(record.packet, CsiPacket):
            grouped.setdefault(record.packet.node_id, []).append(record)

    node_ids = tuple(sorted(grouped))
    if required_nodes is not None:
        node_ids = required_nodes
    if len(node_ids) != 2:
        raise ValueError(f"exactly two receiver nodes are required, got {node_ids}")
    if any(len(grouped.get(node, ())) < min_frames_per_link for node in node_ids):
        raise ValueError("one or more links do not have enough frames")
    bands = motion_subcarriers or {
        node: DEFAULT_MOTION_SUBCARRIERS for node in node_ids
    }
    if any(node not in bands for node in node_ids):
        raise ValueError("motion calibration is missing a receiver link")

    duration = (end_ns - start_ns) / 1_000_000_000.0
    left = _link_features(grouped[node_ids[0]], duration, bands[node_ids[0]])
    right = _link_features(grouped[node_ids[1]], duration, bands[node_ids[1]])
    amp_corr = float(np.corrcoef(left[1], right[1])[0, 1])
    motion_corr = float(np.corrcoef(left[5], right[5])[0, 1])
    if not np.isfinite(amp_corr):
        amp_corr = 0.0
    if not np.isfinite(motion_corr):
        motion_corr = 0.0
    left_fps = left[0][-3]
    right_fps = right[0][-3]
    cross = np.asarray(
        [
            amp_corr,
            motion_corr,
            min(left_fps, right_fps) / max(left_fps, right_fps, 1e-9),
            min(left[4], right[4]) / max(left[4], right[4], 1e-9),
        ]
    )
    motion = np.concatenate(
        [
            left[5],
            right[5],
            np.asarray(
                [
                    np.mean(np.concatenate([left[5], right[5]])),
                    np.std(np.concatenate([left[5], right[5]])),
                    motion_corr,
                ]
            ),
        ]
    )
    vector = np.concatenate([left[0], right[0], cross, motion]).astype(
        np.float32
    )
    if len(vector) != len(feature_names()):
        raise AssertionError(
            f"feature schema mismatch vector={len(vector)} names={len(feature_names())}"
        )
    return WindowFeatures(
        start_ns=start_ns,
        end_ns=end_ns,
        vector=vector,
        motion_energy=float(np.mean([left[7], right[7]])),
        node_ids=(node_ids[0], node_ids[1]),
        frames_per_node=(
            len(grouped[node_ids[0]]),
            len(grouped[node_ids[1]]),
        ),
        loss_per_node=(left[3], right[3]),
        structure_stability_per_node=(left[6], right[6]),
    )


def head_feature_indices(head: str) -> np.ndarray:
    names = feature_names()
    if head == "motion":
        selected = [
            index for index, name in enumerate(names) if name.startswith("motion.")
        ]
    elif head in {"presence", "posture"}:
        selected = [
            index
            for index, name in enumerate(names)
            if not name.startswith("motion.")
        ]
    else:
        raise ValueError(f"unknown model head {head!r}")
    return np.asarray(selected, dtype=np.int64)


def vector_for_head(window: WindowFeatures, head: str) -> np.ndarray:
    return window.vector[head_feature_indices(head)]


def extract_recording_windows(
    path: Path,
    *,
    required_nodes: tuple[int, int] | None = None,
    min_frames_per_link: int = 10,
    motion_subcarriers: dict[int, tuple[int, ...]] | None = None,
) -> list[WindowFeatures]:
    records = list(iter_recording(path))
    csi_records = [
        record for record in records if isinstance(record.packet, CsiPacket)
    ]
    if not csi_records:
        return []
    first = min(record.host_timestamp_ns for record in csi_records)
    last = max(record.host_timestamp_ns for record in csi_records)
    windows: list[WindowFeatures] = []
    start = first
    while start + WINDOW_NS <= last + 1:
        try:
            windows.append(
                extract_window(
                    records,
                    start_ns=start,
                    end_ns=start + WINDOW_NS,
                    required_nodes=required_nodes,
                    min_frames_per_link=min_frames_per_link,
                    motion_subcarriers=motion_subcarriers,
                )
            )
        except ValueError:
            pass
        start += STEP_NS
    return windows
