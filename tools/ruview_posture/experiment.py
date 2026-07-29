"""Thread-safe, timer-driven raw CSI experiment sessions."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
import threading
import time
from typing import Any, Mapping

from .labels import TrialLabels
from .protocol import CsiPacket, Packet
from .recording import RecordingWriter


class SessionConflict(RuntimeError):
    """Raised when an experiment transition would lose or mix evidence."""


def _safe_segment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    if not normalized:
        raise ValueError("trial_id must contain a safe filename character")
    return normalized[:80]


class ExperimentSessionManager:
    def __init__(
        self,
        *,
        output_directory: Path,
        topology_id: str,
        required_nodes: tuple[int, int],
    ):
        self.output_directory = output_directory
        self.topology_id = topology_id
        self.required_nodes = required_nodes
        self.lock = threading.Lock()
        self._writer: RecordingWriter | None = None
        self._request_signature: str | None = None
        self._state = "idle"
        self._session_id: str | None = None
        self._output: Path | None = None
        self._labels: TrialLabels | None = None
        self._countdown_deadline_ns = 0
        self._recording_deadline_ns = 0
        self._duration_seconds = 0.0
        self._counts: Counter[int] = Counter()
        self._failure_reason: str | None = None
        self._started_at: str | None = None
        self._transition_offset_seconds: float | None = None

    def _advance(self, now_ns: int) -> None:
        if self._state == "countdown" and now_ns >= self._countdown_deadline_ns:
            assert self._labels is not None
            assert self._output is not None
            assert self._session_id is not None
            metadata = {
                "schema": "rvp-recording-v2",
                "session_id": self._session_id,
                "topology_id": self.topology_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "started_monotonic_ns": now_ns,
                "started_host_timestamp_ns": time.time_ns(),
                **self._labels.as_metadata(),
            }
            if self._transition_offset_seconds is not None:
                metadata["transition_at_host_ns"] = (
                    metadata["started_host_timestamp_ns"]
                    + int(self._transition_offset_seconds * 1_000_000_000)
                )
            self._writer = RecordingWriter(self._output, metadata)
            self._recording_deadline_ns = now_ns + int(
                self._duration_seconds * 1_000_000_000
            )
            self._state = "recording"
            self._started_at = metadata["started_at"]
        if (
            self._state == "recording"
            and now_ns >= self._recording_deadline_ns
        ):
            self._complete()

    def _complete(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._state = "completed"

    def start(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        labels = TrialLabels(
            occupancy=str(payload["occupancy"]),
            motion=str(payload["motion"]),
            posture=str(payload["posture"]),
            event=str(payload.get("event", "none")),
            zone_id=str(payload["zone_id"]),
            fan_state=str(payload.get("fan_state", "unknown")),
            curtain_state=str(payload.get("curtain_state", "unknown")),
            trial_id=str(payload["trial_id"]),
            dataset_role=str(payload["dataset_role"]),
        )
        labels.validate()
        countdown = float(payload.get("countdown_seconds", 5.0))
        duration = float(payload["duration_seconds"])
        transition_offset_raw = payload.get("transition_offset_seconds")
        transition_offset = (
            None
            if transition_offset_raw is None
            else float(transition_offset_raw)
        )
        if not 0 <= countdown <= 60:
            raise ValueError("countdown_seconds must be between 0 and 60")
        if not 1 <= duration <= 3600:
            raise ValueError("duration_seconds must be between 1 and 3600")
        if transition_offset is not None and not 0 <= transition_offset < duration:
            raise ValueError(
                "transition_offset_seconds must fall inside the recording"
            )
        signature = json.dumps(
            {
                **labels.as_metadata(),
                "countdown_seconds": countdown,
                "duration_seconds": duration,
                "transition_offset_seconds": transition_offset,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.lock:
            self._advance(time.monotonic_ns())
            if self._state in {"countdown", "recording"}:
                if signature == self._request_signature:
                    return self._snapshot_locked(time.monotonic_ns()), False
                raise SessionConflict("another experiment session is active")
            now = datetime.now(timezone.utc)
            self._session_id = (
                now.strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(4)
            )
            filename = (
                f"{self._session_id}-{_safe_segment(labels.trial_id)}.rvp"
            )
            self._output = self.output_directory / filename
            self._labels = labels
            self._request_signature = signature
            self._state = "countdown"
            self._countdown_deadline_ns = time.monotonic_ns() + int(
                countdown * 1_000_000_000
            )
            self._duration_seconds = duration
            self._counts.clear()
            self._failure_reason = None
            self._started_at = None
            self._transition_offset_seconds = transition_offset
            self._advance(time.monotonic_ns())
            return self._snapshot_locked(time.monotonic_ns()), True

    def ingest(
        self,
        host_timestamp_ns: int,
        monotonic_ns: int,
        datagram: bytes,
        packet: Packet,
    ) -> None:
        with self.lock:
            self._advance(monotonic_ns)
            if self._state != "recording" or self._writer is None:
                return
            try:
                self._writer.append(host_timestamp_ns, datagram)
            except Exception as error:
                self._writer.abort()
                self._writer = None
                self._state = "failed"
                self._failure_reason = str(error)
                return
            if isinstance(packet, CsiPacket):
                self._counts[packet.node_id] += 1

    def stop(self) -> dict[str, Any]:
        with self.lock:
            self._advance(time.monotonic_ns())
            if self._state == "completed":
                return self._snapshot_locked(time.monotonic_ns())
            if self._state != "recording":
                raise SessionConflict("session is not recording")
            missing = [
                node for node in self.required_nodes if self._counts[node] == 0
            ]
            if missing:
                raise SessionConflict(
                    f"cannot complete without CSI frames from nodes {missing}"
                )
            self._complete()
            return self._snapshot_locked(time.monotonic_ns())

    def cancel(self) -> dict[str, Any]:
        with self.lock:
            self._advance(time.monotonic_ns())
            if self._state not in {"countdown", "recording"}:
                raise SessionConflict("there is no active session to cancel")
            if self._writer is not None:
                self._writer.abort()
                self._writer = None
            self._state = "cancelled"
            return self._snapshot_locked(time.monotonic_ns())

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            now_ns = time.monotonic_ns()
            self._advance(now_ns)
            return self._snapshot_locked(now_ns)

    def _snapshot_locked(self, now_ns: int) -> dict[str, Any]:
        if self._state == "countdown":
            remaining = max(
                0.0, (self._countdown_deadline_ns - now_ns) / 1_000_000_000
            )
        elif self._state == "recording":
            remaining = max(
                0.0, (self._recording_deadline_ns - now_ns) / 1_000_000_000
            )
        else:
            remaining = 0.0
        return {
            "state": self._state,
            "session_id": self._session_id,
            "started_at": self._started_at,
            "labels": self._labels.as_metadata() if self._labels else None,
            "remaining_seconds": remaining,
            "valid_frames": {
                str(node): self._counts[node] for node in self.required_nodes
            },
            "output": str(self._output) if self._output else None,
            "failure_reason": self._failure_reason,
        }
