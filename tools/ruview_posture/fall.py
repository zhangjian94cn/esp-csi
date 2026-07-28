"""Temporal fall suspicion logic for the research-only posture runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FallDecision:
    event: str
    latched_until_ms: int | None


class FallDetector:
    def __init__(
        self,
        *,
        motion_threshold: float,
        transition_ms: int = 1500,
        lying_confirmations: int = 2,
        latch_ms: int = 10_000,
    ):
        self.motion_threshold = motion_threshold
        self.transition_ms = transition_ms
        self.lying_confirmations = lying_confirmations
        self.latch_ms = latch_ms
        self.previous_posture = "unknown"
        self.transition_started_ms: int | None = None
        self.lying_count = 0
        self.latched_until_ms: int | None = None

    def update(
        self, *, now_ms: int, posture: str, motion_energy: float
    ) -> FallDecision:
        if self.latched_until_ms is not None and now_ms < self.latched_until_ms:
            self.previous_posture = posture
            return FallDecision("suspected", self.latched_until_ms)
        self.latched_until_ms = None

        if (
            self.previous_posture in {"standing", "sitting"}
            and motion_energy >= self.motion_threshold
        ):
            self.transition_started_ms = now_ms
            self.lying_count = 0

        if self.transition_started_ms is not None:
            if now_ms - self.transition_started_ms > self.transition_ms:
                self.transition_started_ms = None
                self.lying_count = 0
            elif posture == "lying":
                self.lying_count += 1
                if self.lying_count >= self.lying_confirmations:
                    self.latched_until_ms = now_ms + self.latch_ms
                    self.transition_started_ms = None
                    self.lying_count = 0
                    self.previous_posture = posture
                    return FallDecision("suspected", self.latched_until_ms)
            elif posture not in {"unknown", "lying"}:
                self.lying_count = 0

        self.previous_posture = posture
        return FallDecision("none", None)
