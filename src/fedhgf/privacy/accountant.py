"""Lightweight privacy release ledger."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrivacyEvent:
    round_id: int
    channel: str
    cohort_size: int
    max_weight: float
    clip_norm: float
    noise_multiplier: float


class PrivacyAccountant:
    def __init__(self) -> None:
        self.events: list[PrivacyEvent] = []

    def record(self, event: PrivacyEvent) -> None:
        self.events.append(event)

    def summary(self) -> dict:
        return {
            "num_releases": len(self.events),
            "channels": sorted({e.channel for e in self.events}),
            "cohort_sizes": [e.cohort_size for e in self.events],
            "noise_multipliers": [e.noise_multiplier for e in self.events],
        }

