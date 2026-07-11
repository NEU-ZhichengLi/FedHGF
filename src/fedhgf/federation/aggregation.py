"""Aggregation backends."""

from __future__ import annotations

from typing import Protocol

import torch

from ..privacy.accountant import PrivacyAccountant, PrivacyEvent
from ..privacy.noise import clip_and_noise
from .messages import ClientMessage


class Aggregator(Protocol):
    def aggregate(self, messages: list[ClientMessage]) -> torch.Tensor:
        ...


class PlainAggregator:
    def aggregate(self, messages: list[ClientMessage]) -> torch.Tensor:
        total = sum(max(m.sample_count, 0) for m in messages)
        if total <= 0:
            raise ValueError("messages must contain positive sample counts")
        out = None
        for message in messages:
            weight = message.sample_count / total
            value = message.vector.detach() * weight
            out = value.clone() if out is None else out + value
        assert out is not None
        return out


class DPSimulatedAggregator:
    """Simulated DP release for server-visible client messages.

    This backend clips and noises the payload before weighted aggregation. It is
    not secure aggregation; it models only the DP release seen by the server.
    """

    backend = "dp_simulator"

    def __init__(
        self,
        clip_norm: float,
        noise_multiplier: float,
        accountant: PrivacyAccountant | None = None,
    ) -> None:
        self.clip_norm = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.accountant = accountant or PrivacyAccountant()

    def aggregate(self, messages: list[ClientMessage]) -> torch.Tensor:
        noised = [
            ClientMessage(
                client_id=m.client_id,
                round_id=m.round_id,
                payload_type=m.payload_type,
                vector=clip_and_noise(m.vector, self.clip_norm, self.noise_multiplier),
                sample_count=m.sample_count,
            )
            for m in messages
        ]
        if messages:
            first = messages[0]
            total = max(1, sum(m.sample_count for m in messages))
            self.accountant.record(PrivacyEvent(
                round_id=first.round_id,
                channel=first.payload_type,
                cohort_size=len(messages),
                max_weight=max(m.sample_count for m in messages) / total,
                clip_norm=self.clip_norm,
                noise_multiplier=self.noise_multiplier,
            ))
        return PlainAggregator().aggregate(noised)


class AssumedSecAggAggregator:
    """Secure-aggregation boundary marker.

    This repository does not implement cryptographic secure aggregation. The
    class performs ordinary weighted aggregation while making the experiment
    backend explicit as an assumption/simulator.
    """

    backend = "assumed"

    def aggregate(self, messages: list[ClientMessage]) -> torch.Tensor:
        return PlainAggregator().aggregate(messages)
