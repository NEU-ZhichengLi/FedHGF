"""Server-visible communication messages."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClientMessage:
    client_id: str
    round_id: int
    payload_type: str
    vector: torch.Tensor
    sample_count: int

