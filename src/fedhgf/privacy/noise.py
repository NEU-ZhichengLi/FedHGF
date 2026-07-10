"""Clipping and Gaussian noise for uploaded messages."""

from __future__ import annotations

import torch


def clip_and_noise(vector: torch.Tensor, clip_norm: float, noise_multiplier: float) -> torch.Tensor:
    flat = vector.detach().reshape(-1)
    norm = torch.linalg.vector_norm(flat)
    scale = torch.clamp(torch.tensor(clip_norm, device=flat.device) / (norm + 1e-12), max=1.0)
    clipped = flat * scale
    if noise_multiplier > 0:
        clipped = clipped + torch.randn_like(clipped) * (noise_multiplier * clip_norm)
    return clipped.reshape_as(vector)

