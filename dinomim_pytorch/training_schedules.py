"""Cosine schedules for LR, weight decay, and teacher EMA momentum."""

from __future__ import annotations

import math
from typing import Tuple


def cosine_schedule(
    step: int, total_steps: int, start: float, end: float, floor: bool = False
) -> float:
    if total_steps <= 0:
        return end
    t = min(max(step, 0), total_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * t / max(total_steps, 1)))
    v = end + (start - end) * cos
    if floor:
        return max(v, 0.0)
    return v


__all__ = ["cosine_schedule"]
