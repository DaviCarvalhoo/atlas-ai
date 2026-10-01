"""Prediction-drift monitoring with the Population Stability Index (PSI).

PSI < 0.1: stable · 0.1–0.25: moderate shift · > 0.25: significant shift (investigate / retrain).
"""

from __future__ import annotations

import math
import threading
from collections import Counter, deque


def psi(expected: dict[str, float], actual: dict[str, float], eps: float = 1e-4) -> float:
    keys = set(expected) | set(actual)
    total = 0.0
    for k in keys:
        e, a = max(expected.get(k, 0.0), eps), max(actual.get(k, 0.0), eps)
        total += (a - e) * math.log(a / e)
    return round(total, 4)


def psi_status(value: float) -> str:
    return "stable" if value < 0.1 else "moderate" if value < 0.25 else "significant"


class DriftMonitor:
    """Keeps a sliding window of recent predictions and compares it to the training reference."""

    def __init__(
        self, reference: dict[str, dict[str, float]], window: int = 1000, min_samples: int = 50
    ):
        self.reference = reference
        self.min_samples = min_samples
        self.windows: dict[str, deque[str]] = {k: deque(maxlen=window) for k in reference}
        self._lock = threading.Lock()

    def record(self, field: str, value: str) -> None:
        if field in self.windows:
            with self._lock:
                self.windows[field].append(value)

    def report(self) -> dict:
        out = {}
        with self._lock:
            for field, window in self.windows.items():
                n = len(window)
                if n < self.min_samples:
                    out[field] = {"samples": n, "status": "insufficient_data"}
                    continue
                actual = {k: v / n for k, v in Counter(window).items()}
                value = psi(self.reference[field], actual)
                out[field] = {
                    "samples": n,
                    "psi": value,
                    "status": psi_status(value),
                    "current_distribution": {k: round(v, 3) for k, v in actual.items()},
                }
        return out
