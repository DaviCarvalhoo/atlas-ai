"""Dependency-free Prometheus-style metrics (counters + latency summaries)."""

from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters: dict[tuple[str, tuple], float] = defaultdict(float)
        self.latency_sum: dict[str, float] = defaultdict(float)
        self.latency_count: dict[str, int] = defaultdict(int)
        self.latency_buckets: dict[tuple[str, float], int] = defaultdict(int)
        self.buckets = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self.counters[(name, tuple(sorted(labels.items())))] += value

    def observe_latency(self, route: str, seconds: float) -> None:
        with self._lock:
            self.latency_sum[route] += seconds
            self.latency_count[route] += 1
            for b in self.buckets:
                if seconds <= b:
                    self.latency_buckets[(route, b)] += 1

    def render(self) -> str:
        lines = []
        with self._lock:
            for (name, labels), value in sorted(self.counters.items()):
                lbl = ",".join(f'{k}="{v}"' for k, v in labels)
                lines.append(f"{name}{{{lbl}}} {value}")
            for route in sorted(self.latency_count):
                for b in self.buckets:
                    lines.append(
                        f'atlas_request_seconds_bucket{{route="{route}",le="{b}"}} '
                        f"{self.latency_buckets[(route, b)]}"
                    )
                lines.append(
                    f'atlas_request_seconds_bucket{{route="{route}",le="+Inf"}} '
                    f"{self.latency_count[route]}"
                )
                lines.append(
                    f'atlas_request_seconds_sum{{route="{route}"}} {self.latency_sum[route]:.6f}'
                )
                lines.append(
                    f'atlas_request_seconds_count{{route="{route}"}} {self.latency_count[route]}'
                )
        return "\n".join(lines) + "\n"


METRICS = Metrics()
