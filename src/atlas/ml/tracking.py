"""Thin MLflow wrapper: tracking is on when MLflow is importable, silently a no-op otherwise."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path

log = logging.getLogger(__name__)

try:  # pragma: no cover - import guard
    import mlflow

    _HAS_MLFLOW = True
except Exception:  # pragma: no cover
    mlflow = None
    _HAS_MLFLOW = False


class Tracker:
    def __init__(self, tracking_uri: str, experiment: str = "atlas", enabled: bool = True):
        self.enabled = enabled and _HAS_MLFLOW
        if self.enabled:
            try:
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment)
            except Exception as exc:  # backend unavailable → keep training, just don't track
                log.warning("MLflow disabled: %s", exc)
                self.enabled = False

    @contextlib.contextmanager
    def run(self, name: str, nested: bool = False, tags: dict | None = None) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        with mlflow.start_run(run_name=name, nested=nested, tags=tags or {}):
            yield

    def log_params(self, params: dict) -> None:
        if self.enabled:
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

    def log_metrics(self, metrics: dict) -> None:
        if self.enabled:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                if isinstance(v, int | float)})

    def log_artifact(self, path: Path) -> None:
        if self.enabled:
            mlflow.log_artifact(str(path))
