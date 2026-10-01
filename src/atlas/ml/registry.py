"""Minimal file-based model registry: versioned artifacts + metadata + 'production' alias.

Mirrors what MLflow Model Registry / Azure ML provide, without requiring a server, so the API
can load models offline. Every training run is also tracked in MLflow (see ``train``).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ModelVersion:
    name: str
    version: int
    created_at: str
    metrics: dict[str, float]
    params: dict[str, Any]
    data_fingerprint: str
    git_sha: str | None
    path: str


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


class ModelRegistry:
    def __init__(self, root: Path):
        self.root = root

    def _dir(self, name: str) -> Path:
        return self.root / name

    def versions(self, name: str) -> list[int]:
        d = self._dir(name)
        if not d.exists():
            return []
        return sorted(int(p.name[1:]) for p in d.glob("v*") if p.name[1:].isdigit())

    def register(self, name: str, model: Any, metrics: dict, params: dict,
                 data_fingerprint: str, promote: bool = True) -> ModelVersion:
        version = (self.versions(name) or [0])[-1] + 1
        vdir = self._dir(name) / f"v{version}"
        vdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, vdir / "model.joblib")
        mv = ModelVersion(name, version, datetime.now(UTC).isoformat(), metrics, params,
                          data_fingerprint, _git_sha(), str(vdir / "model.joblib"))
        (vdir / "metadata.json").write_text(json.dumps(asdict(mv), indent=2), encoding="utf-8")
        if promote:
            self.promote(name, version)
        return mv

    def promote(self, name: str, version: int) -> None:
        (self._dir(name) / "production").write_text(str(version), encoding="utf-8")

    def production_version(self, name: str) -> int | None:
        alias = self._dir(name) / "production"
        return int(alias.read_text()) if alias.exists() else None

    def metadata(self, name: str, version: int | None = None) -> ModelVersion:
        version = version or self.production_version(name)
        if version is None:
            raise FileNotFoundError(f"No production version for model '{name}'. Run `atlas train`.")
        data = json.loads((self._dir(name) / f"v{version}" / "metadata.json").read_text())
        return ModelVersion(**data)

    def load(self, name: str, version: int | None = None) -> Any:
        return joblib.load(self.metadata(name, version).path)
