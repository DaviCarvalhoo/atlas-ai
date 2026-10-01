"""Named SQL analytics loaded from ``sql/analytics.sql`` and executed read-only."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from atlas.db import read_only_query

ANALYTICS_PATH = Path(__file__).resolve().parents[2] / "sql" / "analytics.sql"


@dataclass(frozen=True)
class NamedQuery:
    name: str
    description: str
    sql: str


@lru_cache
def load_queries(path: Path = ANALYTICS_PATH) -> dict[str, NamedQuery]:
    queries = {}
    for block in re.split(r"^-- name:\s*", path.read_text(encoding="utf-8"), flags=re.MULTILINE)[
        1:
    ]:
        name, _, rest = block.partition("\n")
        desc = re.search(r"^-- description:\s*(.+)$", rest, re.MULTILINE)
        sql = (
            "\n".join(ln for ln in rest.splitlines() if not ln.startswith("--")).strip().rstrip(";")
        )
        queries[name.strip()] = NamedQuery(name.strip(), desc.group(1).strip() if desc else "", sql)
    return queries


def run_named(name: str) -> pd.DataFrame:
    queries = load_queries()
    if name not in queries:
        raise KeyError(f"Unknown analytics query '{name}'. Available: {sorted(queries)}")
    return read_only_query(queries[name].sql)
