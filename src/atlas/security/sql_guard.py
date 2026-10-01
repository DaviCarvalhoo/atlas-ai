"""Validates SQL produced by the agent / LLM before execution.

Rules: single statement, SELECT/WITH only, no write/DDL keywords, only whitelisted relations,
and a mandatory LIMIT. Execution additionally happens on a read-only transaction (see ``db``).
"""

from __future__ import annotations

import re

ALLOWED_TABLES = {"orders", "tickets", "v_customer_safe"}
MAX_LIMIT = 100

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|attach|detach|pragma|"
    r"replace|merge|copy|vacuum|exec|execute|call)\b",
    re.IGNORECASE,
)
RELATION = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][\w.]*)", re.IGNORECASE)
CTE_NAME = re.compile(r"(?:\bwith|,)\s*([a-zA-Z_]\w*)\s+as\s*\(", re.IGNORECASE)
LIMIT = re.compile(r"\blimit\s+(\d+)\s*$", re.IGNORECASE)


class UnsafeSQLError(ValueError):
    """Raised when a query violates the guard policy."""


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)


def validate_sql(sql: str) -> str:
    """Return a normalised, safe query or raise :class:`UnsafeSQLError`."""
    cleaned = _strip_comments(sql).strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeSQLError("empty query")
    if ";" in cleaned:
        raise UnsafeSQLError("multiple statements are not allowed")
    if not re.match(r"^(select|with)\b", cleaned, re.IGNORECASE):
        raise UnsafeSQLError("only SELECT queries are allowed")
    if m := FORBIDDEN.search(cleaned):
        raise UnsafeSQLError(f"forbidden keyword: {m.group(1).upper()}")

    ctes = {c.lower() for c in CTE_NAME.findall(cleaned)}
    for rel in RELATION.findall(cleaned):
        name = rel.lower()
        if name not in ALLOWED_TABLES and name not in ctes:
            raise UnsafeSQLError(f"relation not allowed: {rel}")

    if m := LIMIT.search(cleaned):
        if int(m.group(1)) > MAX_LIMIT:
            cleaned = LIMIT.sub(f"LIMIT {MAX_LIMIT}", cleaned)
    else:
        cleaned = f"{cleaned} LIMIT {MAX_LIMIT}"
    return cleaned
