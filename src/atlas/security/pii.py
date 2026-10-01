"""PII detection & masking (LGPD). Applied to every text before it reaches an LLM or a log."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "CPF": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    "CNPJ": re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "PHONE": re.compile(r"(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}-?\d{4}\b"),
}


@dataclass
class MaskResult:
    text: str
    found: dict[str, int] = field(default_factory=dict)

    @property
    def has_pii(self) -> bool:
        return bool(self.found)


def mask_pii(text: str) -> MaskResult:
    """Replace PII with typed placeholders, e.g. ``[CPF]``. Order matters: CNPJ before CPF/PHONE."""
    found: dict[str, int] = {}
    for label in ("EMAIL", "CNPJ", "CPF", "CARD", "PHONE"):
        text, n = PATTERNS[label].subn(f"[{label}]", text)
        if n:
            found[label] = n
    return MaskResult(text=text, found=found)
