"""Input guardrails: prompt-injection heuristics and size limits, plus a JSONL audit log."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

MAX_INPUT_CHARS = 4000

INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) (instructions|prompts?)",
    r"ignore (as |todas as )?instru[cç][oõ]es anteriores",
    r"desconsidere (as )?instru[cç][oõ]es",
    r"you are now",
    r"voc[eê] agora [eé]",
    r"system prompt",
    r"prompt do sistema",
    r"reveal (your|the) (instructions|prompt|secrets?)",
    r"act as (an? )?(admin|developer|dba)",
    r"\b(drop|truncate)\s+table\b",
]
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


@dataclass
class GuardResult:
    allowed: bool
    reason: str | None = None


def check_input(text: str) -> GuardResult:
    if not text or not text.strip():
        return GuardResult(False, "empty input")
    if len(text) > MAX_INPUT_CHARS:
        return GuardResult(False, f"input exceeds {MAX_INPUT_CHARS} characters")
    if m := _INJECTION.search(text):
        return GuardResult(False, f"possible prompt injection: '{m.group(0)}'")
    return GuardResult(True)


class AuditLog:
    """Append-only JSONL audit trail. Stores a hash of the (already masked) input, never raw PII."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, event: str, payload: dict) -> None:
        record = {"ts": datetime.now(UTC).isoformat(), "event": event, **payload}
        if "input" in record:
            record["input_sha256"] = hashlib.sha256(record.pop("input").encode()).hexdigest()[:16]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
