"""Structure-aware chunking for markdown: one chunk per ``##`` section, with title context."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    title: str
    section: str
    text: str

    @property
    def embedding_text(self) -> str:
        # Prepending title/section gives each chunk standalone meaning (contextual chunking).
        return f"{self.title} — {self.section}\n{self.text}"


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts, buf = [], ""
    for p in paragraphs:
        if buf and len(buf) + len(p) > max_chars:
            parts.append(buf.strip())
            buf = buf[-overlap:]
        buf += "\n\n" + p
    if buf.strip():
        parts.append(buf.strip())
    return parts


def chunk_markdown(path: Path, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem
    sections = re.split(r"^##\s+", raw, flags=re.MULTILINE)[1:]
    chunks = []
    for sec in sections:
        heading, _, body = sec.partition("\n")
        for i, part in enumerate(_split_long(body.strip(), max_chars, overlap)):
            cid = hashlib.md5(f"{path.name}:{heading}:{i}".encode()).hexdigest()[:12]
            chunks.append(Chunk(cid, path.name, title, heading.strip(), part))
    return chunks


def load_corpus(kb_dir: Path) -> list[Chunk]:
    return [c for p in sorted(kb_dir.glob("*.md")) for c in chunk_markdown(p)]
