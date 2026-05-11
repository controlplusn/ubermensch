from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from vault.cli.logger import done, log, step

HEADING_RE = re.compile(r"^#{1,3} .+", re.MULTILINE)


@dataclass
class Chunk:
    note_title: str
    note_path: Path
    text: str
    chunk_index: int
    chunk_id: str


def _chunk_note(
    title: str,
    path: Path,
    content: str,
    chunk_size: int,
    overlap: int
) -> list[Chunk]:
    if not content.strip():
        return []
    
    # Split heading
    sections = HEADING_RE.split(content)
    sections = [s.strip() for s in sections if s.strip()]

    chunks = []
    idx = 0

    for section in sections:
        words = section.split()
        if not words:
            continue

        # Sliding window within section
        start = 0
        words_len = len(words)

        while start < words_len:
            end = min(start + chunk_size, words_len)
            chunk_text = " ".join(words[start:end]).strip()

            if chunk_text:
                chunk_id = f"{path.stem}__{idx}"
                chunks.append(Chunk(
                    note_title=title,
                    note_path=path,
                    text=chunk_text,
                    chunk_index=idx,
                    chunk_id=chunk_id,
                ))
                idx += 1

            if end == words_len:
                break

            start += chunk_size - overlap   # Slide forward with overlap

    return chunks