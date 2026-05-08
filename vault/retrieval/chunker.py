# Splits note content into overlapping chunks, respecting markdown headings

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
 
HEADING_RE = re.compile(r"^#{1,3} .+", re.MULTILINE)
 
@dataclass
class Chunk:
    note_title: str
    note_path: Path
    text: str
    chunk_index: int
 
def chunk_note(title: str, path: Path, content: str,
               chunk_size: int = 512, overlap: int = 64) -> list[Chunk]:
    sections = HEADING_RE.split(content)
    chunks, idx = [], 0
    for section in sections:
        words = section.split()
        for i in range(0, max(1, len(words) - overlap), chunk_size - overlap):
            chunk_text = " ".join(words[i:i + chunk_size])
            if chunk_text.strip():
                chunks.append(Chunk(title, path, chunk_text.strip(), idx))
                idx += 1
    return chunks