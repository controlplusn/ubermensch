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


def chunk_notes(
    notes,
    chunk_size: int = 200,
    overlap: int = 40
) -> list[Chunk]:
    # Called after ingestion
    step("CHUNK", "Splitting notes into embeddable chunks")
    log("Strategy: heading-aware splitting → word-count sliding window")
    log(f"Chunk size: {chunk_size} words  |  Overlap: {overlap} words")
    log("Overlap ensures sentences near boundaries aren't lost from retrieval")
 
    all_chunks: list[Chunk] = []
    total_skipped = 0
 
    for note in notes:
        note_chunks = _chunk_note(note.title, note.path, note.content,
                                  chunk_size, overlap)
        if not note_chunks:
            total_skipped += 1
            continue
        all_chunks.extend(note_chunks)
        log(f"  {note.title:<40}  {len(note_chunks)} chunk(s)")
 
    done(
        "CHUNK",
        f"{len(all_chunks)} chunks from {len(notes) - total_skipped} notes"
        + (f"  ({total_skipped} empty notes skipped)" if total_skipped else ""),
    )
    return all_chunks



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