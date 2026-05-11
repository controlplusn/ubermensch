from pathlib import Path

from vault.retrieval.chunker import _chunk_note


def test_chunk_note():
    content = """
    # Introduction
    This is the first section of the note. It contains several words
    to demonstrate chunking behavior.

    ## Details
    This is another section with more text so we can observe overlaps
    between chunks properly.
    """

    chunks = _chunk_note(
        title="Sample Note",
        path=Path("sample_note.md"),
        content=content,
        chunk_size=10,
        overlap=4,
    )

    for chunk in chunks:
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Index: {chunk.chunk_index}")
        print(f"Text: {chunk.text}")
        print("-" * 50)


if __name__ == "__main__":
    test_chunk_note()