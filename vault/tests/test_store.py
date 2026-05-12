from pathlib import Path

from vault.retrieval.chunker import Chunk
from vault.retrieval.embedder import embed_chunks, embed_query
from vault.retrieval.store import (
    upsert_chunks,
    retrieve,
    store_stats,
)


def test_store_pipeline():
    chunks = [
        Chunk(
            note_title="AI Notes",
            note_path=Path("ai_notes.md"),
            text="Machine learning is a subset of artificial intelligence.",
            chunk_index=0,
            chunk_id="ai_notes__0",
        ),
        Chunk(
            note_title="Python Notes",
            note_path=Path("python_notes.md"),
            text="Python is commonly used for AI and data science.",
            chunk_index=1,
            chunk_id="python_notes__1",
        ),
    ]

    # Convert text to embeddings
    embeddings = embed_chunks(
        [c.text for c in chunks],
        show_log=False,
    )

    # Store in ChromaDB
    upsert_chunks(chunks, embeddings)

    # Print stats
    stats = store_stats()
    print("\nStore stats:")
    print(stats)

    # Query
    query = "What language is used in AI?"
    query_embedding = embed_query(query)

    # Retrieve similar chunks
    results = retrieve(query_embedding, top_k=3)

    print("\nRetrieved Results:")
    for r in results:
        print("-" * 50)
        print(f"Title : {r.note_title}")
        print(f"Score : {r.score}")
        print(f"Chunk : {r.chunk_id}")
        print(f"Text  : {r.text}")


if __name__ == "__main__":
    test_store_pipeline()