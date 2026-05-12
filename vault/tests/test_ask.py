from pathlib import Path

from vault.agents.ask import run_ask
from vault.retrieval.chunker import Chunk
from vault.retrieval.embedder import embed_chunks
from vault.retrieval.store import upsert_chunks


def seed_test_data():
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
            text="Python is widely used in AI systems and data science.",
            chunk_index=1,
            chunk_id="python_notes__1",
        ),
        Chunk(
            note_title="Database Notes",
            note_path=Path("db_notes.md"),
            text="ChromaDB is a vector database used for semantic retrieval.",
            chunk_index=2,
            chunk_id="db_notes__2",
        ),
    ]

    embeddings = embed_chunks(
        [c.text for c in chunks],
        show_log=False,
    )

    upsert_chunks(chunks, embeddings)


def test_run_ask():
    # Seed fake notes into vector store
    seed_test_data()

    # Ask a question
    result = run_ask(
        query="What language is commonly used in AI?",
        top_k=3,
        show_eval=True,
    )

    print("\n" + "=" * 80)
    print("FINAL ASK RESULT")
    print("=" * 80)

    print("\nQuery:")
    print(result.query)

    print("\nModel:")
    print(result.model)

    print("\nSources:")
    print(result.sources)

    print("\nFaithfulness:")
    print(result.faithfulness)

    print("\nAnswer:")
    print(result.answer)

    print("\nRetrieved Chunks:")
    for chunk in result.retrieved_chunks:
        print("-" * 50)
        print(f"Title : {chunk.note_title}")
        print(f"Score : {chunk.score}")
        print(f"Chunk : {chunk.chunk_id}")
        print(f"Text  : {chunk.text}")


if __name__ == "__main__":
    test_run_ask()