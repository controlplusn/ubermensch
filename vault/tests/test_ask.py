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
            text=(
                "Machine learning is a subset of artificial intelligence. "
                "Python is widely used for machine learning and data science."
            ),
            chunk_index=0,
            chunk_id="ai_notes__0",
        ),
        Chunk(
            note_title="Database Notes",
            note_path=Path("db_notes.md"),
            text=(
                "ChromaDB is a vector database used for semantic retrieval "
                "in retrieval augmented generation systems."
            ),
            chunk_index=1,
            chunk_id="db_notes__1",
        ),
        Chunk(
            note_title="Philosophy Notes",
            note_path=Path("philosophy.md"),
            text=(
                "Nietzsche introduced the idea of the Ubermensch "
                "as a model of self-overcoming."
            ),
            chunk_index=2,
            chunk_id="philosophy__2",
        ),
    ]

    embeddings = embed_chunks(
        [c.text for c in chunks],
        show_log=False,
    )

    upsert_chunks(chunks, embeddings)


def print_result(result):
    print("\n" + "=" * 80)
    print("FINAL ASK RESULT")
    print("=" * 80)

    print("\nQuery:")
    print(result.query)

    print("\nAttempts:")
    print(result.attempts)

    print("\nModel:")
    print(result.model)

    print("\nFaithfulness:")
    print(f"{result.faithfulness:.1%}")

    if result.eval_result:
        print("\nConfidence:")
        print(result.eval_result.confidence_label)

        print("\nSupported Claims:")
        print(
            f"{result.eval_result.supported_count}/"
            f"{result.eval_result.total_count}"
        )

        print("\nNeeds Retrieval:")
        print(result.eval_result.needs_retrieval)

    print("\nSources:")
    for s in result.sources:
        print(f"- {s}")

    print("\nAnswer:")
    print(result.answer)

    print("\nRetrieved Chunks:")
    for chunk in result.retrieved_chunks:
        print("-" * 50)
        print(f"Title : {chunk.note_title}")
        print(f"Score : {chunk.score}")
        print(f"Chunk : {chunk.chunk_id}")
        print(f"Text  : {chunk.text[:120]}...")


def test_successful_query():
    print("\n" + "#" * 80)
    print("TEST 1 — SUCCESSFUL HIGH-FAITHFULNESS QUERY")
    print("#" * 80)

    result = run_ask(
        query="What language is commonly used in machine learning?",
        top_k=3,
        show_eval=True,
        eval_verbose=True,
    )

    print_result(result)


def test_query_with_reretrieval():
    print("\n" + "#" * 80)
    print("TEST 2 — QUERY THAT MAY REQUIRE RE-RETRIEVAL")
    print("#" * 80)

    result = run_ask(
        query="Who created vector databases for neural intelligence systems?",
        top_k=2,
        show_eval=True,
        eval_verbose=True,
    )

    print_result(result)


def test_no_eval_mode():
    print("\n" + "#" * 80)
    print("TEST 3 — NO EVALUATION MODE")
    print("#" * 80)

    result = run_ask(
        query="What is ChromaDB?",
        top_k=2,
        show_eval=False,
    )

    print_result(result)


if __name__ == "__main__":
    seed_test_data()

    test_successful_query()

    test_query_with_reretrieval()

    test_no_eval_mode()