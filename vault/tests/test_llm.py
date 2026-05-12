from vault.agents.llm import (
    GeminiBackend,
    build_rag_prompt,
)

from vault.retrieval.store import RetrievedChunk


def test_build_prompt():
    chunks = [
        RetrievedChunk(
            note_title="AI Notes",
            note_path="ai_notes.md",
            text="Machine learning is a subset of AI.",
            score=0.92,
            chunk_id="ai_0",
        ),
        RetrievedChunk(
            note_title="Python Notes",
            note_path="python_notes.md",
            text="Python is widely used in AI systems.",
            score=0.89,
            chunk_id="py_0",
        ),
    ]

    query = "What language is used in AI?"

    prompt = build_rag_prompt(query, chunks)

    print("\n===== GENERATED PROMPT =====\n")
    print(prompt)


def test_gemini():
    chunks = [
        RetrievedChunk(
            note_title="AI Notes",
            note_path="ai_notes.md",
            text="Machine learning is a subset of AI.",
            score=0.92,
            chunk_id="ai_0",
        ),
        RetrievedChunk(
            note_title="Python Notes",
            note_path="python_notes.md",
            text="Python is widely used in AI systems.",
            score=0.89,
            chunk_id="py_0",
        ),
    ]

    query = "What language is used in AI?"

    # Build RAG prompt
    prompt = build_rag_prompt(query, chunks)

    # Call Gemini
    llm = GeminiBackend()
    response = llm.ask(prompt)

    print("\n===== LLM RESPONSE =====\n")
    print(f"Model: {response.model}")
    print("\nAnswer:")
    print(response.answer)


if __name__ == "__main__":
    test_build_prompt()

    print("\n" + "=" * 80 + "\n")

    test_gemini()