from __future__ import annotations
 
from dataclasses import dataclass, field
 
from vault.cli.logger import blank, done, fail, log, section, step, warn
from vault.retrieval.embedder import embed_query
from vault.retrieval.store import retrieve, store_stats
from vault.agents.llm import GeminiBackend, build_rag_prompt
from vault.eval.faithfulness import FaithfulnessEvaluator


@dataclass
class AskResult:
    query: str
    answer: str
    sources: list[str] = field(default_factory=list)
    faithfulness: float = 0.0
    retrieved_chunks: list = field(default_factory=list)
    model: str = ""


def run_ask(
    query: str,
    top_k: int = 5,
    api_key: str | None = None,
    show_eval: bool = True,
) -> AskResult:
    section("vault ask")
    log(f"Question: [bold]{query}[/bold]")
    blank()

    # Sanity check
    stats = store_stats()
    if stats["total_chunks"] == 0:
        fail("ASK", "Vector store is empty. Run `vault init --path <your_vault>` first.")
        raise SystemExit(1)
 
    log(f"Vector store has {stats['total_chunks']} indexed chunks")
    blank()

    # Embed query
    step("EMBED", "Embedding your question into a semantic vector")
    log("The query is converted to the same 384-dim space as your note chunks")
    log("We then find chunks whose vectors are closest to this query vector")

    query_vector = embed_query(query)

    done("EMBED", "Query embedded (384-dim vector)")
    blank()

    # Retrieve top-k chunks
    chunks = retrieve(query_vector, top_k=top_k)
    if not chunks:
        warn("ASK", "No relevant chunks found. Try rephrasing your question.")
        return AskResult(query=query, answer="No relevant notes found.", sources=[])
    blank()

    # Build grounded prompt
    prompt = build_rag_prompt(query, chunks)
    blank()

    # Call LLM
    backend = GeminiBackend(api_key=api_key)
    llm_response = backend.ask(prompt)
    blank()

    # Evaluate faithfulness
    faithfulness = 0.0
    if show_eval:
        evaluator = FaithfulnessEvaluator()
        context = "\n\n".join(c.text for c in chunks)
        faithfulness = evaluator.score(llm_response.answer, context)
 
    sources = list(dict.fromkeys(c.note_title for c in chunks))
 
    return AskResult(
        query=query,
        answer=llm_response.answer,
        sources=sources,
        faithfulness=faithfulness,
        retrieved_chunks=chunks,
        model=llm_response.model,
    )