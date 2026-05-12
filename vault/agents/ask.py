from __future__ import annotations
 
from dataclasses import dataclass, field
 
from vault.cli.logger import blank, done, fail, log, section, step, warn
from vault.retrieval.embedder import embed_query
from vault.retrieval.store import retrieve, store_stats
from vault.agents.llm import GeminiBackend, build_rag_prompt
from vault.eval.faithfulness import FaithfulnessEvaluator, EvalResult

MAX_RETRIES = 2
RE_RETRIEVAL_THRESHOLD = 0.4


@dataclass
class AskResult:
    query: str
    answer: str
    sources: list[str] = field(default_factory=list)
    faithfulness: float = 0.0
    eval_result: object = None   # EvalResult
    retrieved_chunks: list = field(default_factory=list)
    model: str = ""
    attempts: int = 1


def run_ask(
    query: str,
    top_k: int = 5,
    api_key: str | None = None,
    show_eval: bool = True,
    eval_verbose: bool = False,
) -> AskResult:
    section("vault ask")
    log(f"Question: [bold]{query}[/bold]")
    blank()

    # Sanity check
    stats = store_stats()
    if stats["total_chunks"] == 0:
        from vault.cli.logger import fail
        fail("ASK", "Vector store is empty. Run `vault init --path <your_vault>` first.")
        raise SystemExit(1)
 
    log(f"Vector store: {stats['total_chunks']} indexed chunks")
    blank()
 
    evaluator = FaithfulnessEvaluator(re_retrieval_threshold=RE_RETRIEVAL_THRESHOLD)
    backend   = GeminiBackend(api_key=api_key)
    best: AskResult | None = None

    for attempt in range(1, MAX_RETRIES + 2):   # attempts 1, 2, 3
        current_top_k = top_k * attempt          # 5 → 10 → 15
        current_query = _expand_query(query, attempt)
 
        if attempt > 1:
            blank()
            step(
                "RETRY",
                f"Attempt {attempt}/{MAX_RETRIES + 1} — "
                f"expanding query and retrieving {current_top_k} chunks",
            )
            log(f"Expanded query: \"{current_query}\"")
            log("Re-retrieval broadens the context window to find supporting evidence")

        # Embed
        step("EMBED", "Embedding query into semantic vector")
        log("Converting question to 384-dim vector for similarity search")
        query_vector = embed_query(current_query)
        done("EMBED", "Query embedded")
        blank()

        # Retrieve
        chunks = retrieve(query_vector, top_k=current_top_k)
        if not chunks:
            warn("ASK", "No relevant chunks found — try rephrasing your question")
            return AskResult(query=query, answer="No relevant notes found.", sources=[])
        blank()

        # Build prompt
        prompt = build_rag_prompt(query, chunks)
        blank()

        # Call LLM
        llm_response = backend.ask(prompt)
        blank()

        # Evaluate
        eval_result: EvalResult | None = None
        faith_score = 0.0
 
        if show_eval:
            eval_result = evaluator.evaluate(
                llm_response.answer,
                "\n\n".join(c.text for c in chunks),
                verbose=eval_verbose,
            )
            faith_score = eval_result.score
            blank()
        else:
            faith_score = 1.0

        sources = list(dict.fromkeys(c.note_title for c in chunks))

        result = AskResult(
            query=query,
            answer=llm_response.answer,
            sources=sources,
            faithfulness=faith_score,
            eval_result=eval_result,
            retrieved_chunks=chunks,
            model=llm_response.model,
            attempts=attempt,
        )

        # Decide whether to re-retrieve
        if best is None or faith_score > best.faithfulness:
            best = result
 
        if not show_eval:
            break
 
        if not eval_result.needs_retrieval:
            log(f"[green]Faithfulness acceptable ({faith_score:.1%}) — stopping[/green]")
            break
 
        if attempt == MAX_RETRIES + 1:
            warn(
                "RETRY",
                f"Max retries reached. Best faithfulness: {best.faithfulness:.1%}",
            )
            log("Returning best result seen across all attempts")
 
    return best


# Query expansion
def _expand_query(query: str, attempt: int) -> str:
    if attempt == 1:
        return query
 
    if attempt == 2:
        # Encourage the embedder to match more general content
        return f"explain summarize overview {query}"
 
    # attempt 3+: strip question words to get to core topic
    strip_words = {"what", "how", "why", "when", "where", "who", "which",
                   "is", "are", "was", "were", "do", "does", "did",
                   "can", "could", "would", "should", "my", "i", "me"}
    words = [w for w in query.lower().split() if w not in strip_words]
    return " ".join(words) if words else query