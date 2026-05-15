from __future__ import annotations
 
from dataclasses import dataclass, field
 
from vault.cli.logger import blank, done, fail, log, section, step, warn
from vault.retrieval.embedder import embed_query
from vault.retrieval.store import retrieve, store_stats
from vault.agents.llm import GeminiBackend, build_rag_prompt, get_backend
from vault.eval.faithfulness import FaithfulnessEvaluator, EvalResult

MAX_RETRIES = 2
RE_RETRIEVAL_THRESHOLD = 0.4


@dataclass
class AskResult:
    query: str
    answer: str
    sources: list[str] = field(default_factory=list)
    faithfulness: float = 0.0
    eval_result: object = None
    retrieved_chunks: list = field(default_factory=list)
    model: str = ""
    attempts: int = 1


def run_ask(
    query: str,
    top_k: int  = 5,
    api_key: str | None = None,
    show_eval: bool = True,
    eval_verbose: bool = False,
    backend_name: str  = "gemini",
    model: str  = ""
) -> AskResult:
    section("vault ask")
    log(f"Question: [bold]{query}[/bold]")
    blank()

    # Sanity check
    stats = store_stats()
    if stats["total_chunks"] == 0:
        fail("ASK", "Vector store is empty. Run `vault init --path <your_vault>` first.")
        raise SystemExit(1)
 
    log(f"Vector store: {stats['total_chunks']} indexed chunks")
    blank()
 
    evaluator = FaithfulnessEvaluator(re_retrieval_threshold=RE_RETRIEVAL_THRESHOLD)
    backend = get_backend(backend_name, api_key, model)
    best: AskResult | None = None


    for attempt in range(1, MAX_RETRIES + 2):
        current_top_k = top_k * attempt
        current_query = _expand_query(query, attempt)
 
        if attempt > 1:
            blank()
            step(
                "RETRY",
                f"Attempt {attempt}/{MAX_RETRIES + 1} — "
                f"expanding query, retrieving {current_top_k} chunks",
            )
            log(f"Expanded query: \"{current_query}\"")
 
        step("EMBED", "Embedding query into semantic vector")
        query_vector = embed_query(current_query)
        done("EMBED", "Query embedded")
        blank()
 
        chunks = retrieve(query_vector, top_k=current_top_k)
        if not chunks:
            warn("ASK", "No relevant chunks found — try rephrasing")
            return AskResult(query=query, answer="No relevant notes found.", sources=[])
        blank()
 
        prompt       = build_rag_prompt(query, chunks)
        blank()
 
        llm_response = backend.ask(prompt)
        blank()
 
        faith_score  = 0.0
        eval_result  = None
 
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
 
        if best is None or faith_score > best.faithfulness:
            best = result
 
        if not show_eval:
            break
 
        if not eval_result.needs_retrieval:
            break
 
        if attempt == MAX_RETRIES + 1:
            warn("RETRY", f"Max retries reached. Best: {best.faithfulness:.1%}")
 
    return best


# Query expansion
def _expand_query(query: str, attempt: int) -> str:
    if attempt == 1:
        return query
 
    if attempt == 2:
        # Encourage the embedder to match more general content
        return f"explain summarize overview {query}"
 
    # attempt 3+: strip question words to get to core topic
    strip_words = {
        "what", "how", "why", "when", "where", "who", "which",
        "is", "are", "was", "were", "do", "does", "did",
        "can", "could", "would", "should", "my", "i", "me",
    }
    words = [w for w in query.lower().split() if w not in strip_words]
    return " ".join(words) if words else query