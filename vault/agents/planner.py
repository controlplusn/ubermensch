"""
Decomposes complex queries into sub-questions, answers each one,
then synthesizes a final unified answer.

What decides a "complex" query?
Simple: "What is attention mechanism?"
        -> Single RAG retrieval is enough

Complex: "What are all the key ideas in my LLM research and how do they connect?"
        -> needs multiple retrieval passes across different note clusters
        -> needs synthesis across answers

Pipeline:
    1. Decompose - LLM breaks the query into N sub-questions
    2. Retrieve - RAG answers each sub-question independently
    3. Synthesize - LLM combines all sub-answers into one final answers
    4. Eval - Faithfulness scored against all retrieved chunks
"""

from __future__ import annotations
 
import re
from dataclasses import dataclass, field
 
from vault.cli.logger import blank, done, log, section, step, warn
 
MAX_SUB_QUESTIONS = 4


@dataclass
class SubAnswer:
    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    chunks: list      = field(default_factory=list)

@dataclass
class PlanResult:
    original_query: str
    sub_questions: list[str] = field(default_factory=list)
    sub_answers: list[SubAnswer] = field(default_factory=list)
    final_answer: str = ""
    all_sources: list[str] = field(default_factory=list)
    faithfulness: float = 0.0



def run_plan(
    query: str,
    llm_fn,
    retriever_fn,
    evaluator,
    top_k: int = 5,
    verbose: bool = False,
) -> PlanResult:
    section("Multi-step planner")
    step("PLAN", "Decomposing complex query into sub-questions")
    log(f"Query: [bold]{query}[/bold]")
    log("Strategy: LLM decomposes → RAG answers each → LLM synthesizes")
    log(f"Max sub-questions: {MAX_SUB_QUESTIONS}")

    # 1. Decompose
    sub_questions = _decompose(query, llm_fn)
    
    if not sub_questions:
        warn("PLAN", "Could not decompose query — falling back to single retrieval")
        sub_questions = [query]
 
    log(f"Decomposed into {len(sub_questions)} sub-question(s):")

    for i, q in enumerate(sub_questions, 1):
        log(f"  {i}. {q}")

    done("PLAN", f"{len(sub_questions)} sub-questions ready")
    blank()

    # 2. Answer each sub-question
    sub_answers: list[SubAnswer] = []
    all_chunks = []
 
    for i, sub_q in enumerate(sub_questions, 1):
        step("SUB", f"Sub-question {i}/{len(sub_questions)}: {sub_q}")
        log("Running RAG retrieval for this sub-question")
 
        chunks = retriever_fn(sub_q, top_k)

        if not chunks:
            warn("SUB", f"No chunks found for: {sub_q}")
            sub_answers.append(SubAnswer(question=sub_q, answer="No relevant notes found."))
            blank()
            continue
 
        context = "\n\n".join(f"[{c.note_title}]\n{c.text}" for c in chunks)
        sources = list(dict.fromkeys(c.note_title for c in chunks))
        all_chunks.extend(chunks)
 
        prompt = _build_sub_prompt(sub_q, context)
        answer = llm_fn(prompt)
 
        log(f"  Sources: {', '.join(sources)}")
        done("SUB", f"Sub-question {i} answered")
 
        sub_answers.append(SubAnswer(
            question=sub_q,
            answer=answer,
            sources=sources,
            chunks=chunks,
        ))
        blank()

    # 3. Synthesize
    step("SYNTHESIZE", "Synthesizing all sub-answers into final answer")
    log("LLM combines all sub-answers into one coherent response")
    log(f"Total source notes across all sub-questions: {len(set(s for sa in sub_answers for s in sa.sources))}")
 
    final_answer = _synthesize(query, sub_answers, llm_fn)
    done("SYNTHESIZE", "Final answer synthesized")
    blank()

    # 4. Evaluate
    all_context = "\n\n".join(c.text for c in all_chunks)
    eval_result = evaluator.evaluate(final_answer, all_context, verbose=verbose)
    blank()
 
    all_sources = list(dict.fromkeys(
        s for sa in sub_answers for s in sa.sources
    ))
 
    return PlanResult(
        original_query=query,
        sub_questions=sub_questions,
        sub_answers=sub_answers,
        final_answer=final_answer,
        all_sources=all_sources,
        faithfulness=eval_result.score,
    )

def _decompose(query: str, llm_fn) -> list[str]:
    prompt = f"""You are a research assistant helping to answer a complex question about personal notes.
 
Break this question into {MAX_SUB_QUESTIONS} or fewer specific, focused sub-questions that together would fully answer it.
Each sub-question should be answerable from a small set of notes.
 
Rules:
- Output ONLY the sub-questions, one per line
- Number them: 1. 2. 3.
- No explanations, no preamble
- If the question is already simple and focused, output it as-is as question 1
 
Question: {query}
 
Sub-questions:"""
    
    try:
        raw = llm_fn(prompt)
        lines = raw.strip().split("\n")
        questions = []

        for line in lines:
            # Strip numbering like "1." "1)" "- "
            clean = re.sub(r"^[\d]+[.)]\s*|^[-•]\s*", "", line.strip())

            if clean and len(clean) > 5:
                questions.append(clean)

        return questions[:MAX_SUB_QUESTIONS]
    except Exception:
        return [query]
    
def _build_sub_prompt(question: str, context: str) -> str:
    return f"""Answer this specific question using ONLY the provided notes.
Cite [Note Title] for every claim. Be concise.
 
Notes:
{context}
 
Question: {question}
 
Answer:"""

def _synthesize(original_query: str, sub_answers: list[SubAnswer], llm_fn) -> str:
    sub_section = "\n\n".join(
        f"Sub-question: {sa.question}\nAnswer: {sa.answer}"
        for sa in sub_answers
        if sa.answer and sa.answer != "No relevant notes found."
    )
 
    prompt = f"""You are synthesizing research notes into a final answer.
 
Original question: {original_query}
 
Here are answers to specific sub-questions from the notes:
 
{sub_section}
 
Write a single, unified, well-structured answer to the original question.
Rules:
- Use ONLY information from the sub-answers above
- Cite [Note Title] for every claim
- Use headers or bullet points where helpful
- Do not add outside knowledge
 
Final answer:"""
 
    try:
        return llm_fn(prompt)
    except Exception as exc:
        return f"Synthesis failed: {exc}"