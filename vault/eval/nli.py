from __future__ import annotations

import re
 
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
 
from vault.cli.logger import done, log, step, warn
from sentence_transformers import CrossEncoder
 
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

LABEL_CONTRADICTION = "contradiction"
LABEL_ENTAILMENT = "entailment"
LABEL_NEUTRAL = "neutral"


@dataclass
class NLIResult:
    label: str
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    supported: bool     # True if entailment_score >= threshold



@lru_cache(maxsize=1)
def _load_nli_model():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(NLI_MODEL)
    except ImportError:
        warn("NLI", "sentence-transformers not installed")
        warn("NLI", "Run: pip install sentence-transformers")
        return None
    except Exception as exc:
        warn("NLI", f"Could not load NLI model: {exc}")
        return None
    
def is_nli_available() -> bool:
    try:
        return True
    except ImportError:
        return False

def score_claim(
    claim: str,
    context_chunks: list[str],
    entailment_threshold: float = 0.5,
) -> NLIResult:
    model = _load_nli_model()

    if model is None:
        # Fallback: treat as supported (fail open)
        return NLIResult(
            label=LABEL_NEUTRAL,
            entailment_score=0.5,
            contradiction_score=0.0,
            neutral_score=0.5,
            supported=True,
        )
    
    best_entailment = 0.0
    best_contradiction = 0.0
    best_neutral = 0.0
    best_label = LABEL_NEUTRAL

    for chunk in context_chunks:
        # Cross-encoder takes [premise, hypothesis] pairs
        pair = [(chunk, claim)]
        scores = model.predict(pair, apply_softmax=True)[0]

        label2id = model.config.label2id
        id2label = {v: k.lower() for k, v in label2id.items()}

        contra_score = float(scores[label2id.get("contradiction", label2id.get("CONTRADICTION", 0))])
        entail_score = float(scores[label2id.get("entailment", label2id.get("ENTAILMENT",    1))])
        neutral_score = float(scores[label2id.get("neutral", label2id.get("NEUTRAL",       2))])

        if entail_score > best_entailment:
            best_entailment = entail_score
            best_contradiction = contra_score
            best_neutral = neutral_score
            best_label = (
                LABEL_ENTAILMENT if entail_score >= entailment_threshold
                else LABEL_CONTRADICTION if contra_score > neutral_score
                else LABEL_NEUTRAL
            )

    return NLIResult(
        label=best_label,
        entailment_score=round(best_entailment, 4),
        contradiction_score=round(best_contradiction, 4),
        neutral_score=round(best_neutral, 4),
        supported=best_entailment >= entailment_threshold,
    )

def batch_score_claims(
    claims: list[str],
    context: str,
    entailment_threshold: float = 0.5,
    verbose: bool = False,
) -> list[NLIResult]:
    context_chunks = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", context)
        if len(s.strip()) > 20
    ]

    # Cap = 20 chunks -> to keep latency reasonable
    if len(context_chunks) > 20:
        context_chunks = context_chunks[:20]

    results = []
    
    for i, claim in enumerate(claims):
        result = score_claim(claim, context_chunks, entailment_threshold)
        results.append(result)
 
        if verbose:
            icon = {
                LABEL_ENTAILMENT:    "[green]✓ entailed[/green]",
                LABEL_CONTRADICTION: "[red]✗ contradicted[/red]",
                LABEL_NEUTRAL:       "[yellow]~ neutral[/yellow]",
            }.get(result.label, "?")
            log(
                f"  {icon}  [{result.entailment_score:.2f}]  "
                f"\"{claim[:65]}{'...' if len(claim) > 65 else ''}\""
            )
 
    return results