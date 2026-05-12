from __future__ import annotations
 
import re
from dataclasses import dataclass, field
 
from vault.cli.logger import blank, done, log, section, step, warn

DEFAULT_THRESHOLD = 0.35

# Words too common to be useful signal for overlap scoring
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "used",
    "in", "of", "to", "for", "on", "with", "at", "by", "from", "up",
    "about", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again", "then",
    "once", "and", "but", "or", "nor", "so", "yet", "both", "either",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "i", "you", "he", "she", "we", "my", "your", "his", "her", "our",
    "not", "no", "if", "as", "than", "because", "while", "although",
    "also", "just", "more", "there", "here", "when", "where", "which",
    "who", "what", "how", "all", "each", "any", "some", "other", "only",
}

# Sentences that are meta-responses, not real claims — skip from scoring
META_PATTERNS = [
    re.compile(r"i couldn.t find", re.IGNORECASE),
    re.compile(r"the closest relevant", re.IGNORECASE),
    re.compile(r"your notes (do|don.t|doesn.t) (contain|have|include)", re.IGNORECASE),
    re.compile(r"there (is|are) no (information|mention|note)", re.IGNORECASE),
    re.compile(r"based on (the|your) (provided |available )?notes", re.IGNORECASE),
    re.compile(r"^(note|disclaimer|caveat):", re.IGNORECASE),
]


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
CITATION_RE = re.compile(r"\[[^\]]+\]")  # strip [Note Title] citations before scoring


# Data models
@dataclass
class ClaimResult:
    text:      str
    supported: bool
    overlap:   float
    key_words: list[str]
    matched:   list[str]


@dataclass
class EvalResult:
    score: float
    claims: list[ClaimResult] = field(default_factory=list)
    supported_count: int = 0
    total_count: int = 0
    needs_retrieval: bool = False   # True if score is below re-retrieval threshold
    skipped: int = 0

    @property
    def confidence_label(self) -> str:
        if self.score >= 0.7:
            return "High confidence"
        elif self.score >= 0.4:
            return "Moderate confidence"
        else:
            return "Low confidence — verify against your notes"
        
    @property
    def confidence_color(self) -> str:
        if self.score >= 0.7:
            return "green"
        elif self.score >= 0.4:
            return "yellow"
        else:
            return "red"


# Evaluator
class FaithfulnessEvaluator:
    """
    threshold: keyword overlap ratio for a claim to be "supported"
    re_retrieval_threshold : EvalResult.needs_retrieval = True when score below this
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        re_retrieval_threshold: float = 0.4,
    ):
        self.threshold = threshold
        self.re_retrieval_threshold = re_retrieval_threshold

    def evaluate(
        self,
        answer: str,
        context: str,
        verbose: bool = False,
    ) -> EvalResult:
        step("EVAL", "Scoring answer faithfulness against retrieved notes")
        log(f"Method: RAGAS-style keyword overlap  |  threshold: {self.threshold:.0%}")
        log("Each sentence in the answer is checked against retrieved note content")

        raw_claims = self._extract_sentences(answer)
        scoreable, skipped = self._filter_meta(raw_claims)

        if skipped:
            log(f"Skipped {skipped} meta-sentence(s) (e.g. 'I couldn't find...')")
 
        if not scoreable:
            log("No scoreable claims — answer may be empty or entirely meta")
            done("EVAL", "Faithfulness: N/A (no scoreable claims)")
            return EvalResult(score=1.0, skipped=skipped)
 
        log(f"Scoring {len(scoreable)} claim(s)")
 
        claim_results: list[ClaimResult] = []
        for sentence in scoreable:
            cr = self._score_claim(sentence, context)
            claim_results.append(cr)
 
            status = "[green]✓[/green]" if cr.supported else "[red]✗[/red]"
            overlap_str = f"{cr.overlap:.0%} overlap"
 
            if verbose:
                # Detailed per-claim output for --eval flag
                log(f"  {status}  [{overlap_str}]  \"{cr.text[:70]}{'...' if len(cr.text) > 70 else ''}\"")
                if cr.matched:
                    log(f"     matched words: {', '.join(cr.matched[:8])}")
                if not cr.supported and cr.key_words:
                    unmatched = [w for w in cr.key_words if w not in cr.matched]
                    if unmatched:
                        log(f"     [dim]unmatched: {', '.join(unmatched[:6])}[/dim]")
            else:
                # Compact output for normal ask
                log(f"  {status}  {cr.text[:60]}{'...' if len(cr.text) > 60 else ''}")
 
        supported = sum(1 for cr in claim_results if cr.supported)
        total = len(claim_results)
        score = round(supported / total, 3) if total else 1.0
        needs_retrieval = score < self.re_retrieval_threshold
 
        color = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
        done(
            "EVAL",
            f"Faithfulness: [{color}]{score:.1%}[/{color}]"
            f"  ({supported}/{total} claims supported)"
            + ("  [yellow]→ will re-retrieve[/yellow]" if needs_retrieval else ""),
        )
 
        if score < 0.4:
            warn("EVAL", "Low faithfulness — answer may contain hallucinated content")
            log("The agent will automatically re-retrieve with an expanded query")
 
        return EvalResult(
            score=score,
            claims=claim_results,
            supported_count=supported,
            total_count=total,
            needs_retrieval=needs_retrieval,
            skipped=skipped,
        )
 
    def score(self, answer: str, context: str) -> float:
        return self.evaluate(answer, context, verbose=False).score
 
    def _extract_sentences(self, text: str) -> list[str]:
        clean = CITATION_RE.sub("", text)
        sentences = SENTENCE_RE.split(clean.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 12]
    
    def _filter_meta(self, sentences: list[str]) -> tuple[list[str], int]:
        scoreable, skipped = [], 0
        for s in sentences:
            if any(p.search(s) for p in META_PATTERNS):
                skipped += 1
            else:
                scoreable.append(s)
        return scoreable, skipped
    
    def _score_claim(self, claim: str, context: str) -> ClaimResult:
        key_words = [
            w.lower().strip(".,;:!?\"'()-")
            for w in claim.split()
            if w.lower().strip(".,;:!?\"'()-") not in STOPWORDS
            and len(w.strip(".,;:!?\"'()-")) > 2
        ]
 
        if not key_words:
            # All stopwords — treat as neutral/supported
            return ClaimResult(
                text=claim, supported=True, overlap=1.0,
                key_words=[], matched=[],
            )
 
        context_lower = context.lower()
        matched = [w for w in key_words if w in context_lower]
        overlap = len(matched) / len(key_words)
 
        return ClaimResult(
            text=claim,
            supported=overlap >= self.threshold,
            overlap=overlap,
            key_words=key_words,
            matched=matched,
        )