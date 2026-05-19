from __future__ import annotations
 
import re
from dataclasses import dataclass, field
 
from vault.cli.logger import blank, done, log, section, step, warn
from vault.eval.nli import is_nli_available, batch_score_claims

DEFAULT_THRESHOLD = 0.35    # Keyword overlap threshold -> Fallback
NLI_THRESHOLD = 0.5     # Entailment probability threshold 
RE_RETRIEVAL_THRESHOLD = 0.4

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
    re.compile(r"i could(n.t| not) find", re.IGNORECASE),
    re.compile(r"the closest (relevant |)notes", re.IGNORECASE),
    re.compile(r"your notes (do|don.t|doesn.t) (contain|have|include)", re.IGNORECASE),
    re.compile(r"there (is|are) no (information|mention|note)", re.IGNORECASE),
    re.compile(r"based on (the|your) (provided |available )?notes", re.IGNORECASE),
    re.compile(r"^(note|disclaimer|caveat):", re.IGNORECASE),
    re.compile(r"closest notes are", re.IGNORECASE),
]


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
CITATION_RE = re.compile(r"\[[^\]]+\]")  # strip [Note Title] citations before scoring


# Data models
@dataclass
class ClaimResult:
    text: str
    supported: bool
    overlap: float
    key_words: list[str]
    matched: list[str]
    nli_label: str  = ""    # "entailment" / "contradiction" / "neutral"
    nli_entailment: float = 0.0     # NLI entailment probability
    nli_contradiction: float = 0.0  # NLI contradiction probability
    scorer_used: str  = "keyword"  # "nli" or "keyword"

@dataclass
class EvalResult:
    score: float
    claims: list[ClaimResult] = field(default_factory=list)
    supported_count: int = 0
    total_count: int = 0
    needs_retrieval: bool = False   # True if score is below re-retrieval threshold
    skipped: int = 0
    scorer_used: str = "keyword" # -> default

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
        nli_threshold: float = NLI_THRESHOLD,
        re_retrieval_threshold: float = RE_RETRIEVAL_THRESHOLD,
        force_keyword: bool = False,
    ):
        self.threshold = threshold
        self.nli_threshold = nli_threshold
        self.re_retrieval_threshold = re_retrieval_threshold
        self.force_keyword = force_keyword
        self._nli_available: bool | None = None 

    def _use_nli(self) -> bool:
        if self.force_keyword:
            return False
        if self._nli_available is None:
            self._nli_available = is_nli_available()
        return self._nli_available

    def evaluate(
        self,
        answer: str,
        context: str,
        verbose: bool = False,
    ) -> EvalResult:
        step("EVAL", "Scoring answer faithfulness against retrieved notes")
        log(f"Method: RAGAS-style keyword overlap  |  threshold: {self.threshold:.0%}")
        log("Each sentence in the answer is checked against retrieved note content")

        use_nli      = self._use_nli()
        scorer_label = "NLI cross-encoder" if use_nli else "keyword overlap (heuristic)"
        log(f"Scorer: [bold]{scorer_label}[/bold]")
 
        if use_nli:
            log("Model: cross-encoder/nli-deberta-v3-small")
            log("Labels: entailment ✓  |  contradiction ✗  |  neutral ~")
        else:
            log(f"Threshold: {self.threshold:.0%} keyword overlap")
            log("[dim]Tip: install sentence-transformers for NLI scoring (more accurate)[/dim]")

        raw_claims = self._extract_sentences(answer)
        scoreable, skipped = self._filter_meta(raw_claims)

        if skipped:
            log(f"Skipped {skipped} meta-sentence(s) (e.g. 'I couldn't find...')")
 
        if not scoreable:
            log("No scoreable claims found")
            done("EVAL", "Faithfulness: N/A")
            return EvalResult(score=1.0, skipped=skipped, scorer_used=scorer_label)
        
        log(f"Scoring {len(scoreable)} claim(s)")

        if use_nli:
            claim_results = self._score_with_nli(scoreable, context, verbose)
        else:
            claim_results = self._score_with_keywords(scoreable, context, verbose)
 
        supported = sum(1 for cr in claim_results if cr.supported)
        total = len(claim_results)
        score = round(supported / total, 3) if total else 1.0
        needs_ret = score < self.re_retrieval_threshold
 
        color = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"

        done(
            "EVAL",
            f"Faithfulness: [{color}]{score:.1%}[/{color}]"
            f"  ({supported}/{total} claims supported)"
            f"  [dim]via {scorer_label}[/dim]"
            + ("  [yellow]→ will re-retrieve[/yellow]" if needs_ret else ""),
        )
 
        if score < 0.4:
            warn("EVAL", "Low faithfulness — answer may contain hallucinated content")
 
        return EvalResult(
            score=score,
            claims=claim_results,
            supported_count=supported,
            total_count=total,
            needs_retrieval=needs_ret,
            skipped=skipped,
            scorer_used=scorer_label,
        )
 
    def score(self, answer: str, context: str) -> float:
        return self.evaluate(answer, context, verbose=False).score
    


    # NLI Scoring
    def _score_with_nli(
        self, claims: list[str], context: str, verbose: bool
    ) -> list[ClaimResult]:
        nli_results = batch_score_claims(
            claims=claims,
            context=context,
            entailment_threshold=self.nli_threshold,
            verbose=verbose,
        )

        claim_results = []
        for claim, nli in zip(claims, nli_results):
            kw_result = self._score_claim_keywords(claim, context)
 
            cr = ClaimResult(
                text=claim,
                supported=nli.supported,
                overlap=kw_result.overlap,
                key_words=kw_result.key_words,
                matched=kw_result.matched,
                nli_label=nli.label,
                nli_entailment=nli.entailment_score,
                nli_contradiction=nli.contradiction_score,
                scorer_used="nli",
            )
            claim_results.append(cr)
 
            if not verbose:
                status = "[green]✓[/green]" if nli.supported else "[red]✗[/red]"
                log(f"  {status}  [{nli.label:<15}]  {claim[:60]}{'...' if len(claim) > 60 else ''}")
 
        return claim_results
    


    # Keyword scoring
    def _score_with_keywords(
        self, claims: list[str], context: str, verbose: bool
    ) -> list[ClaimResult]:
        claim_results = []

        for claim in claims:
            cr = self._score_claim_keywords(claim, context)
            claim_results.append(cr)
 
            status  = "[green]✓[/green]" if cr.supported else "[red]✗[/red]"
            overlap = f"{cr.overlap:.0%} overlap"
 
            if verbose:
                log(f"  {status}  [{overlap}]  \"{cr.text[:70]}{'...' if len(cr.text) > 70 else ''}\"")
                if cr.matched:
                    log(f"     matched: {', '.join(cr.matched[:8])}")
            else:
                log(f"  {status}  {cr.text[:60]}{'...' if len(cr.text) > 60 else ''}")
 
        return claim_results
    
    def _score_claim_keywords(self, claim: str, context: str) -> ClaimResult:
        key_words = [
            w.lower().strip(".,;:!?\"'()-")
            for w in claim.split()
            if w.lower().strip(".,;:!?\"'()-") not in STOPWORDS
            and len(w.strip(".,;:!?\"'()-")) > 2
        ]

        if not key_words:
            return ClaimResult(
                text=claim, supported=True, overlap=1.0,
                key_words=[], matched=[], scorer_used="keyword",
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
            scorer_used="keyword",
        )

 
    # Sentence utils
    def _extract_sentences(self, text: str) -> list[str]:
        clean = CITATION_RE.sub("", text)
        return [s.strip() for s in SENTENCE_RE.split(clean.strip()) if len(s.strip()) > 12]
 
    def _filter_meta(self, sentences: list[str]) -> tuple[list[str], int]:
        scoreable, skipped = [], 0
        for s in sentences:
            if any(p.search(s) for p in META_PATTERNS):
                skipped += 1
            else:
                scoreable.append(s)
        return scoreable, skipped