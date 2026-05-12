from __future__ import annotations
 
import re
 
from vault.cli.logger import done, log, step, warn

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
}

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")



class FaithfulnessEvaluator:
    """
    Usage:
        - evaluator = FaithfulnessEvaluator()
        - score = evaluator.score(answer_text, context_text)
        - # score ∈ [0.0, 1.0]
    """

    def __init__(self, overlap_threshold: float = 0.35):
        self.threshold = overlap_threshold

    def score(self, answer: str, context: str) -> float:
        step("EVAL", "Scoring answer faithfulness against retrieved notes")
        log("Method: RAGAS-style claim-level keyword overlap (MVP heuristic)")
        log(f"A claim is 'supported' if ≥{self.threshold:.0%} of its key words appear in context")

        claims = self._extract_claims(answer)
        if not claims:
            log("No scoreable claims found in answer")
            done("EVAL", "Faithfulness: N/A")
            return 1.0
 
        log(f"Extracted {len(claims)} claim(s) from answer")
 
        supported = 0
        for i, claim in enumerate(claims, 1):
            is_supported = self._claim_supported(claim, context)
            status = "[green]✓ supported[/green]" if is_supported else "[red]✗ unsupported[/red]"
            log(f"  Claim {i}: {status}  — \"{claim[:60]}{'...' if len(claim) > 60 else ''}\"")
            if is_supported:
                supported += 1
 
        score = round(supported / len(claims), 3)
        color = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
        done("EVAL", f"Faithfulness: [{color}]{score:.1%}[/{color}]  ({supported}/{len(claims)} claims supported)")
 
        if score < 0.4:
            warn("EVAL", "Low faithfulness — answer may contain hallucinated content")
            log("Consider rephrasing your question or checking if relevant notes are indexed")
 
        return score
    
    def _extract_claims(self, text: str) -> list[str]:
        # Strip citation patterns like [Note Title] before scoring
        clean = re.sub(r"\[[^\]]+\]", "", text)
        sentences = SENTENCE_RE.split(clean.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 15]
    
    def _claim_supported(self, claim: str, context: str) -> bool:
        claim_words = {
            w.lower().strip(".,;:!?\"'")
            for w in claim.split()
            if w.lower() not in STOPWORDS and len(w) > 2
        }
        if not claim_words:
            return True   # claim is all stopwords — treat as neutral
 
        context_lower = context.lower()
        matching = sum(1 for w in claim_words if w in context_lower)
        overlap = matching / len(claim_words)
        return overlap >= self.threshold