from __future__ import annotations
 
from dataclasses import dataclass, field
 
from vault.cli.logger import blank, done, log, section, step, warn
 
SUGGESTION_THRESHOLD = 0.65 


@dataclass
class LinkSuggestion:
    source: str
    target: str
    score: float
    semantic_score: float
    shared_tags: list[str] = field(default_factory=list)
    reason: str = ""



def find_suggestions(G, top_n: int = 20) -> list[LinkSuggestion]:
    section("Backlink suggestions")
    step("SUGGEST", "Scanning knowledge graph for unlinked similar notes")
    log("Looking for: high semantic similarity + no existing [[wikilink]]")
    log(f"Minimum similarity threshold: {SUGGESTION_THRESHOLD}")
    log("These are connections you talk about in similar terms but haven't linked yet")
 
    if G is None:
        warn("SUGGEST", "No graph found. Run `vault graph build` first.")
        return []
 
    suggestions: list[LinkSuggestion] = []
    checked = 0
    skipped_linked = 0
 
    for u, v, data in G.edges(data=True):
        checked += 1
        edge_types = data.get("edge_types", [])
        semantic_score = data.get("semantic_score", 0.0)
 
        # Only consider edges that have a semantic component
        if "semantic" not in edge_types or semantic_score < SUGGESTION_THRESHOLD:
            continue
 
        # Skip if they already have an explicit wikilink
        if "wikilink" in edge_types:
            skipped_linked += 1
            log(f"  [dim]skip (already linked): {u}  <->  {v}[/dim]")
            continue
 
        # Compute shared tags for bonus score + context
        u_tags = set(G.nodes[u].get("tags", []))
        v_tags = set(G.nodes[v].get("tags", []))
        shared_tags = list(u_tags & v_tags)
        tag_bonus = min(len(shared_tags) * 0.1, 0.3)   # cap tag bonus at 0.3
 
        combined_score = round(semantic_score * 0.7 + tag_bonus, 4)
 
        reason = _build_reason(semantic_score, shared_tags)
 
        suggestions.append(LinkSuggestion(
            source=u,
            target=v,
            score=combined_score,
            semantic_score=semantic_score,
            shared_tags=shared_tags,
            reason=reason,
        ))
        log(
            f"  [green]candidate[/green]  {u}  <->  {v}"
            f"  [dim](sim={semantic_score:.3f}  score={combined_score:.3f})[/dim]"
        )
 
    log(f"Checked {checked} edges  |  {skipped_linked} already linked  |  {len(suggestions)} candidates")
    suggestions.sort(key=lambda s: -s.score)
    result = suggestions[:top_n]
 
    done("SUGGEST", f"Found {len(result)} backlink suggestion(s)")

    return result


def _build_reason(semantic_score: float, shared_tags: list[str]) -> str:
    parts = [f"semantic similarity {semantic_score:.0%}"]

    if shared_tags:
        parts.append(f"shared tags: {', '.join(f'#{t}' for t in shared_tags[:3])}")

    return "  ·  ".join(parts)