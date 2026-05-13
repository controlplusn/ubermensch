from __future__ import annotations

import pickle
import networkx as nx
from dataclasses import dataclass, field
from pathlib import Path

from vault.cli.logger import blank, done, log, section, step, warn


GRAPH_PATH = Path.home() / ".vault" / "graph" / "vault.gpickle"
SEMANTIC_THRESHOLD = 0.65
TAG_WEIGHT = 0.3
WIKILINK_WEIGHT = 1.0


@dataclass
class GraphStats:
    total_nodes: int
    total_edges: int
    wikilink_edges: int
    tag_edges: int = 0
    semantic_edges: int = 0
    orphan_nodes: int = 0
    avg_degree: float = 0.0
    top_hubs: list = field(default_factory=list)



def build_graphs(notes, embeddings=None):
    try:
        import networkx as nx
    except ImportError:
        from vault.cli.logger import fail
        fail("GRAPH", "networkx not installed. Run: pip install networkx")
        raise SystemExit(1)

    section("Graph build")
    step("GRAPH", f"Building knowledge graph from {len(notes)} notes")
    log("Graph model: nodes = notes,  edges = relationships between notes")
    log("Edge types:  [wikilink]  [tag co-occurrence]  [semantic similarity]")
 
    G = nx.Graph()

    # Nodes
    step("NODES", "Adding one node per note")
    log("Each node stores: title, tags, path, wikilinks, backlinks, word_count")

    for note in notes:
        G.add_node(
            note.title,
            tags=note.tags,
            path=str(note.path),
            wikilinks=note.wikilinks,
            backlinks=note.backlinks,
            word_count=len(note.content.split()),
        )
        log(f"  node: {note.title}  [dim](tags={len(note.tags)} links={len(note.wikilinks)})[/dim]")
    done("NODES", f"{G.number_of_nodes()} nodes added")
    blank()

    # Wikilink edges
    step("WIKILINKS", "Adding edges from explicit [[wikilinks]]")
    log(f"Weight per wikilink edge: {WIKILINK_WEIGHT}")
    log("Strongest edges — you deliberately created these connections")

    wikilink_count = 0
    title_set = set(G.nodes())

    for note in notes:
        for link in note.wikilinks:
            if link in title_set and link != note.title:
                if G.has_edge(note.title, link):
                    G[note.title][link]["weight"] += WIKILINK_WEIGHT
                else:
                    G.add_edge(note.title, link, weight=WIKILINK_WEIGHT, edge_types=["wikilink"])
                wikilink_count += 1
                log(f"  {note.title}  ->  {link}  (wikilink)")
    done("WIKILINKS", f"{wikilink_count} wikilink edge(s) added")
    blank()

    # Tag edges
    step("TAGS", "Adding edges from shared #tags")
    log(f"Weight per shared tag: {TAG_WEIGHT}  (accumulates: 3 shared tags = weight 0.9)")
    log("Notes sharing tags talk about similar topics even without explicit links")

    tag_count = 0
    note_list = list(notes)
    for i, note_a in enumerate(note_list):
        for note_b in note_list[i + 1:]:
            shared = set(note_a.tags) & set(note_b.tags)
            if not shared:
                continue
            weight = len(shared) * TAG_WEIGHT
            if G.has_edge(note_a.title, note_b.title):
                G[note_a.title][note_b.title]["weight"] += weight
                if "tag" not in G[note_a.title][note_b.title]["edge_types"]:
                    G[note_a.title][note_b.title]["edge_types"].append("tag")
            else:
                G.add_edge(note_a.title, note_b.title, weight=weight, edge_types=["tag"])
            tag_count += 1
            log(f"  {note_a.title}  <->  {note_b.title}  [dim](shared: {', '.join(f'#{t}' for t in shared)})[/dim]")
    done("TAGS", f"{tag_count} tag edge(s) added")
    blank()

    # Semantic edges
    semantic_count = 0
    
    if embeddings:
        step("SEMANTIC", "Computing semantic similarity between all note pairs")
        log(f"Threshold: {SEMANTIC_THRESHOLD} — only pairs with cosine similarity >= this get an edge")
        log("These are UNDISCOVERED connections — notes that discuss similar things")
        log("but you have not explicitly linked yet. Powers `vault graph suggest`.")

        titles = list(embeddings.keys())

        for i, title_a in enumerate(titles):
            for title_b in titles[i + 1:]:
                if not (G.has_node(title_a) and G.has_node(title_b)):
                    continue
                sim = _cosine_similarity(embeddings[title_a], embeddings[title_b])

                if sim < SEMANTIC_THRESHOLD:
                    continue
                if G.has_edge(title_a, title_b):
                    G[title_a][title_b]["semantic_score"] = sim
                    if "semantic" not in G[title_a][title_b]["edge_types"]:
                        G[title_a][title_b]["edge_types"].append("semantic")
                else:
                    G.add_edge(title_a, title_b, weight=sim, edge_types=["semantic"], semantic_score=sim)

                semantic_count += 1
                
                log(f"  {title_a}  ~  {title_b}  [dim](similarity: {sim:.3f})[/dim]")
                
        done("SEMANTIC", f"{semantic_count} semantic edge(s) added (sim >= {SEMANTIC_THRESHOLD})")
        blank()

    else:
        warn("SEMANTIC", "No embeddings provided — skipping semantic edges")
        log("Run vault init first, then vault graph build to include semantic edges")
        blank()

    # Stats
    step("STATS", "Computing graph statistics")

    degrees = dict(G.degree())
    orphans = sum(1 for d in degrees.values() if d == 0)
    avg_deg = sum(degrees.values()) / max(len(degrees), 1)
    top_hubs = sorted(degrees.items(), key=lambda x: -x[1])[:5]

    log(f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}")
    log(f"  Wikilink: {wikilink_count}  |  Tag: {tag_count}  |  Semantic: {semantic_count}")
    log(f"Orphans: {orphans}  |  Avg degree: {avg_deg:.2f}")
    log("Top hub notes (most connections):")

    for title, deg in top_hubs:
        log(f"  [{deg:>3} connections]  {title}")

    done("STATS", "Graph statistics computed")
    blank()
 
    # Save
    step("SAVE", f"Saving graph to {GRAPH_PATH}")
    log("Saved as .gpickle — loads instantly on next run")
    
    _save_graph(G)
    
    done("SAVE", f"Graph saved  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    
    return G

def load_graph():
    if not GRAPH_PATH.exists():
        return None
    try:
        with open(GRAPH_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None
 
 
def _save_graph(G) -> None:
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
 
 
def _cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5

    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    return round(dot / (mag_a * mag_b), 4)