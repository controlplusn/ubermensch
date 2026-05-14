from __future__ import annotations
 
from collections import deque
from dataclasses import dataclass, field
 
from vault.cli.logger import blank, done, log, section, step, warn
 
 
@dataclass
class MapNode:
    title: str
    depth: int
    edge_weight: float
    edge_types: list[str] = field(default_factory=list)
    path: str = ""
    tags: list[str] = field(default_factory=list)



def map_topic(G, topic: str, max_depth: int = 2) -> list[MapNode]:
    section("Idea map")
    step("MAP", f"Building idea cluster around: [bold]{topic}[/bold]")
    log(f"Algorithm: BFS  |  Max depth: {max_depth} hops from seed")
    log("depth 0 = seed note  |  depth 1 = direct connections  |  depth 2 = extended network")
 
    if G is None:
        warn("MAP", "No graph found. Run `vault graph build` first.")
        return []
 
    # Resolve seed node
    seed = _resolve_seed(G, topic)
    if seed is None:
        warn("MAP", f"Could not find a note matching '{topic}'")
        log("Try: `vault graph map \"exact note title\"` or a shorter keyword")
        return []
 
    log(f"Seed resolved to: [bold]{seed}[/bold]")
    blank()

    # BFS
    step("BFS", f"Exploring graph from '{seed}' up to depth {max_depth}")
    log("Each level reveals notes further from the seed topic")
 
    visited: dict[str, int] = {}   # title -> depth first seen
    queue: deque[tuple[str, int, float, list]] = deque()
    queue.append((seed, 0, 1.0, []))
    result_nodes: list[MapNode] = []
 
    while queue:
        current, depth, weight, edge_types = queue.popleft()
 
        if current in visited:
            continue
        visited[current] = depth
 
        node_data = G.nodes.get(current, {})
        result_nodes.append(MapNode(
            title=current,
            depth=depth,
            edge_weight=round(weight, 3),
            edge_types=edge_types,
            path=node_data.get("path", ""),
            tags=node_data.get("tags", []),
        ))
 
        depth_label = "seed" if depth == 0 else f"depth {depth}"
        log(
            f"  [{depth_label}]  {current}"
            + (f"  [dim](via {'+'.join(edge_types)}  w={weight:.2f})[/dim]" if depth > 0 else "")
        )
 
        if depth < max_depth:
            neighbors = sorted(
                G[current].items(),
                key=lambda x: -x[1].get("weight", 0),
            )
            for neighbor, edge_data in neighbors:
                if neighbor not in visited:
                    queue.append((
                        neighbor,
                        depth + 1,
                        edge_data.get("weight", 0.5),
                        edge_data.get("edge_types", []),
                    ))

    # Sort: depth first, then edge weight descending within each depth
    result_nodes.sort(key=lambda n: (n.depth, -n.edge_weight))
 
    done(
        "BFS",
        f"Found {len(result_nodes)} connected note(s) within {max_depth} hop(s) of '{seed}'"
    )
    return result_nodes

def _resolve_seed(G, topic: str) -> str | None:
    topic_lower = topic.lower().strip()
    nodes = list(G.nodes())
 
    # Exact match
    for node in nodes:
        if node.lower() == topic_lower:
            return node
 
    # Partial match — prefer shortest title (most specific)
    matches = [n for n in nodes if topic_lower in n.lower()]
    if matches:
        return min(matches, key=len)
 
    return None