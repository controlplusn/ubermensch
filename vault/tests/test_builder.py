from dataclasses import dataclass, field
from pathlib import Path

from vault.graph.builder import (
    build_graphs,
    load_graph,
    _cosine_similarity,
)


@dataclass
class MockNote:
    title: str
    path: Path
    content: str
    tags: list[str] = field(default_factory=list)
    wikilinks: list[str] = field(default_factory=list)
    backlinks: list[str] = field(default_factory=list)


def create_mock_notes():
    return [
        MockNote(
            title="AI",
            path=Path("ai.md"),
            content="Machine learning and neural networks.",
            tags=["ai", "ml"],
            wikilinks=["Python"],
        ),
        MockNote(
            title="Python",
            path=Path("python.md"),
            content="Python is used in machine learning.",
            tags=["python", "ai"],
            wikilinks=["AI", "ChromaDB"],
        ),
        MockNote(
            title="ChromaDB",
            path=Path("chromadb.md"),
            content="ChromaDB is a vector database.",
            tags=["database", "ai"],
            wikilinks=[],
        ),
        MockNote(
            title="Philosophy",
            path=Path("philosophy.md"),
            content="Nietzsche discussed the Ubermensch.",
            tags=["philosophy"],
            wikilinks=[],
        ),
    ]


def create_mock_embeddings():
    return {
        # Similar embeddings
        "AI": [1.0, 0.9, 0.8],
        "Python": [0.9, 0.85, 0.75],
        "ChromaDB": [0.88, 0.82, 0.7],

        # Dissimilar embedding
        "Philosophy": [0.1, 0.05, 0.0],
    }


def print_graph_summary(G):
    print("\n" + "=" * 80)
    print("GRAPH SUMMARY")
    print("=" * 80)

    print(f"\nNodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    print("\nNode Details:")
    for node, data in G.nodes(data=True):
        print("-" * 50)
        print(f"Node: {node}")
        print(f"Tags: {data['tags']}")
        print(f"Wikilinks: {data['wikilinks']}")
        print(f"Word Count: {data['word_count']}")

    print("\nEdge Details:")
    for a, b, data in G.edges(data=True):
        print("-" * 50)
        print(f"{a} <-> {b}")
        print(f"Weight: {round(data.get('weight', 0), 3)}")
        print(f"Edge Types: {data.get('edge_types')}")

        if "semantic_score" in data:
            print(f"Semantic: {data['semantic_score']}")


def test_cosine_similarity():
    print("\n" + "=" * 80)
    print("TEST 1 — COSINE SIMILARITY")
    print("=" * 80)

    a = [1, 0, 0]
    b = [1, 0, 0]
    c = [0, 1, 0]

    sim_same = _cosine_similarity(a, b)
    sim_diff = _cosine_similarity(a, c)

    print(f"\nSimilarity(a, b): {sim_same}")
    print(f"Similarity(a, c): {sim_diff}")


def test_build_graph():
    print("\n" + "=" * 80)
    print("TEST 2 — BUILD KNOWLEDGE GRAPH")
    print("=" * 80)

    notes = create_mock_notes()
    embeddings = create_mock_embeddings()

    G = build_graphs(notes, embeddings)

    print_graph_summary(G)


def test_load_graph():
    print("\n" + "=" * 80)
    print("TEST 3 — LOAD SAVED GRAPH")
    print("=" * 80)

    G = load_graph()

    if G is None:
        print("Graph failed to load")
        return

    print("\nLoaded graph successfully")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")


if __name__ == "__main__":
    test_cosine_similarity()

    test_build_graph()

    test_load_graph()