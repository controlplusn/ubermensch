from vault.retrieval.embedder import embed_chunks, embed_query


def test_embed_chunks():
    texts = [
        "Machine learning is a branch of AI.",
        "Embeddings convert text into vectors.",
        "Python is commonly used in AI systems.",
    ]

    vectors = embed_chunks(texts)

    print(f"Total vectors: {len(vectors)}")
    print(f"Vector dimension: {len(vectors[0])}")

    for i, vector in enumerate(vectors):
        print(f"\nVector {i}:")
        print(vector[:10]) 


def test_embed_query():
    query = "What is artificial intelligence?"

    vector = embed_query(query)

    print(f"Query vector dimension: {len(vector)}")
    print(vector[:10])


if __name__ == "__main__":
    test_embed_chunks()
    print("\n" + "=" * 50 + "\n")
    test_embed_query()