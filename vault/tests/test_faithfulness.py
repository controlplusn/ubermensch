from vault.eval.faithfulness import FaithfulnessEvaluator


def test_faithfulness():
    evaluator = FaithfulnessEvaluator()

    context = """
    Python is widely used in AI and machine learning systems.
    ChromaDB is a vector database used for semantic retrieval.
    Machine learning is a subset of artificial intelligence.
    """

    print("\n" + "=" * 80)
    print("TEST 1 — FULLY SUPPORTED")
    print("=" * 80)

    answer_1 = """
    Python is widely used in AI systems. [Python Notes]
    Machine learning is a subset of artificial intelligence. [AI Notes]
    """

    result_1 = evaluator.evaluate(
        answer=answer_1,
        context=context,
        verbose=True,
    )

    print("\nFinal Score:", result_1.score)
    print("Confidence :", result_1.confidence_label)
    print("Needs Retrieval:", result_1.needs_retrieval)

    print("\n" + "=" * 80)
    print("TEST 2 — PARTIALLY HALLUCINATED")
    print("=" * 80)

    answer_2 = """
    Python is widely used in AI systems.
    ChromaDB stores embeddings for retrieval.
    OpenAI invented ChromaDB in 2021.
    """

    result_2 = evaluator.evaluate(
        answer=answer_2,
        context=context,
        verbose=True,
    )

    print("\nFinal Score:", result_2.score)
    print("Confidence :", result_2.confidence_label)
    print("Needs Retrieval:", result_2.needs_retrieval)

    print("\n" + "=" * 80)
    print("TEST 3 — META RESPONSE")
    print("=" * 80)

    answer_3 = """
    I couldn't find a clear answer in your notes.
    The closest relevant notes are AI Notes and Python Notes.
    """

    result_3 = evaluator.evaluate(
        answer=answer_3,
        context=context,
        verbose=True,
    )

    print("\nFinal Score:", result_3.score)
    print("Skipped:", result_3.skipped)
    print("Confidence :", result_3.confidence_label)

    print("\n" + "=" * 80)
    print("TEST 4 — LOW FAITHFULNESS")
    print("=" * 80)

    answer_4 = """
    Java was invented specifically for neural networks.
    NASA created ChromaDB for spacecraft navigation.
    AI systems only run on quantum computers.
    """

    result_4 = evaluator.evaluate(
        answer=answer_4,
        context=context,
        verbose=True,
    )

    print("\nFinal Score:", result_4.score)
    print("Confidence :", result_4.confidence_label)
    print("Needs Retrieval:", result_4.needs_retrieval)

    print("\n" + "=" * 80)
    print("TEST 5 — DIRECT score() METHOD")
    print("=" * 80)

    score = evaluator.score(
        answer="Python is widely used in AI.",
        context=context,
    )

    print("Score:", score)


if __name__ == "__main__":
    test_faithfulness()