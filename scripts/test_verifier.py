from src.grounding.verifier import GroundingVerifier
from src.models import RetrievalResult
from src.grounding.citations import validate_citations

def main():
    evidence = [
        RetrievalResult(
            chunk_id="business/001::chunk_1",
            text=(
                "It lost 464,000 subscribers in the "
                "fourth quarter. Profits were lower "
                "than in the preceding three quarters."
            ),
            reranker_score=2.7891,
            reranker_rank=1,
        )
    ]
    verifier = GroundingVerifier()
    # --------------------------------------------------
    # TEST 1: GROUNDED ANSWER
    # --------------------------------------------------
    grounded_answer = (
        "AOL lost 464,000 subscribers in the "
        "fourth quarter. [Evidence 1]"
    )

    result = verifier.verify(
        query="What happened to AOL subscribers?",
        answer=grounded_answer,
        evidence=evidence,
    )

    print("=" * 70)
    print("TEST 1 — GROUNDED ANSWER")
    print("=" * 70)

    print(result)

    # --------------------------------------------------
    # TEST 2: HALLUCINATED ANSWER
    # --------------------------------------------------
    hallucinated_answer = (
        "AOL lost 464,000 subscribers in the "
        "fourth quarter because broadband competition "
        "caused customers to leave. [Evidence 1]"
    )

    result = verifier.verify(
        query="What happened to AOL subscribers?",
        answer=hallucinated_answer,
        evidence=evidence,
    )

    print("\n" + "=" * 70)
    print("TEST 2 — HALLUCINATED ANSWER")
    print("=" * 70)

    print(result)

    print("\n" + "=" * 70)
    print("CITATION TEST")
    print("=" * 70)

    valid_answer = ("AOL lost 464,000 subscribers. [Evidence 1]")
    invalid_answer = ("AOL lost 464,000 subscribers. [Evidence 99]")
    print(
        "Valid:",
        validate_citations(
            valid_answer,
            evidence,
        ),
    )

    print(
        "Invalid:",
        validate_citations(
            invalid_answer,
            evidence,
        ),
    )

if __name__ == "__main__":
    main()