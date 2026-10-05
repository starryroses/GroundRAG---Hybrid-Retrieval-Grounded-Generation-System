from src.grounding.decision import (decide_grounding, REFUSAL_MESSAGE)
from src.models import RetrievalResult

def run_test(query: str,evidence: list[RetrievalResult]):
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    decision = decide_grounding(evidence)

    print(f"Allowed: {decision.allowed}")
    print(f"Reason: {decision.reason}")

    if decision.allowed:
        print("\nEvidence:")
        for result in decision.evidence:
            print(
                f"- {result.chunk_id} "
                f"(reranker={result.reranker_score:.4f})"
            )
    else:
        print(f"\nResponse: {REFUSAL_MESSAGE}")

    print()


def main():

    relevant_evidence = [
        RetrievalResult(
            chunk_id="business/001::chunk_1",
            text=(
                "It lost 464,000 subscribers in "
                "the fourth quarter."
            ),
            reranker_score=2.7891,
            reranker_rank=1,
            metadata={
                "document_name": "001.txt",
                "category": "business",
            },
        ),
        RetrievalResult(
            chunk_id="tech/132::chunk_0",
            text=(
                "A brother and sister in the US were "
                "convicted of sending unsolicited "
                "email messages to AOL subscribers."
            ),
            reranker_score=2.8246,
            reranker_rank=2,
            metadata={
                "document_name": "132.txt",
                "category": "tech",
            },
        ),
    ]

    unrelated_evidence = [
        RetrievalResult(
            chunk_id="tech/350::chunk_5",
            text=(
                "AOL announced plans to launch "
                "a net-based phone service."
            ),
            reranker_score=-0.9820,
            reranker_rank=1,
            metadata={
                "document_name": "350.txt",
                "category": "tech",
            },
        ),
        RetrievalResult(
            chunk_id="business/185::chunk_0",
            text=(
                "The company reported changes "
                "in advertising revenue."
            ),
            reranker_score=-1.1032,
            reranker_rank=2,
            metadata={
                "document_name": "185.txt",
                "category": "business",
            },
        ),
    ]

    run_test("What happened to AOL subscribers?",relevant_evidence)

    run_test("What is the capital of France?",unrelated_evidence)

if __name__ == "__main__":
    main()