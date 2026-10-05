from src.grounding.context import build_grounding_context
from src.grounding.prompt import build_grounded_prompt
from src.generation.groq_llm import GroqLLM

from src.models import RetrievalResult


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
            metadata={
                "document_name": "001.txt",
                "category": "business",
            },
        ),
    ]

    query = "What happened to AOL subscribers?"

    context = build_grounding_context(evidence)

    prompt = build_grounded_prompt(
        query=query,
        context=context,
    )

    llm = GroqLLM()

    answer = llm.generate(
        prompt=prompt,
        temperature=0.0,
    )

    print("=" * 70)
    print("GROUNDED GENERATION")
    print("=" * 70)

    print("\nQuestion:")
    print(query)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()