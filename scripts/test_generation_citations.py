from src.generation.groq_llm import GroqLLM
from src.grounding.context import build_grounding_context
from src.grounding.prompt import build_grounded_prompt
from src.models import RetrievalResult


evidence = [
    RetrievalResult(
        chunk_id="tech/132::chunk_0",
        text=(
            "A brother and sister in the US have been convicted "
            "of sending hundreds of thousands of unsolicited "
            "e-mail messages to AOL subscribers."
        ),
        reranker_score=2.8245,
    ),
    RetrievalResult(
        chunk_id="business/001::chunk_1",
        text=(
            "It lost 464,000 subscribers in the fourth quarter. "
            "The company said AOL's underlying profit before "
            "exceptional items rose 8%."
        ),
        reranker_score=2.7891,
    ),
]

context = build_grounding_context(evidence)

prompt = build_grounded_prompt(
    query="What happened to AOL subscribers?",
    context=context,
)

llm = GroqLLM()

answer = llm.generate(
    prompt,
    temperature=0.0,
)

print("\n" + "=" * 80)
print("GENERATED ANSWER")
print("=" * 80)
print(answer)