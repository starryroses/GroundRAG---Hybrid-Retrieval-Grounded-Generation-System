from sentence_transformers import CrossEncoder
from src.models import RetrievalResult

class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str,candidates: list[RetrievalResult],top_k: int = 5) -> list[RetrievalResult]:
        if not candidates:
            return []

        pairs = [[query, candidate.text] for candidate in candidates]
        scores = self.model.predict(pairs)
        scored_candidates = []

        for candidate, score in zip(candidates,scores):
            scored_candidates.append((candidate,float(score)))

        scored_candidates.sort(key=lambda item: item[1], reverse=True)

        final_results = []

        for reranker_rank, (candidate,score) in enumerate(scored_candidates[:top_k], start=1):
            candidate.reranker_score = score
            candidate.reranker_rank = reranker_rank
            final_results.append(candidate)

        return final_results