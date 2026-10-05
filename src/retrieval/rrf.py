from collections import defaultdict
from src.models import RetrievalResult

DEFAULT_RRF_K = 60

def reciprocal_rank_fusion(result_lists: list[list[RetrievalResult]],k: int = DEFAULT_RRF_K,top_k: int = 10) -> list[RetrievalResult]:

    fused_scores = defaultdict(float)
    results_by_chunk: dict[str, RetrievalResult] = {}

    for results in result_lists:

        for result in results:
            # result.rank is no longer available,
            # so use the appropriate source rank.
            if result.bm25_rank is not None:
                source_rank = result.bm25_rank
            elif result.dense_rank is not None:
                source_rank = result.dense_rank
            else:
                raise ValueError(f"No source rank for {result.chunk_id}")

            fused_scores[result.chunk_id] += (1.0 / (k + source_rank))

            if result.chunk_id not in results_by_chunk:
                results_by_chunk[result.chunk_id] = result

            else:
                existing = results_by_chunk[result.chunk_id]
                if result.bm25_score is not None:
                    existing.bm25_score = result.bm25_score
                    existing.bm25_rank = result.bm25_rank

                if result.dense_score is not None:
                    existing.dense_score = result.dense_score
                    existing.dense_distance = result.dense_distance
                    existing.dense_rank = result.dense_rank

    ranked_chunks = sorted(fused_scores.items(),key=lambda item: item[1],reverse=True)

    final_results = []

    for fusion_rank, (chunk_id, fusion_score) in enumerate(ranked_chunks[:top_k],start=1):

        result = results_by_chunk[chunk_id]
        final_results.append(
            RetrievalResult(
                chunk_id=result.chunk_id,
                text=result.text,
                bm25_score=result.bm25_score,
                dense_score=result.dense_score,
                dense_distance=result.dense_distance,
                fusion_score=fusion_score,
                bm25_rank=result.bm25_rank,
                dense_rank=result.dense_rank,
                fusion_rank=fusion_rank,
                metadata=result.metadata,
            )
        )

    return final_results