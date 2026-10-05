from collections import defaultdict
from src.models import RetrievalResult

DEFAULT_RRF_K = 60

def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = DEFAULT_RRF_K,
    top_k: int = 10,
) -> list[RetrievalResult]:

    fused_scores = defaultdict(float)
    results_by_chunk = {}

    for results in result_lists:
        for result in results:
            fused_scores[result.chunk_id] += (1.0 / (k + result.rank))
            if result.chunk_id not in results_by_chunk:
                results_by_chunk[result.chunk_id] = result

    ranked = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    final_results = []
    for final_rank, (chunk_id, fusion_score) in enumerate(
        ranked[:top_k],
        start=1,
    ):
        original = results_by_chunk[chunk_id]
        final_results.append(
            RetrievalResult(
                chunk_id=original.chunk_id,
                text=original.text,
                rank=final_rank,
                bm25_score=original.bm25_score,
                dense_score=original.dense_score,
                dense_distance=original.dense_distance,
                fusion_score=fusion_score,
                metadata=original.metadata,
            )
        )

    return final_results