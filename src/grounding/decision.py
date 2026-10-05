from dataclasses import dataclass
from src.models import RetrievalResult

REFUSAL_MESSAGE = ("Sorry, I'm not equipped with that knowledge.")

@dataclass
class GroundingDecision:
    allowed: bool
    reason: str
    evidence: list[RetrievalResult]

def decide_grounding(candidates: list[RetrievalResult],min_evidence: int = 1, min_score_margin: float = 0.0) -> GroundingDecision:
    if not candidates:
        return GroundingDecision(
            allowed=False,
            reason="No evidence was retrieved.",
            evidence=[],
        )

    usable_candidates = [candidate for candidate in candidates if candidate.text.strip()]

    if len(usable_candidates) < min_evidence:
        return GroundingDecision(
            allowed=False,
            reason="Insufficient usable evidence.",
            evidence=usable_candidates,
        )

    reranked_candidates = [candidate for candidate in usable_candidates if candidate.reranker_score is not None]

    if not reranked_candidates:
        return GroundingDecision(
            allowed=False,
            reason="No reranker scores are available.",
            evidence=usable_candidates,
        )
    

    reranked_candidates.sort(
        key=lambda candidate: candidate.reranker_score,
        reverse=True,
    )

    best_score = reranked_candidates[0].reranker_score

    if best_score <= min_score_margin:
        return GroundingDecision(
            allowed=False,
            reason=(
                "Retrieved evidence does not appear "
                "sufficiently relevant."
            ),
            evidence=reranked_candidates,
        )

    return GroundingDecision(
        allowed=True,
        reason="Strong relevant evidence found.",
        evidence=reranked_candidates,
    )