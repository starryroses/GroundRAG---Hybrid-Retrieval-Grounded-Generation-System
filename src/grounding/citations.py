import re
from src.models import RetrievalResult

CITATION_PATTERN = re.compile(r"\[Evidence\s+(\d+)\]")


def validate_citations(answer: str,evidence: list[RetrievalResult]) -> dict:
    citations = CITATION_PATTERN.findall(answer)
    if not citations:
        return {
            "valid": False,
            "reason": "No evidence citations found.",
            "citations": [],
        }
    
    available_ids = set(range(1, len(evidence) + 1))
    cited_ids = {int(citation) for citation in citations}
    invalid_ids = cited_ids - available_ids

    if invalid_ids:
        return {
            "valid": False,
            "reason": (
                f"Invalid evidence citations: "
                f"{sorted(invalid_ids)}"
            ),
            "citations": sorted(cited_ids),
        }
    return {
        "valid": True,
        "reason": "All evidence citations are valid.",
        "citations": sorted(cited_ids),
    }

def build_citations(evidence: list[RetrievalResult], cited_ids: list[int]) -> list:
    from src.models import Citation
    citations = []
    for citation_id in cited_ids:
        evidence_index = citation_id - 1
        if evidence_index < 0 or evidence_index >= len(evidence):
            continue

        result = evidence[evidence_index]
        metadata = result.metadata or {}

        citations.append(
            Citation(
                citation_id=citation_id,
                chunk_id=result.chunk_id,
                document_name=metadata.get(
                    "document_name",
                    "Unknown",
                ),
                category=metadata.get(
                    "category",
                    "Unknown",
                ),
                text=result.text,
            )
        )

    return citations