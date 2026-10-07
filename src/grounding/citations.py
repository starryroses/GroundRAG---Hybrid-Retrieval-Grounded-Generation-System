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