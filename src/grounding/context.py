from src.models import RetrievalResult

def build_grounding_context(evidence: list[RetrievalResult]) -> str:
    context_blocks = []
    for index, result in enumerate(evidence,start=1):
        metadata = result.metadata or {}
        document_name = metadata.get("document_name","Unknown document")

        chunk_id = result.chunk_id
        context_blocks.append(
            f"[Evidence {index}]\n"
            f"Document: {document_name}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Text: {result.text}"
        )

    return "\n\n".join(context_blocks)