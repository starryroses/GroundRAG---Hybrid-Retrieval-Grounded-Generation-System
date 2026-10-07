from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.app import state

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.post("/query")
def query_rag(request: QueryRequest):
    if state.rag_container is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not initialized.",
        )
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )
    result = state.rag_container.answer(query)
    return {
        "query": result.query,
        "answer": result.answer,
        "grounded": result.grounded,
        "grounding_reason": result.grounding_reason,
        "citations": [
            {
                "citation_id": citation.citation_id,
                "chunk_id": citation.chunk_id,
                "document_name": citation.document_name,
                "category": citation.category,
                "text": citation.text,
            }
            for citation in result.citations
        ],
        "retrieval_results": [
            {
                "reranker_rank": retrieval.reranker_rank,
                "chunk_id": retrieval.chunk_id,
                "bm25_score": retrieval.bm25_score,
                "dense_score": retrieval.dense_score,
                "dense_distance": retrieval.dense_distance,
                "fusion_score": retrieval.fusion_score,
                "reranker_score": retrieval.reranker_score,
                "bm25_rank": retrieval.bm25_rank,
                "dense_rank": retrieval.dense_rank,
                "fusion_rank": retrieval.fusion_rank,
                "reranker_rank": retrieval.reranker_rank,
                "metadata": retrieval.metadata,
            }
            for retrieval in result.retrieval_results
        ],
    }