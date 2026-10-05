from dataclasses import dataclass
from typing import Optional

@dataclass
class Document:
    document_id: str
    document_name: str
    title: str
    text: str
    category: str
    source_path: str
    date: Optional[str] = None

    #generated during preprocessing
    normalized_text: Optional[str] = None
    sentences: Optional[list[str]] = None

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    document_name: str
    title: str
    category: str

    chunk_index: int
    text: str

    source_path: str
    page_number: Optional[int] = None

@dataclass
class RetrievalResult:
    chunk_id: str
    text: str

    # Retrieval scores
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    dense_distance: Optional[float] = None
    fusion_score: Optional[float] = None

    # Ranks from individual retrieval systems
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    fusion_rank: Optional[int] = None
    reranker_score: Optional[float] = None
    reranker_rank: Optional[int] = None

    # Metadata
    metadata: Optional[dict[str, any]] = None