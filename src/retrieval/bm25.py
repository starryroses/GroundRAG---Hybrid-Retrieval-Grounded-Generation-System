from rank_bm25 import BM25Okapi
from src.ingestion.bbc_loader import load_bbc_corpus
from src.ingestion.chunker import chunk_document
from src.nlp.preprocess import preprocess_document
from src.models import Chunk, RetrievalResult
from src.nlp.preprocess import tokenize_for_bm25


class BM25Retriever:

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Cannot build BM25 index with no chunks.")

        self.chunks = chunks
        self.tokenized_chunks = [
            tokenize_for_bm25(chunk.text)
            for chunk in chunks
        ]
        self.index = BM25Okapi(self.tokenized_chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:

        query_tokens = tokenize_for_bm25(query)
        scores = self.index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for rank, index in enumerate(ranked_indices, start=1):
            chunk = self.chunks[index]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    rank=rank,
                    bm25_score=float(scores[index]),
                    metadata={
                        "document_id": chunk.document_id,
                        "document_name": chunk.document_name,
                        "title": chunk.title,
                        "category": chunk.category,
                        "chunk_index": chunk.chunk_index,
                        "source_path": chunk.source_path,
                    },
                )
           )

        return results