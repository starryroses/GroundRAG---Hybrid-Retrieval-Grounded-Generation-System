from src.ingestion.bbc_loader import load_bbc_corpus
from src.nlp.preprocess import preprocess_document
from src.ingestion.chunker import chunk_document

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.rrf import reciprocal_rank_fusion

from src.reranking.cross_encoder import CrossEncoderReranker


DATA_PATH = "data/raw/bbc"


def build_chunks():

    documents = load_bbc_corpus(DATA_PATH)

    chunks = []

    for document in documents:

        document = preprocess_document(document)

        chunks.extend(
            chunk_document(document)
        )

    return chunks


def main():

    chunks = build_chunks()

    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever()

    reranker = CrossEncoderReranker()

    query = "What happened to AOL subscribers?"

    bm25_results = bm25.search(
        query,
        top_k=20,
    )

    dense_results = dense.search(
        query,
        top_k=20,
    )

    hybrid_results = reciprocal_rank_fusion(
        [
            bm25_results,
            dense_results,
        ],
        top_k=20,
    )

    reranked_results = reranker.rerank(
        query,
        hybrid_results,
        top_k=5,
    )

    print("\n" + "=" * 80)
    print("RERANKED RESULTS")
    print("=" * 80)

    for result in reranked_results:

        print(
            f"\nReranker Rank: "
            f"{result.reranker_rank}"
        )

        print(
            f"Chunk: "
            f"{result.chunk_id}"
        )

        print(
            f"BM25 Rank: "
            f"{result.bm25_rank}"
        )

        print(
            f"Dense Rank: "
            f"{result.dense_rank}"
        )

        print(
            f"RRF Rank: "
            f"{result.fusion_rank}"
        )

        print(
            f"RRF Score: "
            f"{result.fusion_score}"
        )

        print(
            f"Reranker Score: "
            f"{result.reranker_score:.4f}"
        )

        print(
            f"Text: "
            f"{result.text[:500]}"
        )


if __name__ == "__main__":
    main()