from src.ingestion.bbc_loader import load_bbc_corpus
from src.nlp.preprocess import preprocess_document
from src.ingestion.chunker import chunk_document

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.rrf import reciprocal_rank_fusion


DATA_PATH = "data/raw/bbc"


def build_chunks():

    documents = load_bbc_corpus(DATA_PATH)

    all_chunks = []

    for document in documents:

        document = preprocess_document(document)

        all_chunks.extend(
            chunk_document(document)
        )

    return all_chunks


def main():

    print("Building chunks...")

    chunks = build_chunks()

    print(f"Total chunks: {len(chunks)}")

    print("\nInitializing BM25...")

    bm25 = BM25Retriever(chunks)

    print("\nInitializing dense retriever...")

    dense = DenseRetriever()

    query = "What happened to AOL subscribers?"

    print("\nRunning BM25...")

    bm25_results = bm25.search(
        query,
        top_k=10,
    )

    print("Running dense retrieval...")

    dense_results = dense.search(
        query,
        top_k=10,
    )

    print("Running RRF...")

    fused_results = reciprocal_rank_fusion(
        [
            bm25_results,
            dense_results,
        ],
        top_k=10,
    )

    print("\n" + "=" * 80)
    print("HYBRID RRF RESULTS")
    print("=" * 80)

    for result in fused_results:

        metadata = result.metadata or {}

        print("\n" + "-" * 80)

        print(f"Final Rank       : {result.rank}")
        print(f"Chunk ID         : {result.chunk_id}")

        print(
            f"BM25 Score      : "
            f"{result.bm25_score}"
        )

        print(
            f"Dense Score     : "
            f"{result.dense_score}"
        )

        print(
            f"RRF Score       : "
            f"{result.fusion_score:.6f}"
        )

        print(
            f"Category        : "
            f"{metadata.get('category')}"
        )

        print(
            f"Document        : "
            f"{metadata.get('document_name')}"
        )

        print("\nText:")
        print(result.text)


if __name__ == "__main__":
    main()