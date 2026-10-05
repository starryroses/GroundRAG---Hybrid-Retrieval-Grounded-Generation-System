from src.ingestion.bbc_loader import load_bbc_corpus
from src.nlp.preprocess import preprocess_document
from src.ingestion.chunker import chunk_document
from src.retrieval.bm25 import BM25Retriever

DATA_PATH = "data/raw/bbc"

def build_chunks():

    documents = load_bbc_corpus(DATA_PATH)
    all_chunks = []
    for document in documents:
        document = preprocess_document(document)
        chunks = chunk_document(document)
        all_chunks.extend(chunks)
    return all_chunks


def main():

    print("Loading BBC corpus...")
    chunks = build_chunks()
    print(f"Total chunks: {len(chunks)}")
    print("\nBuilding BM25 index...")

    retriever = BM25Retriever(chunks)
    print("BM25 index ready.")

    query = "What happened to AOL subscribers?"

    results = retriever.search(
        query=query,
        top_k=5,
    )

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for result in results:

        chunk = result["chunk"]
        print("\n" + "-" * 80)

        print(f"Rank       : {result['rank']}")
        print(f"BM25 score : {result['bm25_score']:.4f}")
        print(f"Chunk ID   : {chunk.chunk_id}")
        print(f"Document   : {chunk.document_name}")
        print(f"Category   : {chunk.category}")

        print("\nText:")
        print(chunk.text)

if __name__ == "__main__":
    main()