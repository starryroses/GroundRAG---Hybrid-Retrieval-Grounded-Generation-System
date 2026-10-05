from src.ingestion.bbc_loader import load_bbc_corpus
from src.nlp.preprocess import preprocess_document
from src.ingestion.chunker import chunk_document
from src.retrieval.dense import DenseRetriever

DATA_PATH = "data/raw/bbc"

def build_chunks():

    documents = load_bbc_corpus(DATA_PATH)
    all_chunks = []
    for i, document in enumerate(documents, start=1):

        document = preprocess_document(document)
        chunks = chunk_document(document)
        all_chunks.extend(chunks)
        if i % 100 == 0:
            print(f"Processed {i}/{len(documents)} documents")

    return all_chunks


def main():

    print("Loading and chunking BBC corpus...")
    chunks = build_chunks()
    print(f"\nTotal chunks: {len(chunks)}")

    print("\nInitializing dense retriever...")
    retriever = DenseRetriever()

    print("\nBuilding ChromaDB index...")
    retriever.index_chunks(chunks)

    print("\nDense index ready.")


if __name__ == "__main__":
    main()