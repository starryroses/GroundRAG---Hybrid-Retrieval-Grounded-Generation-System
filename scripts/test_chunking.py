from src.ingestion.bbc_loader import load_bbc_corpus
from src.nlp.preprocess import preprocess_document
from src.ingestion.chunker import chunk_document

DATA_PATH = "data/raw/bbc"

def main():

    documents = load_bbc_corpus(DATA_PATH)
    document = documents[0]
    document = preprocess_document(document)
    chunks = chunk_document(document)

    print("=" * 70)
    print("DOCUMENT")
    print("=" * 70)

    print(f"ID       : {document.document_id}")
    print(f"Title    : {document.title}")
    print(f"Category : {document.category}")
    print(f"Sentences: {len(document.sentences)}")

    print("\n" + "=" * 70)
    print("CHUNKS")
    print("=" * 70)

    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks[:5]:

        print("\n" + "-" * 70)

        print(f"Chunk ID : {chunk.chunk_id}")
        print(f"Index    : {chunk.chunk_index}")
        print(f"Length   : {len(chunk.text)}")

        print("\nText:")
        print(chunk.text)

if __name__ == "__main__":
    main()