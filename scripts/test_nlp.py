from src.ingestion.bbc_loader import load_bbc_corpus
from src.nlp.preprocess import (
    preprocess_document,
    preprocess_query,
    tokenize_for_bm25,
)


DATA_PATH = "data/raw/bbc"

def main():

    documents = load_bbc_corpus(DATA_PATH)
    document = documents[0]
    preprocess_document(document)
    print("\n===== ORIGINAL =====")
    print(document.text[:500])

    print("\n===== NORMALIZED =====")
    print(document.normalized_text[:500])

    print("\n===== SENTENCES =====")
    for i, sentence in enumerate(document.sentences[:5]):
        print(f"{i}: {sentence}")

    print("\n===== BM25 TOKENS =====")
    tokens = tokenize_for_bm25(document.normalized_text)
    print(tokens[:30])

    print("\n===== QUERY =====")
    query = "What did the IAAF do to combat doping?"
    print(preprocess_query(query))

if __name__ == "__main__":
    main()