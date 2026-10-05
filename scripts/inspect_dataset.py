from collections import Counter
from src.ingestion.bbc_loader import load_bbc_corpus

DATA_PATH = "data/raw/bbc"

def main():
    documents = load_bbc_corpus(DATA_PATH)
    print(f"\nTotal documents: {len(documents)}")

    category_counts = Counter(
        document.category
        for document in documents
    )

    print("\nDocuments per category:")

    for category, count in sorted(category_counts.items()):
        print(f"{category:15} {count}")

    print("\nFirst document:")
    print("-------------------------")

    first = documents[0]

    print("Document ID :", first.document_id)
    print("Filename    :", first.document_name)
    print("Category    :", first.category)
    print("Title       :", first.title)
    print("Source      :", first.source_path)
    print("\nText preview:")
    print(first.text[:500])


if __name__ == "__main__":
    main()