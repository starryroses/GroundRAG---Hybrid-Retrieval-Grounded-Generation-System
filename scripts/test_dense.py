from src.retrieval.dense import DenseRetriever


def main():

    retriever = DenseRetriever()

    query = "What happened to AOL subscribers?"

    results = retriever.search(
        query=query,
        top_k=5,
    )

    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for result in results:

        print("\n" + "-" * 80)

        print(f"Rank     : {result['rank']}")
        print(f"Distance : {result['distance']:.4f}")
        print(f"Chunk ID : {result['chunk_id']}")
        print(
            f"Document : "
            f"{result['metadata']['document_name']}"
        )
        print(
            f"Category : "
            f"{result['metadata']['category']}"
        )

        print("\nText:")
        print(result["text"])


if __name__ == "__main__":
    main()