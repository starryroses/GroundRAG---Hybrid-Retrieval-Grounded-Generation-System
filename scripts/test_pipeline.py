from src.generation.groq_llm import GroqLLM
from src.grounding.verifier import GroundingVerifier
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.ingestion.bbc_loader import load_bbc_corpus
from src.ingestion.chunker import chunk_document
from src.nlp.preprocess import preprocess_document

def main():

    # -----------------------------------------
    # LOAD DOCUMENTS
    # -----------------------------------------
    documents = load_bbc_corpus("data/raw/bbc")
    # -----------------------------------------
    # PREPROCESS DOCUMENTS
    # -----------------------------------------
    processed_documents = [preprocess_document(document) for document in documents]
    # -----------------------------------------
    # CHUNK DOCUMENTS
    # -----------------------------------------
    chunks = []
    for document in processed_documents:
        document_chunks = chunk_document(document)
        chunks.extend(document_chunks)
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    # -----------------------------------------
    # RETRIEVERS
    # -----------------------------------------
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever()
    # -----------------------------------------
    # RERANKER
    # -----------------------------------------
    reranker = CrossEncoderReranker()
    # -----------------------------------------
    # GENERATOR + VERIFIER
    # -----------------------------------------
    llm = GroqLLM()
    verifier = GroundingVerifier()
    # -----------------------------------------
    # PIPELINE
    # -----------------------------------------
    pipeline = RAGPipeline(
        bm25_retriever=bm25,
        dense_retriever=dense,
        reranker=reranker,
        llm=llm,
        verifier=verifier,
    )
    # -----------------------------------------
    # QUERY
    # -----------------------------------------
    query = "What happened to AOL subscribers?"
    response = pipeline.answer(query)
    # -----------------------------------------
    # OUTPUT
    # -----------------------------------------
    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80) 

    print(response.answer)
    print("\nGrounded:")
    print(response.grounded)

    print("\nReason:")
    print(response.grounding_reason)

    print("\nCitations:")
    for citation in response.citations:
        print(
            f"[{citation.citation_id}] "
            f"{citation.document_name} "
            f"({citation.chunk_id})"
        )

    print("\nRetrieval Inspector:")
    for result in response.retrieval_results:
        print(
            f"\nRank: {result.reranker_rank}"
            f"\nChunk: {result.chunk_id}"
            f"\nBM25: {result.bm25_score}"
            f"\nDense: {result.dense_score}"
            f"\nRRF: {result.fusion_score}"
            f"\nReranker: {result.reranker_score}"
        )

if __name__ == "__main__":
    main()