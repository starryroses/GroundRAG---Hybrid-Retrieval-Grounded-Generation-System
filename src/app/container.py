from src.ingestion.bbc_loader import load_bbc_corpus
from src.ingestion.chunker import chunk_document
from src.nlp.preprocess import preprocess_document
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.generation.groq_llm import GroqLLM
from src.grounding.verifier import GroundingVerifier
from src.pipeline.rag_pipeline import RAGPipeline


class RAGContainer:
    """
    Initializes and owns all components required by the RAG system.
    Components are created once when the application starts and
    reused for subsequent queries.
    """
    def __init__(self,data_path: str = "data/raw/bbc"):
        print("=" * 80)
        print("INITIALIZING RAG SYSTEM")
        print("=" * 80)
        # --------------------------------------------------
        # 1. Load documents
        # --------------------------------------------------
        print("\n[1/7] Loading BBC corpus...")
        documents = load_bbc_corpus(data_path)
        print(f"Loaded {len(documents)} documents.")
        # --------------------------------------------------
        # 2. Preprocess documents
        # --------------------------------------------------
        print("\n[2/7] Preprocessing documents...")
        processed_documents = [preprocess_document(document) for document in documents]
        # --------------------------------------------------
        # 3. Create chunks
        # --------------------------------------------------
        print("\n[3/7] Creating chunks...")
        chunks = []
        for document in processed_documents:
            chunks.extend(chunk_document(document))
        print(f"Created {len(chunks)} chunks.")
        # --------------------------------------------------
        # 4. Initialize retrieval
        # --------------------------------------------------
        print("\n[4/7] Initializing BM25...")
        self.bm25_retriever = BM25Retriever(chunks)
        print("BM25 ready.")
 
        print("\nInitializing dense retriever...")
        self.dense_retriever = DenseRetriever()
        print("Dense retriever ready.")
        # --------------------------------------------------
        # 5. Initialize reranker
        # --------------------------------------------------
        print("\n[5/7] Loading cross-encoder...")
        self.reranker = CrossEncoderReranker()
        print("Cross-encoder ready.")
        # --------------------------------------------------
        # 6. Initialize generation + verification
        # --------------------------------------------------
        print("\n[6/7] Initializing Groq LLM...")
        self.llm = GroqLLM()
        print("Groq LLM ready.")

        print("\nInitializing grounding verifier...")
        self.verifier = GroundingVerifier()
        print("Grounding verifier ready.")
        # --------------------------------------------------
        # 7. Construct pipeline
        # --------------------------------------------------
        print("\n[7/7] Building RAG pipeline...")
        self.pipeline = RAGPipeline(
            bm25_retriever=self.bm25_retriever,
            dense_retriever=self.dense_retriever,
            reranker=self.reranker,
            llm=self.llm,
            verifier=self.verifier,
        )

        print("\n" + "=" * 80)
        print("RAG SYSTEM READY")
        print("=" * 80)

    def answer(self, query: str):
        """
        Run a query through the already initialized RAG system.
        """
        return self.pipeline.answer(query)