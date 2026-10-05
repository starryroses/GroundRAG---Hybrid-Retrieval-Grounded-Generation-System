from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from src.models import Chunk, RetrievalResult

class DenseRetriever:

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "bbc_chunks_bge_small_cosine_v1",
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):

        self.model = SentenceTransformer(model_name)
        Path(persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": model_name,
            },
        )

    def index_chunks(self,chunks: list[Chunk],batch_size: int = 64):
        if not chunks:
            raise ValueError("Cannot index an empty chunk list.")

        existing_count = self.collection.count()
        if existing_count > 0:
            print(
                f"Collection already contains "
                f"{existing_count} chunks."
            )
            return

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            ids = [chunk.chunk_id for chunk in batch]

            documents = texts
            metadatas = [
                {
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "title": chunk.title,
                    "category": chunk.category,
                    "chunk_index": chunk.chunk_index,
                    "source_path": chunk.source_path,
                }
                for chunk in batch
            ]
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
            )
            print(
                f"Indexed {min(start + batch_size, len(chunks))}"
                f"/{len(chunks)} chunks"
            )

    def search(self, query: str,top_k: int = 10,) -> list[RetrievalResult]:
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        chroma_results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["documents","metadatas","distances"],
        )

        retrieved = []
        ids = chroma_results["ids"][0]
        documents = chroma_results["documents"][0]
        metadatas = chroma_results["metadatas"][0]
        distances = chroma_results["distances"][0]

        for rank, (chunk_id,document,metadata,distance) in enumerate(zip(ids, documents,metadatas,distances), start=1):
            retrieved.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=document,
                    dense_score=1.0 - float(distance),
                    dense_distance=float(distance),
                    dense_rank=rank,
                    metadata=metadata,
                )
            )

        return retrieved