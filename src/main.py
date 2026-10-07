from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.app.container import RAGContainer
from src.app import state
from src.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\n" + "=" * 80)
    print("STARTING APPLICATION")
    print("=" * 80)

    # Initialize the complete RAG system once.
    state.rag_container = RAGContainer()

    print("\n" + "=" * 80)
    print("APPLICATION STARTUP COMPLETE")
    print("=" * 80)

    yield
    print("\n" + "=" * 80)
    print("SHUTTING DOWN APPLICATION")
    print("=" * 80)

    state.rag_container = None
    print("Application shutdown complete.")


app = FastAPI(
    title="Advanced RAG System",
    description="Production-style hybrid RAG system",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "rag_initialized": state.rag_container is not None,
    }