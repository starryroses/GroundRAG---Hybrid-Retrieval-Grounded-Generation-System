from src.grounding.citations import (build_citations,validate_citations)
from src.grounding.context import (build_grounding_context)
from src.grounding.decision import (decide_grounding,REFUSAL_MESSAGE,)
from src.grounding.prompt import (build_grounded_prompt,)
from src.grounding.verifier import (GroundingVerifier,)
from src.generation.groq_llm import (GroqLLM,)
from src.models import RAGResponse

class RAGPipeline:
    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        reranker,
        llm,
        verifier,
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        fusion_top_k: int = 20,
        final_top_k: int = 5,
    ):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.reranker = reranker
        self.llm = llm
        self.verifier = verifier
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.fusion_top_k = fusion_top_k
        self.final_top_k = final_top_k

    def answer(self,query: str) -> RAGResponse:

        # -----------------------------------------
        # 1. BM25 RETRIEVAL
        # -----------------------------------------
        bm25_results = self.bm25_retriever.search(
            query,
            top_k=self.bm25_top_k,
        )
        # -----------------------------------------
        # 2. DENSE RETRIEVAL
        # -----------------------------------------
        dense_results = self.dense_retriever.search(
            query,
            top_k=self.dense_top_k,
        )
        # -----------------------------------------
        # 3. HYBRID FUSION
        # -----------------------------------------
        from src.retrieval.rrf import (reciprocal_rank_fusion,)
        fused_results = reciprocal_rank_fusion(
            [
                bm25_results,
                dense_results,
            ],
            top_k=self.fusion_top_k,
        )
        # -----------------------------------------
        # 4. CROSS-ENCODER RERANKING
        # -----------------------------------------
        reranked_results = self.reranker.rerank(
            query,
            fused_results,
            top_k=self.final_top_k,
        )
        # -----------------------------------------
        # 5. GROUNDING DECISION
        # -----------------------------------------
        grounding = decide_grounding(reranked_results)
        if not grounding.allowed:
            return RAGResponse(
                query=query,
                answer=REFUSAL_MESSAGE,
                citations=[],
                grounded=False,
                grounding_reason=grounding.reason,
                retrieval_results=reranked_results,
            )
        # -----------------------------------------
        # 6. BUILD CONTEXT
        # -----------------------------------------
        context = build_grounding_context(grounding.evidence)
        # -----------------------------------------
        # 7. GROUNDED GENERATION
        # -----------------------------------------
        prompt = build_grounded_prompt(query=query,context=context)
        answer = self.llm.generate(prompt,temperature=0.0,)
        # -----------------------------------------
        # 8. POST-GENERATION VERIFICATION
        # -----------------------------------------
        verification = self.verifier.verify(
            query=query,
            answer=answer,
            evidence=grounding.evidence,
        )
        if not verification["grounded"]:
            return RAGResponse(
                query=query,
                answer=REFUSAL_MESSAGE,
                citations=[],
                grounded=False,
                grounding_reason=(
                    "Generated answer failed grounding "
                    f"verification: "
                    f"{verification['reason']}"
                ),
                retrieval_results=reranked_results,
            )
        # -----------------------------------------
       # 9. CITATION VALIDATION
        # -----------------------------------------
        citation_validation = validate_citations(answer,grounding.evidence,)
        if not citation_validation["valid"]:
            return RAGResponse(
                query=query,
                answer=REFUSAL_MESSAGE,
                citations=[],
                grounded=False,
                grounding_reason=(
                    "Generated answer failed citation "
                    f"validation: "
                    f"{citation_validation['reason']}"
                ),
                retrieval_results=reranked_results,
            )
        # -----------------------------------------
        # 10. BUILD FINAL CITATIONS
        # -----------------------------------------
        citations = build_citations(grounding.evidence,citation_validation["citations"],)
        return RAGResponse(
            query=query,
            answer=answer,
            citations=citations,
            grounded=True,
            grounding_reason=(
                "Answer passed retrieval, grounding, "
                "and citation validation."
            ),
            retrieval_results=reranked_results,
        )