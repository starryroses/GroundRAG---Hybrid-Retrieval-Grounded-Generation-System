import json
from src.generation.groq_llm import GroqLLM
from src.models import RetrievalResult

class GroundingVerifier:
    def __init__(self,model_name: str = "openai/gpt-oss-20b"):
        self.llm = GroqLLM(model_name=model_name)

    def verify(self,query: str,answer: str,evidence: list[RetrievalResult])-> dict:
        evidence_text = "\n\n".join(
            f"[Evidence {index}]\n{result.text}"
            for index, result in enumerate(
                evidence,
                start=1,
            )
        )

        prompt = f"""You are a strict grounding verifier.

Your task is to determine whether the ANSWER is fully
supported by the provided EVIDENCE.

Do not use your own knowledge.

QUESTION: {query}
EVIDENCE: {evidence_text}
ANSWER: {answer}

Return ONLY valid JSON in exactly this structure:
{{
    "grounded": true,
    "reason": "brief explanation"
}}
Rules:
1. grounded must be true only if the answer is supported
   by the evidence.
2. If the answer contains a factual claim that is not
   supported by the evidence, grounded must be false.
3. If the answer contradicts the evidence, grounded must
   be false.
4. Do not consider outside knowledge.
5. Do not add any fields.
"""
        response = self.llm.generate(prompt=prompt,temperature=0.0)
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            return {
                "grounded": False,
                "reason": "Verifier returned invalid JSON.",
            }

        if not isinstance(result.get("grounded"), bool):
            return {
                "grounded": False,
                "reason": "Verifier returned an invalid grounding value.",
            }

        return result