import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class GroqLLM:
    def __init__(self,model_name: str = "openai/gpt-oss-20b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file."
            )

        self.model_name = model_name
        self.client = Groq(api_key=api_key)

    def generate(self, prompt: str,temperature: float = 0.0,) -> str:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temperature,
            max_completion_tokens=512,
        )

        return response.choices[0].message.content.strip()