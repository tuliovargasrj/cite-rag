"""The boundary with the paid model. Tests replace it with a fake client."""

from typing import Protocol

from cite_rag.prompts import ModelAnswer

DEFAULT_MODEL = "gpt-4o-mini"
_MAX_OUTPUT_TOKENS = 1000
_TIMEOUT_SECONDS = 60.0


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text, which the caller must not trust."""
        ...


class OpenAIClient:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        # Imported here so that the free commands never load the SDK.
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(timeout=_TIMEOUT_SECONDS)

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_completion_tokens=_MAX_OUTPUT_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "cited_answer",
                    "strict": True,
                    "schema": ModelAnswer.model_json_schema(),
                },
            },
        )
        # A model refusal has no content. An empty string then fails validation.
        return response.choices[0].message.content or ""
