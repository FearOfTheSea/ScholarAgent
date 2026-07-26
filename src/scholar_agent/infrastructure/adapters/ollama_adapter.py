"""Local Ollama implementation of the language-model port."""

from typing import Any

import httpx

from scholar_agent.application.output_ports.llm_provider import ILLMProvider


class OllamaAdapter(ILLMProvider):
    """Uses the local Ollama HTTP API without external inference services."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        context_length: int,
        maximum_tokens: int,
        client: httpx.Client | None = None,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._context_length = context_length
        self._maximum_tokens = maximum_tokens
        self._client = client or httpx.Client(timeout=60.0)

    def generate(self, prompt: str) -> str:
        """Generate one non-streaming response from the configured local model."""
        try:
            response = self._client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model_name,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "options": {
                        "num_ctx": self._context_length,
                        "num_predict": self._maximum_tokens,
                        "temperature": 0.2,
                    },
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise RuntimeError("The local Ollama service is unavailable.") from error

        payload = self._response_payload(response)
        generated_text = payload.get("response")
        if not isinstance(generated_text, str) or not generated_text.strip():
            raise RuntimeError("Ollama returned an empty response.")
        return generated_text.strip()

    def is_available(self) -> bool:
        """Return whether the local Ollama API responds to a model-list request."""
        try:
            response = self._client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    def has_model(self) -> bool:
        """Return whether the configured model is installed in local Ollama."""
        try:
            response = self._client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        payload = self._response_payload(response)
        models = payload.get("models")
        if not isinstance(models, list):
            return False
        return any(
            isinstance(model, dict) and model.get("name") == self._model_name
            for model in models
        )

    @staticmethod
    def _response_payload(response: httpx.Response) -> dict[str, Any]:
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Ollama returned an invalid response payload.")
        return payload
