"""OpenAI-compatible chat-completions LLM backend — Story 4.1 concrete implementation.

Implements LLMInterface against any endpoint that speaks the OpenAI
``/chat/completions`` protocol. This covers the Nous Portal (Hermes models),
OpenRouter, and local servers such as Ollama, llama.cpp, or vLLM, so
swapping models is a configuration change rather than a code change.

Configuration (environment variables, read by ``from_env``):
    DOCURESEARCH_LLM_BASE_URL  — e.g. https://inference-api.nousresearch.com/v1
    DOCURESEARCH_LLM_MODEL     — model name as the provider expects it
    DOCURESEARCH_LLM_API_KEY   — bearer token (optional for local servers)
    DOCURESEARCH_LLM_TIMEOUT   — request timeout in seconds (default 120)

The backend only transports text. Parsing into a GeneratedAnswer is done
by ``app.generation.schema.parse_llm_output`` so every backend is held to
the same output contract.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from app.generation.interface import LLMResponse
from app.generation.schema import parse_llm_output

_SYSTEM_MESSAGE = (
    "You are a careful research assistant. You answer only from the source "
    "passages you are given and you always respond with a single JSON object."
)


class LLMConfigError(RuntimeError):
    """Raised when the LLM backend is not configured."""


class OpenAICompatibleLLM:
    """LLMInterface implementation for OpenAI-compatible chat endpoints.

    Args:
        base_url: API base URL, including the version prefix (``.../v1``).
        model: Model name to request.
        api_key: Bearer token. May be None for local servers.
        timeout_seconds: HTTP timeout for a single generation call.
        temperature: Sampling temperature. Defaults to 0 for reproducible
            evaluation runs.
        client: Optional preconfigured ``httpx.Client`` (used by tests).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        temperature: float = 0.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._temperature = temperature
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._headers = headers

    @classmethod
    def from_env(cls) -> OpenAICompatibleLLM:
        """Build a backend from DOCURESEARCH_LLM_* environment variables."""
        base_url = os.environ.get("DOCURESEARCH_LLM_BASE_URL")
        model = os.environ.get("DOCURESEARCH_LLM_MODEL")
        if not base_url or not model:
            raise LLMConfigError(
                "LLM backend not configured: set DOCURESEARCH_LLM_BASE_URL and "
                "DOCURESEARCH_LLM_MODEL (and DOCURESEARCH_LLM_API_KEY if required). "
                "See .env.example."
            )
        return cls(
            base_url=base_url,
            model=model,
            api_key=os.environ.get("DOCURESEARCH_LLM_API_KEY"),
            timeout_seconds=float(os.environ.get("DOCURESEARCH_LLM_TIMEOUT", "120")),
        )

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Send *prompt* to the chat endpoint and parse the structured answer.

        Transport and parse failures are reported via ``parse_error`` with
        ``structured=None``; they are never turned into an answer here.
        """
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": kwargs.get("temperature", self._temperature),
        }
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        start = time.monotonic()
        try:
            resp = self._client.post(self._url, json=payload, headers=self._headers)
            resp.raise_for_status()
            raw_text = resp.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            return LLMResponse(
                raw_text="",
                parse_error=f"LLM request failed: {exc}",
                latency_seconds=time.monotonic() - start,
            )
        latency = time.monotonic() - start

        structured = parse_llm_output(raw_text)
        return LLMResponse(
            raw_text=raw_text,
            structured=structured,
            parse_error=None if structured else "LLM output did not match the answer schema",
            latency_seconds=latency,
        )
