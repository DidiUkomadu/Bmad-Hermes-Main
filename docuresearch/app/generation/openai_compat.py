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

import logging
import os
import time
from datetime import datetime
from typing import Any

import httpx

from app.generation import allowance as allowance_clock
from app.generation.allowance import ProviderAllowance, next_utc_midnight, parse_reset
from app.generation.interface import LLMResponse
from app.generation.schema import parse_llm_output

_SYSTEM_MESSAGE = (
    "You are a careful research assistant. You answer only from the source "
    "passages you are given and you always respond with a single JSON object."
)


_MAX_RETRY_DELAY_SECONDS = 30.0

# OpenRouter's wording for its free daily allowance being used up.
_DAILY_CAP_TEXT = "free-models-per-day"

logger = logging.getLogger(__name__)


class ProviderError(httpx.HTTPError):
    """An error reported in the response body rather than the HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"provider error {status}: {message}")
        self.status = status


def _error_body(resp: httpx.Response) -> dict[str, Any] | None:
    try:
        body = resp.json()
    except ValueError:
        return None
    if isinstance(body, dict) and isinstance(body.get("error"), dict) and "choices" not in body:
        return body["error"]
    return None


def _effective_status(resp: httpx.Response) -> int:
    """HTTP status, or the provider's embedded error code for a 200 error body."""
    if resp.status_code != 200:
        return resp.status_code
    error = _error_body(resp)
    if error is None:
        return 200
    try:
        return int(error.get("code"))
    except (TypeError, ValueError):
        return 502  # unrecognised provider error: treat as a retryable upstream failure


def _provider_message(resp: httpx.Response) -> str:
    error = _error_body(resp) or {}
    metadata = error.get("metadata")
    raw = metadata.get("raw") if isinstance(metadata, dict) else None
    return str(raw or error.get("message") or "unknown error")[:300]


class DailyCapError(ProviderError):
    """The provider's daily allowance is used up until *until* (Story 9.2)."""

    def __init__(self, until: datetime, message: str) -> None:
        super().__init__(429, message)
        self.until = until


def _rate_limit_headers(resp: httpx.Response) -> dict[str, str]:
    """Rate-limit headers from the HTTP response and the body's ``error.metadata.headers``.

    Keys are lower-cased; HTTP headers win over the body's copy.
    """
    headers: dict[str, str] = {}
    error = _error_body(resp) or {}
    metadata = error.get("metadata")
    body_headers = metadata.get("headers") if isinstance(metadata, dict) else None
    if isinstance(body_headers, dict):
        headers.update({str(k).lower(): str(v) for k, v in body_headers.items()})
    for name in ("x-ratelimit-remaining", "x-ratelimit-reset"):
        if name in resp.headers:
            headers[name] = resp.headers[name]
    return headers


def _is_zero(value: str | None) -> bool:
    try:
        return value is not None and float(value) == 0
    except ValueError:
        return False


def _daily_cap_until(resp: httpx.Response, now: datetime) -> datetime | None:
    """For a 429: when the daily allowance resets, or None for an ordinary busy 429.

    Daily when the body says ``free-models-per-day`` (reset falls back to the
    next 00:00 UTC), or when ``X-RateLimit-Remaining`` is 0 together with a
    usable ``X-RateLimit-Reset``.
    """
    headers = _rate_limit_headers(resp)
    reset = parse_reset(headers.get("x-ratelimit-reset"), now)
    if _DAILY_CAP_TEXT in resp.text:
        return reset or next_utc_midnight(now)
    if _is_zero(headers.get("x-ratelimit-remaining")) and reset is not None:
        return reset
    return None


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    try:
        return max(0.0, float(resp.headers["retry-after"]))
    except (KeyError, ValueError):
        return None


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
        max_retries: Retries for 429 and 5xx responses (never for a daily-cap 429).
        backoff_seconds: First retry delay; doubles on each retry.

    Attributes:
        allowance: In-memory record of the provider's daily allowance being
            used up (Story 9.2). While exhausted, ``generate`` makes no call.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        temperature: float = 0.0,
        client: httpx.Client | None = None,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._temperature = temperature
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._headers = headers
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self.allowance = ProviderAllowance()

    def _post_with_retry(self, payload: dict[str, Any]) -> httpx.Response:
        """POST, retrying rate limits (429) and server errors (5xx) with backoff.

        Free tiers are often briefly rate-limited upstream, and home networks
        drop DNS now and then; a few short retries keep one bad moment from
        failing a request. Network errors (``httpx.TransportError``) are retried
        too. Honors ``Retry-After`` (capped). Other errors are raised immediately.

        Some gateways (e.g. OpenRouter) report provider errors inside a 200
        response as ``{"error": {"code": ..., "message": ...}}``; those are
        treated by their embedded code.

        A daily-cap 429 (see ``_daily_cap_until``) is never retried: it raises
        ``DailyCapError`` at once.
        """
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.post(self._url, json=payload, headers=self._headers)
            except httpx.TransportError:
                # Network blip (DNS failure, refused connection, timeout): retry.
                if attempt == self._max_retries:
                    raise
                time.sleep(min(self._backoff_seconds * 2**attempt, _MAX_RETRY_DELAY_SECONDS))
                continue
            status = _effective_status(resp)
            if status == 429:
                until = _daily_cap_until(resp, allowance_clock._now())
                if until is not None:
                    raise DailyCapError(until, _provider_message(resp))
            retryable = status == 429 or status >= 500
            if not retryable or attempt == self._max_retries:
                resp.raise_for_status()
                if status != resp.status_code:
                    raise ProviderError(status, _provider_message(resp))
                return resp
            delay = _retry_after_seconds(resp) or self._backoff_seconds * 2**attempt
            time.sleep(min(delay, _MAX_RETRY_DELAY_SECONDS))
        raise AssertionError("unreachable")

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._client.close()

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

        While the provider's daily allowance is exhausted no request is sent;
        that, and the request that discovers it, are reported with
        ``provider_exhausted_until`` set as well.
        """
        exhausted_until = self.allowance.exhausted_until()
        if exhausted_until is not None:
            return _exhausted_response(exhausted_until, 0.0)

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
            resp = self._post_with_retry(payload)
            raw_text = resp.json()["choices"][0]["message"]["content"] or ""
        except DailyCapError as exc:
            self.allowance.mark_exhausted(exc.until)
            logger.warning(
                "LLM provider daily allowance exhausted; no calls until %s",
                exc.until.isoformat(),
            )
            return _exhausted_response(exc.until, time.monotonic() - start)
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


def _exhausted_response(until: datetime, latency: float) -> LLMResponse:
    return LLMResponse(
        raw_text="",
        parse_error=f"LLM provider daily allowance exhausted until {until.isoformat()}",
        latency_seconds=latency,
        provider_exhausted_until=until,
    )
