"""
models/llm_client.py
────────────────────
Thin, provider-agnostic wrapper around a single LLM chat-completion call.

Responsibilities (and nothing more):
    1. Accept a user prompt (and optional system prompt).
    2. POST to an OpenAI-compatible /chat/completions endpoint.
    3. Return the raw text of the first choice.

No parsing, no retries, no agent logic.  Those belong in higher layers.

Usage:
    from member1.config.settings import Settings
    from member1.models.llm_client import LLMClient

    client = LLMClient(Settings())
    reply = client.chat("Summarise this document …")
"""

from __future__ import annotations

import httpx

from config.settings import Settings


class LLMClientError(Exception):
    """Raised when the LLM call fails for any reason (HTTP, JSON, etc.)."""


class LLMClient:
    """Synchronous wrapper for an OpenAI-compatible chat-completions API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._http = httpx.Client(
            base_url=settings.OPENAI_BASE_URL,
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=settings.OPENAI_TIMEOUT,
        )

    # ── public API ───────────────────────────────────────────────────

    def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        """Send *prompt* to the LLM and return the raw text response.

        Args:
            prompt:        The user message.
            system_prompt: Optional system-level instruction prepended to
                           the messages list.

        Returns:
            The text content of ``choices[0].message.content``.

        Raises:
            LLMClientError: On any HTTP or response-parsing failure.
        """
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._settings.OPENAI_MODEL,
            "messages": messages,
        }

        try:
            response = self._http.post("/chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
            return body["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise LLMClientError(
                f"HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMClientError(str(exc)) from exc

    # ── cleanup ──────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()
