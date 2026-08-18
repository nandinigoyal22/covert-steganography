"""
tests/test_llm_client.py
────────────────────────
Unit tests for models/llm_client.py.

Every test mocks httpx.Client.post — no real HTTP calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from config.settings import Settings
from models.llm_client import LLMClient, LLMClientError


# ── helpers ──────────────────────────────────────────────────────────


def _make_settings(**overrides) -> Settings:
    """Create a Settings object with safe defaults for testing."""
    defaults = {
        "OPENAI_API_KEY": "test-key-123",
        "OPENAI_BASE_URL": "https://fake.api.test/v1",
        "OPENAI_MODEL": "test-model",
        "OPENAI_TIMEOUT": 5,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _openai_response(content: str) -> httpx.Response:
    """Build a fake 200 response matching the OpenAI chat-completions shape."""
    body = {
        "choices": [
            {"message": {"role": "assistant", "content": content}}
        ]
    }
    return httpx.Response(
        status_code=200,
        json=body,
        request=httpx.Request("POST", "https://fake.api.test/v1/chat/completions"),
    )


# ── tests ────────────────────────────────────────────────────────────


class TestLLMClientChat:
    """Tests for LLMClient.chat()."""

    @patch.object(httpx.Client, "post")
    def test_chat_success(self, mock_post: MagicMock) -> None:
        """Happy path: valid response → returns content string."""
        mock_post.return_value = _openai_response("Hello from the LLM!")

        client = LLMClient(_make_settings())
        result = client.chat("Say hello")

        assert result == "Hello from the LLM!"
        mock_post.assert_called_once()

        # Verify the payload structure
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "test-model"
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    @patch.object(httpx.Client, "post")
    def test_chat_with_system_prompt(self, mock_post: MagicMock) -> None:
        """When system_prompt is provided, messages list has 2 entries."""
        mock_post.return_value = _openai_response("OK")

        client = LLMClient(_make_settings())
        client.chat("Do something", system_prompt="You are a helpful bot.")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        messages = payload["messages"]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are a helpful bot."}
        assert messages[1] == {"role": "user", "content": "Do something"}

    @patch.object(httpx.Client, "post")
    def test_chat_http_error(self, mock_post: MagicMock) -> None:
        """Non-2xx status → LLMClientError."""
        mock_post.return_value = httpx.Response(
            status_code=500,
            text="Internal Server Error",
            request=httpx.Request("POST", "https://fake.api.test/v1/chat/completions"),
        )

        client = LLMClient(_make_settings())
        with pytest.raises(LLMClientError, match="500"):
            client.chat("This should fail")

    @patch.object(httpx.Client, "post")
    def test_chat_malformed_json(self, mock_post: MagicMock) -> None:
        """200 with a body missing 'choices' → LLMClientError."""
        mock_post.return_value = httpx.Response(
            status_code=200,
            json={"unexpected": "shape"},
            request=httpx.Request("POST", "https://fake.api.test/v1/chat/completions"),
        )

        client = LLMClient(_make_settings())
        with pytest.raises(LLMClientError):
            client.chat("Bad response body")
