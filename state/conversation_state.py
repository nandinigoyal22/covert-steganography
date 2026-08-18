"""
state/conversation_state.py
───────────────────────────
Tracks the lifecycle of a single sender → receiver conversation.

Every field is explicit and individually typed — no opaque blobs.
The `detection_result` field embeds the DetectionResult schema directly
so downstream code gets full type-safety without a second lookup.

Status progression (happy path):
    initialized → encoding → encoded → transmitting
        → decoding → decoded → detected → completed

Any step can transition to 'error'.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from messaging.schemas import DetectionResult


class ConversationState(BaseModel):
    """Mutable state object for one end-to-end steganographic exchange."""

    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str                               # "sender" | "receiver" | "orchestrator"
    task: str                                    # human-readable description
    message: str | None = None                  # current plaintext message
    carrier: str | None = None                  # file path to current carrier
    carrier_metadata: dict[str, Any] = Field(default_factory=dict)
    detection_result: DetectionResult | None = None
    status: Literal[
        "initialized",
        "encoding",
        "encoded",
        "transmitting",
        "decoding",
        "decoded",
        "detected",
        "completed",
        "error",
    ] = "initialized"
