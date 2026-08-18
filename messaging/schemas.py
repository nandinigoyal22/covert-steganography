"""
messaging/schemas.py
────────────────────
Pydantic v2 models that define the message contracts between Member 1
(orchestration) and the two external modules:

    Member 2 (steganography):  SteganographyRequest / SteganographyResponse
    Member 3 (detection):      DetectionResult

These are *interface* schemas — the actual encode/decode and detection logic
lives in Member 2 and Member 3 respectively (mocked for now).

Design notes:
    - carrier_input and encoded_carrier are str (file paths) in v1.
      Raw-bytes support deferred until Member 2 has a real reason for it.
    - Literal types enforce valid enum values at construction time.
    - metadata / carrier_metadata are open dicts for forward-compatibility.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Steganography interface (→ Member 2) ─────────────────────────────


class SteganographyRequest(BaseModel):
    """Message sent TO Member 2 asking it to encode or decode."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    operation: Literal["encode", "decode"]
    carrier_type: Literal["image", "pdf", "docx", "text"]
    carrier_input: str                          # file path to the carrier
    message: str                                # secret to embed (encode)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SteganographyResponse(BaseModel):
    """Message received FROM Member 2 after encode/decode completes."""

    request_id: str
    conversation_id: str
    operation: Literal["encode", "decode"]
    status: Literal["success", "failure"]
    encoded_carrier: str | None = None          # file path (encode result)
    decoded_message: str | None = None          # extracted text (decode)
    carrier_metadata: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


# ── Detection interface (→ Member 3) ────────────────────────────────


class DetectionResult(BaseModel):
    """Result returned by Member 3's detection / risk-scoring module."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    carrier_type: Literal["image", "pdf", "docx", "text"]
    prediction: Literal["clean", "suspicious", "stego_detected"]
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    features_used: list[str] = Field(default_factory=list)
