"""
tests/test_schemas.py
─────────────────────
Unit tests for messaging/schemas.py and state/conversation_state.py.

Tests valid construction, invalid-value rejection (Literal / range),
and JSON serialisation round-trips.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from messaging.schemas import (
    DetectionResult,
    SteganographyRequest,
    SteganographyResponse,
)
from state.conversation_state import ConversationState


# ── SteganographyRequest ─────────────────────────────────────────────


class TestSteganographyRequest:
    def test_valid_construction(self) -> None:
        req = SteganographyRequest(
            conversation_id="conv-1",
            operation="encode",
            carrier_type="image",
            carrier_input="/path/to/image.png",
            message="secret message",
        )
        assert req.operation == "encode"
        assert req.carrier_type == "image"
        assert req.carrier_input == "/path/to/image.png"
        assert req.metadata == {}            # default
        assert req.request_id               # auto-generated, non-empty

    def test_invalid_operation(self) -> None:
        with pytest.raises(ValidationError, match="operation"):
            SteganographyRequest(
                conversation_id="conv-1",
                operation="encrypt",         # not in Literal
                carrier_type="image",
                carrier_input="/path/to/img.png",
                message="secret",
            )

    def test_invalid_carrier_type(self) -> None:
        with pytest.raises(ValidationError, match="carrier_type"):
            SteganographyRequest(
                conversation_id="conv-1",
                operation="encode",
                carrier_type="mp3",          # not in Literal
                carrier_input="/path/to/file.mp3",
                message="secret",
            )


# ── SteganographyResponse ───────────────────────────────────────────


class TestSteganographyResponse:
    def test_valid_construction(self) -> None:
        resp = SteganographyResponse(
            request_id="req-1",
            conversation_id="conv-1",
            operation="encode",
            status="success",
            encoded_carrier="/path/to/output.png",
            carrier_metadata={"format": "png", "size_bytes": 1024},
        )
        assert resp.status == "success"
        assert resp.errors == []             # default

    def test_invalid_status(self) -> None:
        with pytest.raises(ValidationError, match="status"):
            SteganographyResponse(
                request_id="req-1",
                conversation_id="conv-1",
                operation="decode",
                status="partial",            # not in Literal
            )


# ── DetectionResult ──────────────────────────────────────────────────


class TestDetectionResult:
    def test_valid_construction(self) -> None:
        result = DetectionResult(
            carrier_type="text",
            prediction="clean",
            risk_score=0.1,
            confidence=0.95,
            features_used=["lsb_analysis", "chi_square"],
        )
        assert result.prediction == "clean"
        assert result.risk_score == 0.1

    def test_risk_score_too_high(self) -> None:
        with pytest.raises(ValidationError, match="risk_score"):
            DetectionResult(
                carrier_type="image",
                prediction="suspicious",
                risk_score=1.5,              # > 1.0
                confidence=0.8,
            )

    def test_risk_score_too_low(self) -> None:
        with pytest.raises(ValidationError, match="risk_score"):
            DetectionResult(
                carrier_type="image",
                prediction="suspicious",
                risk_score=-0.1,             # < 0.0
                confidence=0.8,
            )

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValidationError, match="confidence"):
            DetectionResult(
                carrier_type="pdf",
                prediction="stego_detected",
                risk_score=0.9,
                confidence=2.0,              # > 1.0
            )


# ── ConversationState ────────────────────────────────────────────────


class TestConversationState:
    def test_valid_construction(self) -> None:
        detection = DetectionResult(
            carrier_type="docx",
            prediction="suspicious",
            risk_score=0.7,
            confidence=0.85,
            features_used=["entropy_check"],
        )
        state = ConversationState(
            agent_id="sender",
            task="embed secret in carrier",
            message="hello world",
            carrier="/path/to/doc.docx",
            detection_result=detection,
            status="encoding",
        )
        assert state.agent_id == "sender"
        assert state.status == "encoding"
        assert state.detection_result is not None
        assert state.detection_result.risk_score == 0.7

    def test_minimal_construction(self) -> None:
        """Only required fields — defaults should fill the rest."""
        state = ConversationState(
            agent_id="orchestrator",
            task="coordinate exchange",
        )
        assert state.status == "initialized"
        assert state.message is None
        assert state.carrier is None
        assert state.detection_result is None
        assert state.carrier_metadata == {}
        assert state.conversation_id           # auto-generated

    def test_invalid_status(self) -> None:
        with pytest.raises(ValidationError, match="status"):
            ConversationState(
                agent_id="receiver",
                task="decode carrier",
                status="paused",             # not in Literal
            )


# ── Serialisation round-trip ─────────────────────────────────────────


class TestSerialisationRoundtrip:
    def test_steg_request_roundtrip(self) -> None:
        original = SteganographyRequest(
            conversation_id="conv-42",
            operation="decode",
            carrier_type="pdf",
            carrier_input="/data/file.pdf",
            message="",
            metadata={"source": "email"},
        )
        json_str = original.model_dump_json()
        restored = SteganographyRequest.model_validate_json(json_str)
        assert restored == original

    def test_conversation_state_roundtrip(self) -> None:
        detection = DetectionResult(
            carrier_type="text",
            prediction="stego_detected",
            risk_score=0.92,
            confidence=0.99,
            features_used=["lsb_analysis"],
        )
        original = ConversationState(
            agent_id="sender",
            task="full pipeline test",
            message="top secret",
            carrier="/tmp/carrier.txt",
            detection_result=detection,
            status="detected",
        )
        json_str = original.model_dump_json()
        restored = ConversationState.model_validate_json(json_str)
        assert restored == original
