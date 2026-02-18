import pytest
from pydantic import ValidationError

from app.schemas.twilio_stream import TwilioWsEvent


class TestTwilioWsEvent:
    def test_connected_event(self):
        raw = '{"event": "connected", "connected": {"callSid": "CA123"}}'
        event = TwilioWsEvent.model_validate_json(raw)
        assert event.event == "connected"
        assert event.connected is not None
        assert event.connected.callSid == "CA123"

    def test_start_event(self):
        raw = '{"event": "start", "start": {"callSid": "CA123", "streamSid": "MZ456"}}'
        event = TwilioWsEvent.model_validate_json(raw)
        assert event.event == "start"
        assert event.start is not None
        assert event.start.callSid == "CA123"

    def test_media_event(self):
        raw = '{"event": "media", "media": {"payload": "dGVzdA=="}}'
        event = TwilioWsEvent.model_validate_json(raw)
        assert event.event == "media"
        assert event.media is not None
        assert event.media.payload == "dGVzdA=="

    def test_stop_event(self):
        raw = '{"event": "stop", "stop": {"callSid": "CA123"}}'
        event = TwilioWsEvent.model_validate_json(raw)
        assert event.event == "stop"

    def test_invalid_event_type_rejected(self):
        raw = '{"event": "invalid_type"}'
        with pytest.raises(ValidationError):
            TwilioWsEvent.model_validate_json(raw)

    def test_missing_event_field_rejected(self):
        raw = '{"media": {"payload": "dGVzdA=="}}'
        with pytest.raises(ValidationError):
            TwilioWsEvent.model_validate_json(raw)

    def test_event_with_no_payload(self):
        """connected/start/stop events without their sub-objects should still parse."""
        raw = '{"event": "connected"}'
        event = TwilioWsEvent.model_validate_json(raw)
        assert event.event == "connected"
        assert event.connected is None

    def test_media_event_without_media_payload(self):
        """media event without media body — should parse (all optional)."""
        raw = '{"event": "media"}'
        event = TwilioWsEvent.model_validate_json(raw)
        assert event.event == "media"
        assert event.media is None
