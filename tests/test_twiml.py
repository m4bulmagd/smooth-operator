import xml.etree.ElementTree as ET

from app.services.twilio.twiml import connect_stream_twiml


class TestConnectStreamTwiml:
    def test_valid_xml_output(self):
        xml = connect_stream_twiml("wss://example.com/stream")
        # Should parse without error
        root = ET.fromstring(xml)
        assert root.tag == "Response"

    def test_stream_url_present(self):
        xml = connect_stream_twiml("wss://example.com/stream")
        root = ET.fromstring(xml)
        stream = root.find(".//Stream")
        assert stream is not None
        assert stream.attrib["url"] == "wss://example.com/stream"

    def test_say_element_present(self):
        xml = connect_stream_twiml("wss://example.com/stream")
        root = ET.fromstring(xml)
        say = root.find(".//Say")
        assert say is not None
        assert say.text is not None and len(say.text) > 0

    def test_special_chars_escaped(self):
        """URLs with special XML chars should produce valid XML."""
        url = "wss://example.com/stream?foo=1&bar=2"
        xml = connect_stream_twiml(url)
        # Must parse without error — & would break raw interpolation
        root = ET.fromstring(xml)
        stream = root.find(".//Stream")
        assert stream is not None
        assert stream.attrib["url"] == url
