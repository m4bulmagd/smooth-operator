import pytest
from app.utils.urls import https_to_wss


class TestHttpsToWss:
    def test_https_to_wss(self):
        assert https_to_wss("https://example.com") == "wss://example.com"

    def test_http_to_ws(self):
        assert https_to_wss("http://example.com") == "ws://example.com"

    def test_passthrough_no_scheme(self):
        assert https_to_wss("example.com") == "example.com"

    def test_trailing_slash_stripped(self):
        assert https_to_wss("https://example.com/") == "wss://example.com"

    def test_preserves_path(self):
        assert (
            https_to_wss("https://example.com/foo/bar") == "wss://example.com/foo/bar"
        )

    def test_empty_string(self):
        assert https_to_wss("") == ""
