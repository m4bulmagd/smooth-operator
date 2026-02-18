from xml.sax.saxutils import quoteattr


def connect_stream_twiml(stream_url: str) -> str:
    safe_url = quoteattr(stream_url)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Hi! Go ahead, I'm transcribing.</Say>
  <Connect>
    <Stream url={safe_url} />
  </Connect>
</Response>"""
