def connect_stream_twiml(stream_url: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Hi! Go ahead, I'm transcribing.</Say>
  <Connect>
    <Stream url="{stream_url}" />
  </Connect>
</Response>"""
