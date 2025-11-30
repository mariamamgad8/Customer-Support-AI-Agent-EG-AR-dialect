import os
from typing import Dict, Any, Optional
from twilio.rest import Client
import logging

logger = logging.getLogger("twilio_utils")

def _twilio_client() -> Client:
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    if not sid or not token:
        raise RuntimeError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set in environment")
    return Client(sid, token)

def make_call(to: str, base_url: Optional[str] = None, from_number: Optional[str] = None, twiml_path: str = "/voice/incoming") -> Dict[str, Any]:
    """
    Create an outbound call via Twilio.

    - to: destination phone number in E.164 format
    - base_url: public base URL where your app is reachable (ngrok or deployed host)
    - from_number: optional Twilio From number; if omitted it uses TWILIO_FROM_NUMBER env var
    - twiml_path: endpoint path Twilio should GET/POST to get TwiML (defaults to /voice/incoming)
    Returns Twilio call object data (dict with sid, status).
    """
    client = _twilio_client()
    from_num = from_number or os.getenv("TWILIO_FROM_NUMBER")
    if not from_num:
        raise RuntimeError("TWILIO_FROM_NUMBER must be set in environment or passed to make_call")

    if not base_url:
        base_url = os.getenv("BASE_URL")
    if not base_url:
        raise RuntimeError("base_url must be provided or set in BASE_URL env var")

    twiml_url = f"{base_url.rstrip('/')}{twiml_path}"
    logger.info("Creating Twilio call from %s to %s with twiml_url=%s", from_num, to, twiml_url)
    call = client.calls.create(
        to=to,
        from_=from_num,
        url=twiml_url,
    )
    return {"sid": getattr(call, "sid", None), "status": getattr(call, "status", None)}