# modules/tts.py
# ElevenLabs TTS helper (returns mp3 bytes); fallback to Twilio <Say> if no key
from elevenlabs import ElevenLabs
import logging

logger = logging.getLogger("tts")


def elevenlabs_tts_bytes(text: str, api_key: str = None, voice_id: str = None):
    """
    Returns mp3 bytes. If ElevenLabs key missing raises exception.
    """
    if not api_key:
        raise RuntimeError("ELEVEN_API_KEY missing")
    client = ElevenLabs(api_key=api_key)
    voice_id = voice_id or "LXrTqFIgiubkrMkwvOUr"
    # call the SDK; different SDK versions may return bytes or an iterable of chunks
    stream = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    # If the SDK returns raw bytes, return directly; otherwise join chunks
    if isinstance(stream, (bytes, bytearray)):
        return bytes(stream)
    try:
        buf = b"".join(chunk for chunk in stream)
        return buf
    except TypeError:
        # Not iterable of bytes — attempt to read as file-like
        try:
            data = stream.read()
            return data if isinstance(data, (bytes, bytearray)) else str(data).encode()
        except Exception:
            logger.exception("Failed to convert ElevenLabs stream to bytes")
            raise


def save_bytes_to_file(b: bytes, path: str):
    with open(path, "wb") as f:
        f.write(b)
