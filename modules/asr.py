from dotenv import load_dotenv
load_dotenv()

import os
import requests
import logging

logger = logging.getLogger("asr")

AZURE_WHISPER_ENDPOINT = os.getenv("AZURE_WHISPER_ENDPOINT")
AZURE_WHISPER_API_KEY = os.getenv("AZURE_WHISPER_API_KEY")

def transcribe_file(path: str, language="ar"):
    if not AZURE_WHISPER_ENDPOINT or not AZURE_WHISPER_API_KEY:
        logger.error("Azure Whisper endpoint/API key not configured")
        return ""
    try:
        with open(path, "rb") as audio_file:
            files = {"file": audio_file}
            data = {
                "language": language,
                "response_format": "text"
            }
            headers = {
                "api-key": AZURE_WHISPER_API_KEY
            }
            resp = requests.post(AZURE_WHISPER_ENDPOINT, headers=headers, data=data, files=files)
            resp.raise_for_status()
            return resp.text.strip()
    except Exception as e:
        logger.exception("Azure Whisper transcription failed: %s", e)
        return ""
def transcribe_bytes(audio_bytes: bytes, language="ar"):
    if not AZURE_WHISPER_ENDPOINT or not AZURE_WHISPER_API_KEY:
        logger.error("Azure Whisper endpoint/API key not configured")
        return ""
    try:
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data = {
            "language": language,
            "response_format": "text"
        }
        headers = {
            "api-key": AZURE_WHISPER_API_KEY
        }
        resp = requests.post(AZURE_WHISPER_ENDPOINT, headers=headers, data=data, files=files)
        resp.raise_for_status()
        return resp.text.strip()
    except Exception as e:
        logger.exception("Azure Whisper transcription failed: %s", e)
        return ""