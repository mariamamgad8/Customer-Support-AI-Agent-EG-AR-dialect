# voice-ai-agent (dev-ready)

This repository contains a complete recording-based voice pipeline that integrates:
- Twilio outbound calls (record-on-answer)
- Preprocessing (denoise, VAD, convert)
- ASR (Whisper small for dev)
- NLU (heuristic or optional HF pipeline)
- Dialogue Manager
- RAG retriever (SBERT) + simple LLM fallback
- TTS (ElevenLabs) and static hosting for generated audio

Quick start (local dev, ngrok recommended)
1. Clone or copy files into a folder (voice-ai-agent) on your machine or Colab drive.
2. Create a virtualenv and install requirements:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

3. Create a `.env` file in project root with:
   TWILIO_ACCOUNT_SID=...
   TWILIO_AUTH_TOKEN=...
   TWILIO_FROM_NUMBER=+1...
   BASE_URL=https://your-ngrok-or-domain.ngrok.io
   ELEVEN_API_KEY=sk_...
   (Optionally NLU_MODEL_DIR, EMBED_MODEL)

4. Start the app:
   uvicorn app:app --host 0.0.0.0 --port 5050

5. Expose port 5050 with ngrok (dev):
   ngrok http 5050
   Copy the HTTPS base url (e.g., https://abcd1234.ngrok.io) and set BASE_URL accordingly.

6. Trigger an outbound call:
   curl -X POST "{BASE_URL}/make-call" -H "Content-Type: application/json" -d '{"to":"+2012XXXXXXX","base_url":"https://abcd1234.ngrok.io"}'

7. Answer the phone: Twilio will record the caller and POST the recording to /recording-complete. The server processes the recording in background, transcribes, runs DM/RAG, generates TTS, and saves an mp3 under /static. You can check logs.

Notes & production guidance
- For production and low-latency streaming use Twilio Media Streams instead of recording-on-answer. That requires more complex websocket streaming.
- Heavy LLMs (20B) cannot run in cheap machines. Use hosted inference (Hugging Face Inference, TGI, Azure OpenAI) for production or put heavy models in a separate GPU worker.
- Secure your endpoints (validate Twilio signatures) and use proper secret management.
- Use Celery/RQ and worker pool for LLM/TTS jobs rather than processing them in the request worker.