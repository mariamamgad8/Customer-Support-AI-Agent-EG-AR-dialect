# Voice AI Agent — README

A Twilio-connected voice AI agent with a local web UI and dashboard.  
It records calls, preprocesses audio (channel selection, resampling, VAD), transcribes Arabic speech (Azure Whisper or local), predicts intent (heuristic or optional NLU model), runs a dialogue manager, optionally enriches answers with RAG + LLM, and returns a spoken reply (ElevenLabs TTS). A dashboard and debug recordings are provided for inspection.

This README explains what the project contains and how to run it locally on a Windows PC (PowerShell) or any platform. Copy & paste the commands/sections that apply to you.

---

Table of contents
- Features
- Repo layout
- Prerequisites
- Environment variables (.env / Secrets)
- Install & run locally (Windows / Linux / macOS)
- Expose to Twilio (ngrok)
- How to test (phone call and web UI)
- Debugging & troubleshooting (common errors you saw)
- Notes & next steps

---

Features (short)
- FastAPI backend (Twilio webhooks and web UI)
- Dual-channel Twilio recording support + automatic channel selection
- Preprocessing: ffmpeg/pydub, resample to 16k, VAD with webrtcvad, optional denoising
- ASR: pluggable (Azure Whisper by default; helper supports transcribe_file and transcribe_bytes)
- NLU: heuristic intent detection (default) + optional HF model loader (AraBERT fine-tune)
- Dialogue manager for quick replies + LLM (GPT‑OSS / configurable) for richer replies
- TTS: ElevenLabs (optional) fallback to TwiML <Say>
- Web UI: index page to make test calls & upload audio
- Dashboard: view conversation history and debug audio files
- Saves debug recordings to debug_recordings/

---

Repo layout (key files)
- app.py — main FastAPI app (endpoints: `/`, `/dashboard`, `/make-call`, `/voice/incoming`, `/voice/process`, `/process-audio-direct`, etc.)
- static/ — UI files: `index.html`, `index.css`, `dashboard.html`, `dashboard.js`, `dashboard.css`
- modules/
  - preprocess.py — audio fetch/convert/VAD/channel extraction helpers
  - asr.py — ASR wrapper (Azure Whisper by default)
  - nlu.py — heuristic NLU + optional HF pipeline loader
  - dialogue_manager.py — simple DM logic
  - llm_response_generator.py — wrapper to call configured LLM
  - tts.py — ElevenLabs TTS helper
  - twilio_utils.py — Twilio client wrapper
- debug_recordings/ — saved raw / preprocessed .wav files (created at runtime)
- logs/voice-agent.log — runtime logs (created at runtime)
- README.md — this document
- .env — sample env file (DO NOT commit secrets)

---

Prerequisites

1. Python 3.10+ (3.11/3.13 OK) installed
2. Git
3. ffmpeg available on PATH (recommended for robust conversions)
4. Internet access (for Twilio, Azure Whisper, ElevenLabs, or external LLM)
5. Twilio account with a phone number (for calling/testing)
6. Optional: ElevenLabs account for TTS; Azure / hosted LLM endpoints if used

Python packages (examples — actual `requirements.txt` may vary):
- fastapi
- uvicorn[standard]
- requests
- python-dotenv
- twilio
- pydub
- soundfile
- librosa
- webrtcvad
- numpy
- sentence-transformers
- transformers
- elevenlabs (or official SDK used)
- noisereduce (optional)

---

Environment variables (important)
Create a `.env` file in project root or set these as OS environment variables. IMPORTANT: store raw values without surrounding quotes.

Minimum for local testing (example):
```
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
TWILIO_FROM_NUMBER=+1XXXXXXXXXX

BASE_URL=http://<your-public-url>   # for Twilio use public HTTPS (ngrok) — see below

# Azure Whisper (optional)
AZURE_WHISPER_ENDPOINT=https://your-azure-whisper-endpoint
AZURE_WHISPER_API_KEY=xxxxxxxx

# LLM / RAG / TTS (optional)
GPT_OSS_API_URL=...
GPT_OSS_API_KEY=...
ELEVEN_API_KEY=sk_xxx
ELEVEN_VOICE_ID=LXrTqFIgiubkrMkwvOUr
```

Notes:
- On Windows PowerShell set variables with: $env:TWILIO_ACCOUNT_SID="AC..."
- When using HF Spaces or cloud, set secrets through the hosting UI (no quotes).
- If you see Twilio 401, check for accidental surrounding quotes in env values and remove them.

---

Install & run locally (Windows PowerShell)

1) Clone and open project
```
git clone <repo-url>
cd Voice_AI_Agent
```

2) Create and activate virtualenv
PowerShell:
```
python -m venv .venv
.venv\Scripts\Activate.ps1
```
cmd:
```
.venv\Scripts\activate
```
Linux/macOS:
```
python3 -m venv .venv
source .venv/bin/activate
```

3) Install requirements
(If repo includes `requirements.txt`):
```
pip install -r requirements.txt
```
Or install common packages:
```
pip install fastapi uvicorn requests python-dotenv twilio pydub soundfile librosa webrtcvad numpy sentence-transformers transformers elevenlabs
```

4) Install ffmpeg (required for conversion)
- Windows: download from https://ffmpeg.org/download.html and add ffmpeg to PATH
- Ubuntu: sudo apt install ffmpeg
- macOS (Homebrew): brew install ffmpeg

5) Create `.env` with secrets (see above). Do NOT commit.

6) Run the app
Option A — with uvicorn (preferred):
```
uvicorn app:app --reload --port 5050
```
Option B — if `app.py` has runner logic:
```
python app.py
```

Open browser UI:
- http://localhost:5050 — main UI
- http://localhost:5050/dashboard — dashboard

---

Expose to Twilio (ngrok)
Twilio needs a public HTTPS URL for webhooks. For local testing use ngrok.

1) Start ngrok (after installing) forwarding port 5050:
```
ngrok http 5050
```
2) Copy the public URL displayed (e.g. `https://abcd1234.ngrok.io`)
3) Set BASE_URL environment variable to this URL (no trailing slash, no quotes).
4) In the Twilio console (phone number) set:
   - "A CALL COMES IN" → Webhook → HTTP POST → `https://abcd1234.ngrok.io/voice/incoming`

Alternatively, when using the UI `Make call` feature, pass the ngrok URL as `base_url`.

---

How to test

1) Web UI
- Open http://localhost:5050
- Use "Make an outbound call" — enter a phone number (E.164) and your public base_url (ngrok or deployed).
- Click "Call" — Twilio will call the destination and interact with your app.

2) Upload audio
- Use "Upload audio for quick test" to test preprocess + ASR + pipeline without Twilio.

3) Dashboard
- Open /dashboard to inspect conversation history and debug audio files (saved under debug_recordings/).

4) Manual curl test (Twilio incoming webhook)
```
curl -X POST http://localhost:5050/voice/incoming
```
You should get TwiML XML back (200 OK).

---

Troubleshooting — common errors & fixes

1) Twilio 401 Authentication Error (invalid username)
- Cause: TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN contains surrounding quotes or wrong value.
- Fix: remove quotes in `.env` or in your host's secret settings. Example:
  - Wrong: `TWILIO_ACCOUNT_SID="AC..."`
  - Right: `TWILIO_ACCOUNT_SID=AC...`

2) Twilio recording download 404 Not Found
- Cause: Twilio may POST RecordingUrl before the file is fully available, or the recording belongs to a different account.
- Fix:
  - Use retry logic (app includes a robust fetch helper to retry on 404).
  - Ensure you are using the correct Twilio credentials associated with the call.
  - Use the `recording-complete` callback flow if needed (app supports background processing).

3) Arabic / Unicode logging errors on Windows console
- Cause: Windows console encoding may not support Arabic characters -> logging errors.
- Fix:
  - Use `PYTHONIOENCODING=utf-8` in your environment or run app via `python` in terminals that support UTF-8.
  - The app includes a SafeStreamHandler to reduce crashes; also check `logs/voice-agent.log` for full messages.

4) TTS (ElevenLabs) connection timeouts
- Cause: network or API key issue.
- Fix: verify `ELEVEN_API_KEY`, try a short test script, consider disabling TTS to test the rest of the pipeline.

5) NLU model load error
- Cause: `./models/nlu` missing model files (pytorch_model.bin / safetensors).
- Fix: rely on heuristics (default) or populate `NLU_MODEL_DIR` with a compatible fine-tuned model.

6) Audio conversion issues
- Ensure ffmpeg is installed and on PATH. If ffmpeg is missing the app tries fallback conversion logic.

7) Twilio webhook unreachable
- Ensure ngrok is running (if local) and that Twilio uses HTTPS URL. For cloud deploy, ensure the app URL is public and not protected.

---

Developer notes & tips
- Keep `.env` in `.gitignore`.
- Enable debug_recordings for intermittent issues — these are saved in `debug_recordings/` and can be played locally.
- For production, use a persistent store (Redis/DB) for conversation history instead of in-memory store.
- Consider using Twilio's dual-channel recordings and choosing the caller channel to avoid re-recorded TTS.
- Add rate limits and timeouts for external LLM/ASR calls to prevent hanging requests.

---

Useful commands (summary)

PowerShell quick start:
```powershell
git clone <repo>
cd Voice_AI_Agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Set environment vars in PowerShell (example)
$env:TWILIO_ACCOUNT_SID="AC..."
$env:TWILIO_AUTH_TOKEN="..."
uvicorn app:app --reload --port 5050
```

ngrok:
```
ngrok http 5050
# set BASE_URL to the https://xxx.ngrok.io value
```

Check debug recordings:
- `dir debug_recordings` (Windows) or `ls -l debug_recordings` (Linux/Mac)
- Play chosen WAV files in your audio player.

---

