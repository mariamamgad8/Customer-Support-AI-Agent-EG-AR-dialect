# app.py
"""
Complete FastAPI app (updated logging to avoid Windows console Unicode crashes).

This is your original app.py with one important change:
- Robust logging setup that avoids UnicodeEncodeError when logging Arabic text to the Windows console.
  - File handler uses encoding='utf-8'.
  - Console/stream handler is a SafeStreamHandler that catches UnicodeEncodeError and writes safely.
  - Also suggests setting PYTHONIOENCODING=utf-8 for the environment.

Everything else is left as you had it (Twilio endpoints, preprocessing, channel selection, conversation history, dashboard APIs).
"""
import os
import io
import time
import tempfile
import logging
import requests
import shutil
import webbrowser
import subprocess
import threading
from typing import Optional
from fastapi import FastAPI, Request, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import Response, FileResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client as TwilioClient
from dotenv import load_dotenv

# audio libs
import soundfile as sf
import numpy as np

# load .env
load_dotenv()

# Import your modules (must exist in modules/)
from modules.preprocess import demo_process_url_and_prepare_for_asr, fetch_audio_from_url, convert_to_wav_bytes
from modules import asr as asr_module
from modules.nlu import heuristic_intent_from_text, load_nlu_pipeline, get_predicted_intent
from modules.dialogue_manager import DialogueManager
from modules.rag import Retriever, load_or_build_retriever, rag_answer_or_fallback
from modules.tts import elevenlabs_tts_bytes
from modules.llm_response_generator import call_llm_for_response

# Config from env
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
BASE_URL = os.getenv("BASE_URL")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

# ---------------------------
# Robust logging setup
# ---------------------------
# Ensure log directory and file
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "voice-agent.log")

class SafeStreamHandler(logging.StreamHandler):
    """
    Stream handler that tolerates UnicodeEncodeError on Windows consoles.
    It will write using the stream; if encoding fails it will re-encode with utf-8 and replace errors.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
                stream.flush()
            except UnicodeEncodeError:
                # fallback: encode to UTF-8 bytes and decode back replacing errors to allow write
                safe = msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
                stream.write(safe + self.terminator)
                stream.flush()
        except Exception:
            self.handleError(record)

# Configure root logger
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

# File handler (UTF-8)
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Safe console handler
stream_handler = SafeStreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Also ensure the global logging.basicConfig doesn't add duplicate handlers in some environments
logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])

# Create app
app = FastAPI()

# Mount static files
static_dir = os.path.join(os.getcwd(), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Debug recordings folder
debug_dir = os.path.join(os.getcwd(), "debug_recordings")
os.makedirs(debug_dir, exist_ok=True)

# ---------------------------------------------------------
# Setup models & logic
# ---------------------------------------------------------
nlu_pipeline = load_nlu_pipeline()
if nlu_pipeline:
    logger.info("NLU pipeline loaded.")
else:
    logger.info("NLU pipeline not loaded; using heuristics.")

retriever = None
try:
    retriever = load_or_build_retriever()
    logger.info("Retriever ready.")
except Exception as e:
    logger.warning(f"Retriever not ready: {e}")

dialogue_manager = DialogueManager()

# ---------------------------------------------------------
# In-memory conversation store (thread-safe)
# ---------------------------------------------------------
conversation_store = {}
conversation_lock = threading.Lock()

def append_conversation(call_sid: str, role: str, text: str, meta: dict = None):
    if meta is None:
        meta = {}
    with conversation_lock:
        if call_sid not in conversation_store:
            conversation_store[call_sid] = []
        conversation_store[call_sid].append({"role": role, "text": text, "meta": meta, "ts": time.time()})

def get_conversation(call_sid: str):
    with conversation_lock:
        return list(conversation_store.get(call_sid, []))

def clear_conversation(call_sid: str):
    with conversation_lock:
        if call_sid in conversation_store:
            del conversation_store[call_sid]

# ---------------------------------------------------------
# Channel selection helper (split stereo and pick best channel)
# ---------------------------------------------------------
def _normalize_to_float(arr):
    if arr.dtype.kind in ("i", "u"):
        return arr.astype(np.float32) / 32768.0
    return arr.astype(np.float32)

def choose_best_channel_and_get_mono_bytes(wav_bytes: bytes, language: str = "ar"):
    """
    If wav_bytes is multi-channel, split channels and pick best channel for ASR.
    Strategy:
      - If mono: return original.
      - Compute RMS per channel: if one channel is silent choose louder channel.
      - Otherwise call ASR.transcribe_bytes on each channel and pick the one with longest transcript
        (boost if contains Arabic letters).
    Saves debug channel files under debug_recordings/.
    Returns: chosen_mono_wav_bytes, chosen_channel_index (or None if mono), transcripts_dict
    """
    transcripts = {}
    try:
        f = io.BytesIO(wav_bytes)
        data, sr = sf.read(f, dtype="int16")
    except Exception as e:
        logger.exception("choose_best_channel: soundfile read failed; returning original bytes: %s", e)
        return wav_bytes, None, transcripts

    # mono
    if data.ndim == 1:
        try:
            path = os.path.join(debug_dir, f"mono_{int(time.time())}.wav")
            with open(path, "wb") as fh:
                fh.write(wav_bytes)
            logger.info(f"Saved debug mono -> {path}")
        except Exception:
            pass
        return wav_bytes, None, transcripts

    n_ch = data.shape[1]
    logger.info(f"choose_best_channel: input has {n_ch} channels, sr={sr}")

    # RMS per channel
    rms = []
    for ch in range(n_ch):
        chf = _normalize_to_float(data[:, ch])
        rms_val = float(np.sqrt(np.mean(chf ** 2))) if chf.size > 0 else 0.0
        rms.append(rms_val)
    logger.info(f"Channel RMS values: {rms}")

    max_rms = max(rms)
    if max_rms > 0:
        # quick pick if channel very quiet relative to max
        for idx, r in enumerate(rms):
            if r < 0.01 and max_rms > 0.05:
                chosen = int(np.argmax(rms))
                logger.info(f"choose_best_channel: quick pick channel {chosen} (other channels mostly silent)")
                buf = io.BytesIO()
                sf.write(buf, data[:, chosen], sr, format="WAV", subtype="PCM_16")
                chosen_bytes = buf.getvalue()
                try:
                    path_ch = os.path.join(debug_dir, f"chosen_quick_ch{chosen}_{int(time.time())}.wav")
                    with open(path_ch, "wb") as fh:
                        fh.write(chosen_bytes)
                    logger.info(f"Saved chosen_quick -> {path_ch}")
                except Exception:
                    pass
                return chosen_bytes, chosen, transcripts

    # Otherwise transcribe each channel
    best_idx = None
    best_score = -1
    for ch in range(n_ch):
        ch_arr = data[:, ch]
        buf = io.BytesIO()
        sf.write(buf, ch_arr, sr, format="WAV", subtype="PCM_16")
        ch_bytes = buf.getvalue()

        # save debug channel file
        try:
            ch_path = os.path.join(debug_dir, f"ch{ch}_{int(time.time())}.wav")
            with open(ch_path, "wb") as fh:
                fh.write(ch_bytes)
            logger.info(f"Saved debug channel file -> {ch_path}")
        except Exception:
            pass

        # call ASR on channel bytes
        try:
            tr = asr_module.transcribe_bytes(ch_bytes, language=language) or ""
        except Exception as e:
            logger.warning("ASR failed on channel %d: %s", ch, e)
            tr = ""
        transcripts[ch] = tr

        score = len(tr.strip())
        if any("\u0600" <= c <= "\u06FF" for c in tr):
            score += 5
        score += int(rms[ch] * 100)
        if score > best_score:
            best_score = score
            best_idx = ch

    chosen_idx = int(best_idx) if best_idx is not None else 0
    logger.info(f"choose_best_channel: selected channel {chosen_idx} with score {best_score} transcripts={ {k:(v[:60]+'...' if v else '') for k,v in transcripts.items()} }")

    buf_ch = io.BytesIO()
    sf.write(buf_ch, data[:, chosen_idx], sr, format="WAV", subtype="PCM_16")
    chosen_bytes = buf_ch.getvalue()
    try:
        chosen_path = os.path.join(debug_dir, f"chosen_ch{chosen_idx}_{int(time.time())}.wav")
        with open(chosen_path, "wb") as fh:
            fh.write(chosen_bytes)
        logger.info(f"Saved chosen channel wav -> {chosen_path}")
    except Exception:
        pass

    return chosen_bytes, chosen_idx, transcripts

# ---------------------------------------------------------
# Core pipeline (shared)
# ---------------------------------------------------------
def core_voice_pipeline(wav_path: str, call_sid: str = "web"):
    """
    Shared pipeline: wav file -> ASR -> NLU -> DM -> RAG+LLM -> TTS
    Returns dict with transcript, intent, response_text, audio_url, rag_hits
    """
    # 1. ASR
    transcript = asr_module.transcribe_file(wav_path)
    logger.info(f"[{call_sid}] Transcript (raw): {transcript!r}")

    # Append user to history
    append_conversation(call_sid, "user", transcript, meta={"source": "asr"})

    # Early exit if empty
    if not transcript or not transcript.strip():
        final_text = "Could you please repeat? I did not hear you clearly."
        append_conversation(call_sid, "assistant", final_text, meta={"source": "system_fallback"})
        logger.info(f"[{call_sid}] Empty transcript — asking user to repeat.")
        return {
            "transcript": "",
            "intent": "other",
            "nlu_raw": None,
            "response_text": final_text,
            "audio_url": None,
            "rag_hits": [],
        }

    # 2. NLU
    if nlu_pipeline:
        try:
            nlu_res = nlu_pipeline(transcript)
            logger.debug(f"[{call_sid}] NLU raw output: {nlu_res}")
            intent = get_predicted_intent(nlu_res)
            logger.info(f"[{call_sid}] Predicted intent (from model): {intent}")
        except Exception as e:
            logger.exception(f"[{call_sid}] NLU pipeline failed, falling back to heuristics: {e}")
            nlu_res = None
            intent = heuristic_intent_from_text(transcript)
            logger.info(f"[{call_sid}] Predicted intent (heuristic fallback): {intent}")
    else:
        nlu_res = None
        intent = heuristic_intent_from_text(transcript)
        logger.info(f"[{call_sid}] Predicted intent (heuristic): {intent}")

    # 3. DM (short reply)
    dm_output = dialogue_manager.handle_intent(intent, transcript)
    first_reply = dm_output.get("response", "")
    need_llm = dm_output.get("need_llm", False)
    logger.info(f"[{call_sid}] DM reply: {first_reply!r} (need_llm={need_llm})")
    append_conversation(call_sid, "assistant", first_reply, meta={"source": "dm_short"})

    # 4. RAG / LLM
    final_text = first_reply
    rag_hits = []
    if retriever is not None:
        try:
            query = transcript
            top_k = int(os.getenv("RAG_TOP_K", "3"))
            rag_hits = retriever.retrieve(query, top_k=top_k)
            logger.info(f"[{call_sid}] RAG hits count: {len(rag_hits)}")
            logger.debug(f"[{call_sid}] RAG hits: {rag_hits}")
        except Exception as e:
            logger.warning(f"[{call_sid}] RAG retrieval failed: {e}")

    # Build history and call LLM
    history = get_conversation(call_sid)
    try:
        llm_response = call_llm_for_response(history=history, rag_hits=rag_hits, call_sid=call_sid, intent=intent)
    except Exception as e:
        logger.exception(f"[{call_sid}] LLM call failed: {e}")
        llm_response = None

    logger.info(f"[{call_sid}] LLM response (raw): {llm_response!r}")
    if llm_response:
        final_text = llm_response
        append_conversation(call_sid, "assistant", final_text, meta={"source": "llm"})
        logger.info(f"[{call_sid}] Using LLM-enhanced response")
    elif need_llm and retriever is not None:
        try:
            final_text = rag_answer_or_fallback(query, retriever, top_k=top_k)
            append_conversation(call_sid, "assistant", final_text, meta={"source": "rag_fallback"})
            logger.info(f"[{call_sid}] Fallback RAG answer: {final_text!r}")
        except Exception as e:
            logger.warning(f"[{call_sid}] RAG fallback failed: {e}")

    logger.info(f"[{call_sid}] Final Response: {final_text!r}")

    # 5. TTS (ElevenLabs) - already wrapped with try/except
    mp3_bytes = None
    public_url = None
    if ELEVEN_API_KEY and final_text:
        try:
            # elevenlabs_tts_bytes should internally set timeouts; if not, this may hang on network issues
            mp3_bytes = elevenlabs_tts_bytes(final_text, api_key=ELEVEN_API_KEY)
            if mp3_bytes:
                out_filename = f"response_{call_sid}_{os.urandom(4).hex()}.mp3"
                out_path = os.path.join(static_dir, out_filename)
                with open(out_path, "wb") as f:
                    f.write(mp3_bytes)
                public_url = f"/static/{out_filename}"
        except Exception as e:
            logger.warning("TTS failed: %s", e)

    return {
        "transcript": transcript,
        "intent": intent,
        "nlu_raw": nlu_res,
        "response_text": final_text,
        "audio_url": public_url,
        "rag_hits": rag_hits,
    }

# ---------------------------------------------------------
# Twilio helper to create an outbound call
# ---------------------------------------------------------
def twilio_make_call(to: str, base_url: Optional[str] = None, from_number: Optional[str] = None):
    sid = TWILIO_ACCOUNT_SID or os.getenv("TWILIO_ACCOUNT_SID")
    token = TWILIO_AUTH_TOKEN or os.getenv("TWILIO_AUTH_TOKEN")
    if not sid or not token:
        raise RuntimeError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set")

    client = TwilioClient(sid, token)
    from_num = from_number or TWILIO_FROM_NUMBER or os.getenv("TWILIO_FROM_NUMBER")
    if not from_num:
        raise RuntimeError("TWILIO_FROM_NUMBER must be set")

    if not base_url:
        base_url = BASE_URL or os.getenv("BASE_URL")
    if not base_url:
        raise RuntimeError("base_url must be provided or set in BASE_URL env var")

    twiml_url = f"{base_url.rstrip('/')}/voice/incoming"
    logger.info("Creating Twilio call from %s to %s (twiml_url=%s)", from_num, to, twiml_url)
    call = client.calls.create(to=to, from_=from_num, url=twiml_url)
    return {"sid": getattr(call, "sid", None), "status": getattr(call, "status", None)}

# ---------------------------------------------------------
# Utility: ffmpeg conversion check
# ---------------------------------------------------------
def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None

def convert_to_wav_file(input_path, output_path, sample_rate=16000):
    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", str(sample_rate), "-ac", "1", output_path]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {res.stderr.decode(errors='ignore')}")

# ---------------------------------------------------------
# Web UI endpoints: serve index and logs
# ---------------------------------------------------------
@app.get("/")
async def root():
    # Redirect to static index.html
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/dashboard")
async def dashboard():
    # Serve dashboard.html
    return FileResponse(os.path.join(static_dir, "dashboard.html"))

@app.get("/ui/logs")
async def ui_logs(lines: int = 200):
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as fh:
            all_lines = fh.readlines()
        tail = "".join(all_lines[-lines:])
        return PlainTextResponse(tail)
    except Exception as e:
        return PlainTextResponse(f"Failed reading log: {e}", status_code=500)

# ---------------------------------------------------------
# UI-triggered endpoint to create a Twilio call
# ---------------------------------------------------------
@app.post("/make-call")
async def ui_make_call(request: Request):
    """
    Expects JSON body: { "to": "+201234...", optional "base_url": "https://ngrok.io" }
    Returns JSON with call SID.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    to = payload.get("to")
    if not to:
        raise HTTPException(status_code=400, detail="Missing 'to' phone number")

    base_url = payload.get("base_url") or BASE_URL or str(request.base_url).rstrip("/")
    try:
        res = twilio_make_call(to=to, base_url=base_url)
        return JSONResponse(status_code=200, content=res)
    except Exception as e:
        logger.exception("Failed to create Twilio call: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# Direct audio upload endpoint (for UI testing)
# ---------------------------------------------------------
@app.post("/process-audio-direct")
async def process_audio_direct(file: UploadFile = File(...)):
    """
    Accepts an audio file upload, processes it through the pipeline,
    and returns transcript, intent, response_text, and audio_url.
    """
    call_sid = "web"
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] if file.filename else ".tmp") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Convert to WAV if needed
        wav_path = tmp_path
        if file.filename and not file.filename.lower().endswith('.wav'):
            # Try to convert using ffmpeg if available
            if _ffmpeg_available():
                wav_output = tmp_path + ".wav"
                try:
                    convert_to_wav_file(tmp_path, wav_output)
                    wav_path = wav_output
                    os.remove(tmp_path)  # Remove original
                except Exception as e:
                    logger.warning(f"ffmpeg conversion failed, trying direct: {e}")
                    # Fallback: try convert_to_wav_bytes
                    try:
                        wav_bytes = convert_to_wav_bytes(content)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_tmp:
                            wav_tmp.write(wav_bytes)
                            wav_path = wav_tmp.name
                        os.remove(tmp_path)
                    except Exception as e2:
                        logger.warning(f"convert_to_wav_bytes also failed: {e2}")
            else:
                # Try convert_to_wav_bytes directly
                try:
                    wav_bytes = convert_to_wav_bytes(content)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_tmp:
                        wav_tmp.write(wav_bytes)
                        wav_path = wav_tmp.name
                    os.remove(tmp_path)
                except Exception as e:
                    logger.warning(f"convert_to_wav_bytes failed: {e}")

        # Run pipeline
        result = core_voice_pipeline(wav_path, call_sid=call_sid)

        # Cleanup
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            if os.path.exists(tmp_path) and tmp_path != wav_path:
                os.remove(tmp_path)
        except Exception:
            pass

        return JSONResponse(content=result)

    except Exception as e:
        logger.exception(f"Error processing audio upload: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------------------------------------------------
# Dashboard API endpoints
# ---------------------------------------------------------
@app.get("/api/calls")
async def api_calls():
    """
    Returns list of all calls with metadata (sid, turns, last_ts, last_user).
    """
    with conversation_lock:
        calls = []
        for sid, history in conversation_store.items():
            if not history:
                continue
            last_item = max(history, key=lambda x: x.get("ts", 0))
            calls.append({
                "sid": sid,
                "turns": len(history),
                "last_ts": last_item.get("ts"),
                "last_user": next((h["text"][:50] for h in reversed(history) if h["role"] == "user"), None)
            })
    return JSONResponse(content=calls)

@app.get("/api/call/{sid}")
async def api_call_detail(sid: str):
    """
    Returns detailed information about a specific call including history and debug files.
    """
    history = get_conversation(sid)
    if not history:
        raise HTTPException(status_code=404, detail="Call not found")

    # Find created timestamp (first item)
    created_ts = history[0].get("ts") if history else None

    # Find debug files for this call_sid
    debug_files = []
    try:
        if os.path.exists(debug_dir):
            for filename in os.listdir(debug_dir):
                if sid in filename and filename.endswith(('.wav', '.mp3')):
                    debug_files.append(filename)
    except Exception as e:
        logger.warning(f"Error listing debug files: {e}")

    return JSONResponse(content={
        "sid": sid,
        "created_ts": created_ts,
        "history": history,
        "debug_files": sorted(debug_files)
    })

@app.post("/api/clear/{sid}")
async def api_clear_call(sid: str):
    """
    Clears conversation history for a specific call.
    """
    clear_conversation(sid)
    return JSONResponse(content={"status": "cleared", "sid": sid})

@app.get("/api/debug/{filename}")
async def api_debug_file(filename: str):
    """
    Serves debug audio files from debug_recordings/ directory.
    """
    # Security: prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = os.path.join(debug_dir, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, media_type="audio/wav")

# ---------------------------------------------------------
# Twilio endpoints: incoming call and processing
# ---------------------------------------------------------
@app.post("/voice/incoming")
async def incoming_call(request: Request):
    """
    Entry point for Twilio when a call is answered (or for outbound calls Twilio dials).
    Greets the user and records their reply (dual-channel).
    """
    host = request.url.hostname
    base = BASE_URL or f"https://{host}"
    resp = VoiceResponse()
    # Greet in English (agent persona)
    resp.say("Welcome to our Test & Go Agent. We are getting you connected.", voice="alice", language="en-US")
    process_url = f"{base.rstrip('/')}/voice/process"
    resp.record(action=process_url, max_length=60, play_beep=True, trim="trim-silence", recording_channels="dual")
    resp.redirect(f"{base.rstrip('/')}/voice/incoming")
    return Response(content=str(resp), media_type="application/xml")

@app.post("/voice/process")
async def process_voice_input(request: Request):
    """
    Receives RecordingUrl from Twilio, downloads the recording (with auth if needed),
    selects the correct channel, runs the pipeline synchronously, then returns TwiML
    that plays TTS and records the next turn.
    """
    form = await request.form()
    recording_url = form.get("RecordingUrl")
    call_sid = form.get("CallSid") or "unknown"
    host = request.url.hostname
    base_url = BASE_URL or f"https://{host}"

    if not recording_url:
        resp = VoiceResponse()
        resp.say("I didn't catch that. Please try again.")
        resp.redirect(f"{base_url.rstrip('/')}/voice/incoming")
        return Response(content=str(resp), media_type="application/xml")

    try:
        wav_url = recording_url if recording_url.endswith(".wav") else recording_url + ".wav"
        logger.info(f"Downloading Twilio recording: {wav_url}")

        # Download raw bytes (try with and without auth)
        r = requests.get(wav_url, timeout=30)
        if r.status_code == 401 and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
            logger.info("Unauthorized — retrying Twilio download with auth")
            r = requests.get(wav_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=30)
        r.raise_for_status()
        raw_bytes = r.content

        # save raw for debug
        try:
            raw_path = os.path.join(debug_dir, f"raw_{call_sid}_{int(time.time())}.wav")
            with open(raw_path, "wb") as fh:
                fh.write(raw_bytes)
            logger.info(f"Saved raw Twilio recording -> {raw_path}")
        except Exception:
            pass

        # Choose best channel (mono bytes)
        chosen_bytes, chosen_channel, transcripts = choose_best_channel_and_get_mono_bytes(raw_bytes, language="ar")
        logger.info(f"[{call_sid}] chosen_channel={chosen_channel} transcripts={ {k:(v[:60]+'...' if v else '') for k,v in transcripts.items()} }")

        # save chosen bytes for debug
        try:
            chosen_path = os.path.join(debug_dir, f"chosen_{call_sid}_{int(time.time())}.wav")
            with open(chosen_path, "wb") as fh:
                fh.write(chosen_bytes)
            logger.info(f"Saved chosen channel wav -> {chosen_path}")
        except Exception:
            pass

        # write chosen bytes to temp file and run pipeline
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tmp_wav.write(chosen_bytes)
            tmp_wav_path = tmp_wav.name

        logger.info("Processing audio pipeline...")
        result = core_voice_pipeline(tmp_wav_path, call_sid=call_sid)

        # cleanup tmp
        try:
            os.remove(tmp_wav_path)
        except Exception:
            pass

        # Build TwiML response
        resp = VoiceResponse()
        if result.get("audio_url"):
            relative_path = result['audio_url']
            full_audio_url = relative_path if relative_path.startswith("http") else f"{base_url.rstrip('/')}{relative_path}"
            resp.play(full_audio_url)
        else:
            resp.say(result.get("response_text", "I processed that but lost my voice."))

        # pause briefly before next recording to avoid capturing playback
        resp.pause(length=1)

        next_action_url = f"{base_url.rstrip('/')}/voice/process"
        resp.record(action=next_action_url, max_length=60, play_beep=False, trim="trim-silence", recording_channels="dual")
        resp.say("Goodbye.")
        resp.hangup()
        return Response(content=str(resp), media_type="application/xml")

    except Exception as e:
        logger.exception(f"Error in process_voice_input: {e}")
        resp = VoiceResponse()
        resp.say("An error occurred in the system.")
        return Response(content=str(resp), media_type="application/xml")

@app.post("/recording-complete")
async def recording_complete(request: Request, background_tasks: BackgroundTasks):
    """
    Background processing for recordings (optional). Twilio may call this when recording is completed.
    We support an async background processor for recording-complete if you prefer that flow.
    """
    form = await request.form()
    recording_url = form.get("RecordingUrl")
    call_sid = form.get("CallSid") or "unknown"

    if not recording_url:
        return PlainTextResponse("No recording url", status_code=400)

    wav_url = recording_url if recording_url.endswith(".wav") else recording_url + ".wav"
    background_tasks.add_task(process_twilio_recording_background, wav_url, call_sid)
    return PlainTextResponse("ok")

def process_twilio_recording_background(wav_url, call_sid):
    try:
        logger.info(f"Background: Downloading & preprocessing Twilio recording: {wav_url}")
        try:
            wav_bytes = demo_process_url_and_prepare_for_asr(wav_url)
        except Exception as e:
            logger.warning(f"demo_process_url_and_prepare_for_asr failed in background: {e}. Falling back to direct download.")
            r = requests.get(wav_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=30)
            r.raise_for_status()
            wav_bytes = r.content

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(wav_bytes)
        tmp.close()
        result = core_voice_pipeline(tmp.name, call_sid)
        logger.info(f"Twilio background processing done. AI said: {result.get('response_text')}")
        try:
            os.remove(tmp.name)
        except Exception:
            pass
    except Exception as e:
        logger.exception(f"Twilio background task failed: {e}")

# ---------------------------------------------------------
# Twilio status callback to clear conversation
# ---------------------------------------------------------
@app.post("/call-status")
async def call_status(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    call_status = form.get("CallStatus")
    logger.info(f"Call status event: sid={call_sid} status={call_status}")
    if call_sid and call_status and call_status.lower() in ("completed", "failed", "busy", "no-answer"):
        logger.info(f"Clearing conversation history for {call_sid}")
        clear_conversation(call_sid)
    return PlainTextResponse("ok")

# ---------------------------------------------------------
# Runner
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5050))
    print(f"Starting server on http://0.0.0.0:{port}")
    print(f"Access the UI at http://localhost:{port}")
    print(f"Access the dashboard at http://localhost:{port}/dashboard")
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass
    # Recommend running with environment var to avoid console encoding issues:
    # set PYTHONIOENCODING=utf-8 in your environment before starting python.
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")