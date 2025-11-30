# modules/preprocess.py
# Updated preprocessing: Twilio auth retry and dual-channel extraction for caller channel
import io
import os
import logging
import requests
from requests.auth import HTTPBasicAuth
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import librosa
import webrtcvad

logger = logging.getLogger("preprocess")

try:
    import noisereduce as nr
    _HAS_NOISEREDUCE = True
except Exception:
    _HAS_NOISEREDUCE = False


def fetch_audio_from_url(url: str, timeout: int = 30) -> bytes:
    """
    Fetch audio bytes from a URL. If 401 -> retry with Twilio auth (TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN).
    """
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 401:
            sid = os.getenv("TWILIO_ACCOUNT_SID")
            token = os.getenv("TWILIO_AUTH_TOKEN")
            if sid and token:
                logger.info("fetch_audio_from_url: 401 received, retrying with Twilio auth")
                resp = requests.get(url, auth=HTTPBasicAuth(sid, token), timeout=timeout)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.exception("fetch_audio_from_url failed: %s", e)
        raise


def extract_first_channel_if_stereo(wav_bytes: bytes) -> bytes:
    """
    If wav_bytes contains multi-channel audio, extract the first channel and return mono WAV bytes.
    Uses soundfile to read and write.
    """
    try:
        b = io.BytesIO(wav_bytes)
        data, sr = sf.read(b, dtype='int16')
        # If stereo/multi-channel, extract first channel
        if data.ndim > 1 and data.shape[1] >= 2:
            logger.info("Audio has multiple channels; extracting first channel for caller.")
            channel0 = data[:, 0]
            out_buf = io.BytesIO()
            sf.write(out_buf, channel0, sr, format="WAV", subtype="PCM_16")
            out_buf.seek(0)
            return out_buf.read()
        else:
            return wav_bytes
    except Exception as e:
        logger.exception("extract_first_channel_if_stereo failed, returning original bytes: %s", e)
        return wav_bytes


def convert_to_wav_bytes(
    input_bytes: bytes,
    input_format: str = None,
    target_sr: int = 16000,
    mono: bool = True,
) -> bytes:
    audio_file = io.BytesIO(input_bytes)
    seg = AudioSegment.from_file(audio_file, format=input_format)
    if mono:
        seg = seg.set_channels(1)
    seg = seg.set_frame_rate(target_sr)
    out_buf = io.BytesIO()
    seg.export(out_buf, format="wav", parameters=["-acodec", "pcm_s16le"])
    return out_buf.getvalue()


def load_audio_from_bytes(wav_bytes: bytes):
    b = io.BytesIO(wav_bytes)
    data, sr = sf.read(b, dtype="int16")
    if data.dtype == np.int16 or data.dtype == np.int32:
        max_val = np.iinfo(data.dtype).max
        y = data.astype(np.float32) / float(max_val)
    else:
        y = data.astype(np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y, sr


def normalize_audio(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    peak = np.max(np.abs(y)) + eps
    return y.astype(np.float32) / peak


def trim_silence(y: np.ndarray, sr: int, top_db: int = 20) -> np.ndarray:
    try:
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        return yt
    except Exception:
        return y


def denoise_audio(y: np.ndarray, sr: int) -> np.ndarray:
    if not _HAS_NOISEREDUCE:
        return y
    try:
        noise_len = min(int(0.25 * sr), len(y))
        if noise_len < 100:
            return y
        noise_clip = y[:noise_len]
        reduced = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip, prop_decrease=1.0)
        return reduced.astype(np.float32)
    except Exception:
        return y


def frame_generator(frame_ms: int, audio: np.ndarray, sr: int):
    assert frame_ms in (10, 20, 30)
    frame_length = int(sr * (frame_ms / 1000.0))
    pcm = (audio * 32767.0).astype(np.int16)
    offset = 0
    while offset + frame_length <= len(pcm):
        frame = pcm[offset : offset + frame_length]
        yield (frame.tobytes(), int(offset / sr * 1000))
        offset += frame_length


def vad_collector(
    audio: np.ndarray,
    sr: int,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    padding_ms: int = 500,
):
    vad = webrtcvad.Vad(aggressiveness)
    frames = list(frame_generator(frame_ms, audio, sr))
    if not frames:
        return []
    window_size = int(padding_ms / frame_ms)
    triggered = False
    voiced_frames = []
    segments = []
    start_frame = 0
    for i, (frame_bytes, _) in enumerate(frames):
        is_speech = vad.is_speech(frame_bytes, sample_rate=sr)
        if not triggered:
            if is_speech:
                triggered = True
                start_frame = i
                voiced_frames = [i]
        else:
            voiced_frames.append(i)
            recent = frames[max(0, i - window_size + 1) : i + 1]
            recent_speech = [vad.is_speech(fb, sample_rate=sr) for fb, _ in recent]
            if not any(recent_speech):
                end_frame = i
                start_sample = start_frame * int(sr * (frame_ms / 1000.0))
                end_sample = (end_frame + 1) * int(sr * (frame_ms / 1000.0))
                segments.append((start_sample, end_sample))
                triggered = False
                voiced_frames = []
    if triggered and voiced_frames:
        start_sample = start_frame * int(sr * (frame_ms / 1000.0))
        end_sample = len(audio)
        segments.append((start_sample, end_sample))
    # merge close segments
    merged = []
    pad_samples = int((padding_ms / 1000.0) * sr)
    for s, e in segments:
        if not merged:
            merged.append((s, e))
        else:
            prev_s, prev_e = merged[-1]
            if s <= prev_e + pad_samples:
                merged[-1] = (prev_s, max(prev_e, e))
            else:
                merged.append((s, e))
    return merged


def join_segments_and_get_wav_bytes(segments, sr):
    import soundfile as sf
    import io
    import numpy as np

    if len(segments) == 0:
        return None

    int16 = np.concatenate(segments).astype(np.int16)
    out_buf = io.BytesIO()
    sf.write(out_buf, int16, sr, format="WAV", subtype="PCM_16")
    out_buf.seek(0)
    return out_buf.read()


def save_wav_bytes_from_array(y, sr):
    int16 = (y * 32767.0).astype(np.int16)
    out = io.BytesIO()
    sf.write(out, int16, sr, subtype="PCM_16")
    return out.getvalue()


def demo_process_url_and_prepare_for_asr(url: str, target_sr: int = 16000):
    """
    - fetches audio bytes from URL (with Twilio auth fallback),
    - extracts caller channel if multi-channel,
    - converts to WAV PCM16 @ target_sr,
    - denoises / trims silence / normalizes,
    - runs VAD and returns concatenated voiced segments as WAV bytes.
    """
    raw = fetch_audio_from_url(url)

    # If Twilio produced dual-channel recording, extract the caller channel first
    # (some Twilio dual-channel files are stereo WAVs where channel 0 is caller).
    raw = extract_first_channel_if_stereo(raw)

    wav_bytes = convert_to_wav_bytes(raw, target_sr=target_sr, mono=True)
    y, sr = load_audio_from_bytes(wav_bytes)
    if _HAS_NOISEREDUCE:
        y = denoise_audio(y, sr)
    y = trim_silence(y, sr)
    y = normalize_audio(y)
    segments = vad_collector(y, sr, aggressiveness=2, frame_ms=30, padding_ms=500)
    if segments:
        seg_arrays = [y[s:e] for s, e in segments]
        return join_segments_and_get_wav_bytes(seg_arrays, sr)
    else:
        return save_wav_bytes_from_array(y, sr)