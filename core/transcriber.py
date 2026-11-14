import os
from typing import List, Tuple

import requests
from pydub import AudioSegment
from pydub.silence import detect_silence

from core.segments import Segment, segments_to_text

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# Pieces are at most 25s (5s safety margin) and are cut at silences when possible.
SARVAM_PIECE_SECONDS = 25
SARVAM_SEARCH_WINDOW_MS = 5000   # look for a pause in the last 5s of each piece
MIN_PIECE_MS = 300               # ignore slivers too short to contain speech

SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"


# Settings are read when used (not at import), so a .env loaded later still counts.
def _whisper_model_name() -> str:
    return os.getenv("WHISPER_MODEL", "small")


def _sarvam_key():
    return os.getenv("SARVAM_API_KEY")


def _sarvam_model() -> str:
    return os.getenv("SARVAM_STT_MODEL", "saaras:v3")

_model = None
_backend = None  # "faster" or "openai"


# --------------------------------------------------------------------------
# Whisper (local) - supports faster-whisper (preferred) and openai-whisper
# --------------------------------------------------------------------------
def load_model():
    global _model, _backend

    if _model is None:
        model_name = _whisper_model_name()
        print(f"Loading Whisper model: {model_name} ...")
        try:
            from faster_whisper import WhisperModel
            _model = WhisperModel(model_name, device="cpu", compute_type="int8")
            _backend = "faster"
        except ImportError:
            import whisper
            _model = whisper.load_model(model_name)
            _backend = "openai"
        print(f"Whisper model loaded ({_backend}).")
    return _model


def _whisper_segments(chunk_path: str) -> List[Segment]:
    """Transcribe one audio file; timestamps are relative to that file."""
    model = load_model()
    segments: List[Segment] = []

    if _backend == "faster":
        raw, _info = model.transcribe(chunk_path, task="transcribe", vad_filter=True)
        for s in raw:  # generator - consume it here
            text = s.text.strip()
            if text:
                segments.append(Segment(float(s.start), float(s.end), text))
    else:
        result = model.transcribe(chunk_path, task="transcribe")
        for s in result["segments"]:
            text = s["text"].strip()
            if text:
                segments.append(Segment(float(s["start"]), float(s["end"]), text))
    return segments


def transcribe_chunk_whisper(chunk_path: str) -> str:
    return segments_to_text(_whisper_segments(chunk_path))


# --------------------------------------------------------------------------
# Sarvam (Hinglish -> English)
# --------------------------------------------------------------------------
def _send_to_sarvam(piece_path: str) -> str:
    """Send one <=30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": _sarvam_key()}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": _sarvam_model(), "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def _plan_pieces(
    total_ms: int,
    silences: List[Tuple[int, int]],
    max_ms: int,
    window_ms: int = SARVAM_SEARCH_WINDOW_MS,
) -> List[Tuple[int, int]]:
    """
    Split [0, total_ms) into pieces of at most max_ms, preferring to cut in the
    middle of a silence that falls in the last window_ms of each piece so words
    are not sliced in half. Falls back to a hard cut at max_ms.
    """
    pieces: List[Tuple[int, int]] = []
    cursor = 0
    while cursor < total_ms:
        if total_ms - cursor <= max_ms:
            pieces.append((cursor, total_ms))
            break
        limit = cursor + max_ms
        mids = [(s + e) // 2 for s, e in silences]
        candidates = [m for m in mids if limit - window_ms <= m <= limit]
        cut = max(candidates) if candidates else limit
        pieces.append((cursor, cut))
        cursor = cut
    return pieces


def _sarvam_segments(chunk_path: str) -> List[Segment]:
    if not _sarvam_key():
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_file(chunk_path)
    total_ms = len(audio)

    dbfs = audio.dBFS
    thresh = dbfs - 16 if dbfs != float("-inf") else -60
    silences = detect_silence(audio, min_silence_len=300, silence_thresh=thresh)

    pieces = _plan_pieces(total_ms, silences, SARVAM_PIECE_SECONDS * 1000)

    segments: List[Segment] = []
    for i, (start_ms, end_ms) in enumerate(pieces):
        if end_ms - start_ms < MIN_PIECE_MS:
            continue

        piece_path = f"{chunk_path}_sv_{i}.wav"
        audio[start_ms:end_ms].export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{len(pieces)} ...")
            text = _send_to_sarvam(piece_path).strip()
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

        if text:
            segments.append(Segment(start_ms / 1000.0, end_ms / 1000.0, text))
    return segments


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    return segments_to_text(_sarvam_segments(chunk_path))


# --------------------------------------------------------------------------
# Routing + public API
# --------------------------------------------------------------------------
def _chunk_segments(chunk_path: str, language: str) -> List[Segment]:
    if language.lower() == "hinglish":
        return _sarvam_segments(chunk_path)
    return _whisper_segments(chunk_path)


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  -> Whisper (local model)
    - hinglish -> Sarvam (translates to English while transcribing)
    """
    return segments_to_text(_chunk_segments(chunk_path, language))


def _audio_duration_seconds(path: str) -> float:
    return len(AudioSegment.from_file(path)) / 1000.0


def transcribe_all_segments(chunks: list, language: str = "english") -> List[Segment]:
    """
    Transcribe every chunk and return segments with timestamps relative to the
    START OF THE FULL VIDEO.

    Assumes `chunks` are consecutive, non-overlapping slices of the audio in
    playback order (each chunk's offset = total duration of the chunks before it).
    """
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    all_segments: List[Segment] = []
    offset = 0.0

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        for seg in _chunk_segments(chunk, language):
            all_segments.append(
                Segment(seg.start + offset, seg.end + offset, seg.text)
            )

        offset += _audio_duration_seconds(chunk)

    print("Transcription complete.")
    return all_segments


def transcribe_all(chunks: list, language: str = "english") -> str:
    """Backwards-compatible: returns the plain transcript string."""
    return segments_to_text(transcribe_all_segments(chunks, language))