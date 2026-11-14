import os
import re

import yt_dlp
from pydub import AudioSegment

Download_Dir = "downloads"
os.makedirs(Download_Dir, exist_ok=True)

# YouTube links pasted without the scheme, e.g. "youtube.com/watch?v=..."
_BARE_YOUTUBE = re.compile(r"^(www\.|m\.|music\.)?(youtube\.com|youtu\.be)/", re.IGNORECASE)


def is_url(source: str) -> bool:
    s = source.strip()
    return s.lower().startswith(("http://", "https://")) or bool(_BARE_YOUTUBE.match(s))


_DOWNLOAD_HELP = (
    "Could not download this video from YouTube. Things to try:\n"
    "1) Update yt-dlp: pip install -U \"yt-dlp[default]\" (YouTube changes often; "
    "old versions stop working).\n"
    "2) Install a JavaScript runtime, Deno is the default one "
    "(https://github.com/yt-dlp/yt-dlp/wiki/EJS).\n"
    "3) If YouTube asks you to sign in, add YTDLP_COOKIES_FROM_BROWSER=firefox (or chrome) "
    "or YTDLP_COOKIES_FILE=cookies.txt to your .env file.\n"
    "4) Or download the video yourself and upload it as a file."
)


def download_audio(url : str) -> str:
    # Name files by video id: avoids special characters in titles and stops two
    # different videos (or two sessions) from clobbering each other's file.
    output_path = os.path.join(Download_Dir, '%(id)s.%(ext)s')
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    # Optional: log in with your own browser/cookies when YouTube says
    # "Sign in to confirm you're not a bot". Read at call time so .env works.
    cookies_file = os.getenv("YTDLP_COOKIES_FILE")
    cookies_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER")
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file
    elif cookies_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_browser.strip().lower(),)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # The post-processed file path is reported by yt-dlp itself; the old
            # .replace(".webm"/".m4a", ".wav") broke for any other extension.
            downloads = info.get("requested_downloads") or []
            filename = downloads[0].get("filepath") if downloads else None
            if not filename:
                filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(f"{_DOWNLOAD_HELP}\n\nOriginal error: {e}") from e
    return filename

def convert_to_wav(input_file: str) -> str:
    # Convert the input audio file to WAV format using pydub
    output_path = os.path.splitext(input_file)[0] + ".converted.wav"
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1).set_frame_rate(16000) # 16khz mono audio
    audio.export(output_path, format="wav")
    return output_path

def chunk_audio(wav_path : str, chunk_minutes : int = 10) -> list:
    # NOTE: chunks must stay consecutive and NON-overlapping, in order -
    # transcriber.transcribe_all_segments relies on that to compute timestamps.
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 #convert minutes to milliseconds as Chunk length in pydub is in milliseconds
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks

def cleanup_audio(chunks: list) -> None:
    """Delete the chunk files and the intermediate wav they were cut from.

    The wav is always a file this module created (the downloaded audio or the
    '.converted.wav' copy), never the user's original recording.
    """
    marker = "_chunk_"
    wavs = set()
    for chunk in chunks:
        if os.path.exists(chunk):
            os.remove(chunk)
        if marker in chunk:
            wavs.add(chunk.rsplit(marker, 1)[0])
    for wav in wavs:
        if os.path.exists(wav):
            os.remove(wav)

def process_input(source : str) -> list:
    source = source.strip()
    if is_url(source):
        if not source.lower().startswith(("http://", "https://")):
            source = "https://" + source
        print("Detected URL. Downloading audio...")
        wav_path = download_audio(source)
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")
        print("Detected local file. Converting to wav format...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks