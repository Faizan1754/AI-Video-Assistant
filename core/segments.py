from dataclasses import dataclass
from typing import Iterable, List
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


@dataclass
class Segment:
    """One piece of transcript with its position in the video (seconds)."""
    start: float
    end: float
    text: str


def format_timestamp(seconds: float) -> str:
    """65 -> '01:05', 3725 -> '1:02:05'."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def segments_to_text(segments: Iterable[Segment]) -> str:
    """Plain transcript (no timestamps) - what the summarizer/extractor consume."""
    return " ".join(s.text.strip() for s in segments if s.text.strip())


def segments_to_timestamped_text(segments: Iterable[Segment]) -> str:
    """One line per segment: '[01:05] text' - useful for display/export."""
    return "\n".join(
        f"[{format_timestamp(s.start)}] {s.text.strip()}"
        for s in segments
        if s.text.strip()
    )


def youtube_link(url: str, seconds: float) -> str:
    """Return the YouTube URL with a '&t=<seconds>s' jump parameter."""
    parts = urlparse(url)
    query = dict(parse_qsl(parts.query))
    query["t"] = f"{int(max(0, seconds))}s"
    return urlunparse(parts._replace(query=urlencode(query)))