"""Utility helpers to convert transcription segments into SRT output."""

from __future__ import annotations

from typing import Iterable, Iterator, List, Mapping

_MIN_SEGMENT_DURATION = 0.5  # seconds


def _format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total_milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def iter_srt_blocks(segments: Iterable[Mapping[str, object]]) -> Iterator[str]:
    """Yield SRT-formatted blocks for each provided segment."""
    has_output = False
    prefix = ""

    for index, segment in enumerate(segments, start=1):
        raw_start = float(segment.get("start", 0.0) or 0.0)
        raw_end = float(segment.get("end", raw_start) or raw_start)
        if raw_end <= raw_start:
            raw_end = raw_start + _MIN_SEGMENT_DURATION

        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        start_ts = _format_timestamp(raw_start)
        end_ts = _format_timestamp(raw_end)

        entry = "\n".join((
            str(index),
            f"{start_ts} --> {end_ts}",
            text,
        ))

        yield f"{prefix}{entry}"
        prefix = "\n\n"
        has_output = True

    if has_output:
        yield "\n"


def segments_to_srt(segments: Iterable[Mapping[str, object]]) -> str:
    """Convert verbose transcription segments into an SRT formatted string."""
    rendered = "".join(iter_srt_blocks(segments))
    return rendered if rendered else ""


def build_single_segment(text: str) -> list[dict[str, object]]:
    """Fallback helper producing a single segment when no timing data is available."""
    cleaned = text.strip()
    if not cleaned:
        return []
    duration = max(_MIN_SEGMENT_DURATION, len(cleaned.split()) * 0.4)
    return [{"start": 0.0, "end": duration, "text": cleaned}]
