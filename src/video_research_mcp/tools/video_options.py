"""Per-request video processing options: media resolution, clipping, fps.

Gemini samples video at 1 fps. On Gemini 3 models the default media
resolution spends about 70 tokens per frame; ``high`` spends about 280
(Google's media-resolution docs), roughly 3x the cost per second of video
once audio is included. ``high`` is set per part, which only Gemini 3 models
accept, and only takes effect on uploaded files: on a YouTube URL part it is
ignored. Clipping and fps work on both.

Suggested pattern for on-screen text: analyze the whole video at ``low``,
read the ``detail_segments`` it returns, then re-run ``high`` on just those
ranges with ``start_offset`` / ``end_offset``.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from google.genai import types
from pydantic import Field

MediaResolution = Literal["low", "high"]

MediaResolutionParam = Annotated[MediaResolution | None, Field(
    description="Frame detail Gemini reads. 'low' (the API default, about 70 "
    "tokens per frame) suits speech-led video. 'high' (about 280 tokens per "
    "frame) helps read small on-screen text such as slides, code or numbers; "
    "it needs a Gemini 3 model, and YouTube URLs are downloaded and uploaded "
    "for it because Gemini ignores it on URL parts. Defaults to "
    "VIDEO_MEDIA_RESOLUTION. Pair 'high' with start_offset/end_offset to "
    "re-read only the segments that need it."
)]

OffsetParam = Annotated[str | int | float | None, Field(
    description="Clip boundary: seconds (90 or '90s'), MM:SS ('1:30') or HH:MM:SS ('01:01:30')."
)]

FpsParam = Annotated[float | None, Field(
    gt=0, le=24,
    description="Frames sampled per second (Gemini default 1). Raise for "
    "fast screen demos; lower for long static talks.",
)]

_HIGH = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH

_CLOCK_RE = re.compile(r"^(?:(\d+):)?(\d+):(\d{1,2}(?:\.\d+)?)$")
_SECONDS_RE = re.compile(r"^(\d+(?:\.\d+)?)s?$")


def _format_seconds(seconds: float) -> str:
    text = f"{seconds:.3f}".rstrip("0").rstrip(".")
    return f"{text}s"


def parse_offset(value: str | int | float | None) -> str | None:
    """Normalize a clip boundary to Gemini's ``'<seconds>s'`` form.

    Accepts numbers, plain or suffixed seconds ('90', '90s'), MM:SS ('1:30',
    '90:00') and HH:MM:SS ('01:01:30').

    Raises:
        ValueError: If the value is not one of those forms.
    """
    if value is None or isinstance(value, bool):
        return None if value is None else _invalid(value)
    if isinstance(value, (int, float)):
        if value < 0:
            _invalid(value)
        return _format_seconds(float(value))
    raw = value.strip().lower()
    if not raw:
        return None
    m = _SECONDS_RE.match(raw)
    if m:
        return _format_seconds(float(m.group(1)))
    m = _CLOCK_RE.match(raw)
    if not m:
        _invalid(value)
    hours = int(m.group(1)) if m.group(1) else None
    minutes, secs = int(m.group(2)), float(m.group(3))
    if secs >= 60 or (hours is not None and minutes >= 60):
        _invalid(value)
    return _format_seconds((hours or 0) * 3600 + minutes * 60 + secs)


def _invalid(value) -> None:
    raise ValueError(
        f"Invalid offset {value!r}: use seconds (90), MM:SS ('1:30') or HH:MM:SS ('01:01:30')"
    )


def resolve_options(
    media_resolution: str | None,
    start_offset: str | int | float | None,
    end_offset: str | int | float | None,
    fps: float | None,
    *,
    default_resolution: str,
    model: str,
) -> dict:
    """Validate and normalize options into a dict used for building and cache keys.

    Raises:
        ValueError: On an unknown resolution, 'high' on a model that does not
            accept per-part resolution, a malformed offset, or a clip whose end
            is not after its start.
    """
    resolution = (media_resolution or default_resolution).strip().lower()
    if resolution not in ("low", "high"):
        raise ValueError(f"media_resolution must be 'low' or 'high', got {media_resolution!r}")
    if resolution == "high" and not model.removeprefix("models/").startswith("gemini-3"):
        raise ValueError(
            f"media_resolution='high' needs a Gemini 3 model (per-part media "
            f"resolution); the configured model is {model!r}"
        )
    start, end = parse_offset(start_offset), parse_offset(end_offset)
    if start and end and float(end[:-1]) <= float(start[:-1]):
        raise ValueError(f"end_offset ({end}) must be after start_offset ({start})")
    return {
        "media_resolution": resolution,
        "start_offset": start,
        "end_offset": end,
        "fps": fps,
    }


def cache_key_suffix(options: dict, source: str) -> str:
    """Suffix for the result-cache key so results never cross option or source boundaries."""
    return (
        f"\n[video-options res={options['media_resolution']} "
        f"start={options['start_offset']} end={options['end_offset']} "
        f"fps={options['fps']} source={source}]"
    )


def apply_video_options(
    contents: types.Content,
    options: dict,
    *,
    fps_fallback: float | None = None,
) -> types.Content:
    """Return a copy of ``contents`` with options set on every video part.

    ``media_resolution`` is attached only for 'high': 'low' is the API
    default, and leaving the field off keeps default requests identical to
    before (models without per-part resolution reject the field outright).
    ``fps_fallback`` (e.g. a metadata-derived rate) applies only when the
    caller did not set ``fps``. Text parts are left alone.
    """
    if not isinstance(contents, types.Content):
        return contents
    fps = options["fps"] if options["fps"] is not None else fps_fallback
    new_parts = []
    for part in contents.parts or []:
        if part.text is not None or (part.file_data is None and part.inline_data is None):
            new_parts.append(part)
            continue
        updated = part.model_copy(deep=True)
        if options["media_resolution"] == "high":
            updated.media_resolution = types.PartMediaResolution(level=_HIGH)
        if fps is not None or options["start_offset"] or options["end_offset"]:
            updated.video_metadata = types.VideoMetadata(
                fps=fps,
                start_offset=options["start_offset"],
                end_offset=options["end_offset"],
            )
        new_parts.append(updated)
    return types.Content(role=contents.role, parts=new_parts)
