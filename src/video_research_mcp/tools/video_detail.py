"""Automatic high-resolution re-reads of the segments a first pass flags.

A whole-video pass at the default media resolution returns
``detail_segments``: ranges where on-screen text matters but may not be
legible. This module turns those into short local clips with ffmpeg (padded,
merged, capped), and reads each clip at ``MEDIA_RESOLUTION_HIGH``. Clips are
small enough to send inline, so the full video never has to go through the
File API, and Gemini only bills the seconds in each clip.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path

from google.genai import types
from pydantic import BaseModel, Field

from ..cache import load as cache_load, save as cache_save
from ..client import GeminiClient
from ..config import get_config
from .video_file import _upload_large_file
from .video_options import apply_video_options, parse_offset

logger = logging.getLogger(__name__)

PAD_SECONDS = 3.0
# Gemini video docs: inline data "is suitable for shorter videos under 20MB
# total request size", and inline bytes travel base64 encoded (4/3 larger).
# 12,000,000 bytes encode to 16,000,000, leaving about 4 MB of the 20 MB for
# the prompt and JSON framing; larger clips go through the File API.
INLINE_MAX_BYTES = 12_000_000
MAX_SEGMENTS = 6
MAX_SEGMENT_SECONDS = 120.0
_CONCURRENCY = 3
# ffprobe reads only the container header. ffmpeg re-encodes, so its bound
# grows with clip length: a fixed allowance plus a few seconds per second of
# clip, generous for a slow CPU at 1080p with the veryfast preset.
FFPROBE_TIMEOUT_SECONDS = 30.0
FFMPEG_BASE_TIMEOUT_SECONDS = 60.0
FFMPEG_SECONDS_PER_CLIP_SECOND = 5.0

_READ_PROMPT = (
    "This clip is part of a longer video. Transcribe ALL on-screen text in it "
    "verbatim, in reading order: slides, code, numbers, labels, UI. Keep the "
    "original wording, spelling and numbering. Where any word is not legible, "
    "write [illegible] in its place; never guess or paraphrase. Then add brief "
    "notes on anything on screen that matters but is not text.\n\n"
    "Why this clip was selected: {reason}\n"
    "The caller's overall instruction for the video: {instruction}"
)


class DetailReading(BaseModel):
    """Structured output for one high-resolution clip read."""

    on_screen_text: str = Field(
        default="",
        description="All on-screen text in the clip, verbatim, in reading order, "
        "with [illegible] for anything that cannot be read",
    )
    visual_notes: str = Field(
        default="", description="Non-text visual detail that matters, briefly",
    )


def _seconds(ts: str) -> float | None:
    try:
        parsed = parse_offset(ts)
    except ValueError:
        return None
    return float(parsed[:-1]) if parsed else None


def plan_clips(
    segments: list[dict],
    duration: float | None,
    *,
    pad: float = PAD_SECONDS,
    max_segments: int = MAX_SEGMENTS,
    max_seconds: float = MAX_SEGMENT_SECONDS,
) -> tuple[list[dict], list[str]]:
    """Turn model-reported segments into clip ranges: padded, clamped, merged, capped.

    Model timestamps drift by a few seconds, so each range is padded. Ranges
    that overlap after padding are merged. Unparseable or empty ranges are
    skipped, overlong ones are truncated, and at most ``max_segments`` clips
    are returned, with a note for each thing dropped or cut.

    Returns:
        (clips, notes) where each clip is {start, end, reasons}.
    """
    notes: list[str] = []
    ranges = []
    for seg in segments:
        s, e = _seconds(seg.get("start", "")), _seconds(seg.get("end", ""))
        if s is None or e is None or e <= s:
            notes.append(f"Skipped segment with unusable times: {seg.get('start')!r}-{seg.get('end')!r}")
            continue
        orig = (s, e, seg.get("reason", ""))
        s, e = max(0.0, s - pad), e + pad
        if duration:
            e = min(e, duration)
        if e <= s:
            notes.append(
                f"Skipped segment {seg.get('start')!r}-{seg.get('end')!r}: it starts after "
                f"the end of the video ({duration:.0f}s)"
            )
            continue
        ranges.append({"start": s, "end": e, "parts": [orig]})

    ranges.sort(key=lambda r: r["start"])
    merged: list[dict] = []
    for r in ranges:
        if merged and r["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], r["end"])
            merged[-1]["parts"] += r["parts"]
        else:
            merged.append(r)

    for r in merged:
        if r["end"] - r["start"] > max_seconds:
            notes.append(
                f"Segment at {r['start']:.0f}s cut to {max_seconds:.0f}s for the high-resolution read"
            )
            r["end"] = r["start"] + max_seconds
        # Keep only reasons whose original (unpadded) range is still inside the clip.
        r["reasons"] = [
            reason for (s, e, reason) in r.pop("parts") if s < r["end"] and e > r["start"]
        ]
    if len(merged) > max_segments:
        notes.append(f"Read the first {max_segments} of {len(merged)} segments at high resolution")
        merged = merged[:max_segments]
    return merged, notes


async def _run(cmd: list[str], *, timeout: float) -> tuple[int, str, str]:
    """Run ``cmd``; kill it and raise RuntimeError if it outlives ``timeout`` seconds."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"{cmd[0]} did not finish within {timeout:.0f}s and was stopped") from None
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


async def probe_duration(path: Path) -> float | None:
    """Video duration in seconds via ffprobe, or None if it cannot be read.

    Raises:
        RuntimeError: If ffprobe runs past ``FFPROBE_TIMEOUT_SECONDS``.
    """
    code, out, _ = await _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], timeout=FFPROBE_TIMEOUT_SECONDS)
    try:
        return float(out.strip()) if code == 0 else None
    except ValueError:
        return None


async def trim_clip(source: Path, start: float, end: float, out_path: Path) -> Path:
    """Cut [start, end) from ``source`` into an MP4 at ``out_path``.

    Re-encodes rather than stream-copying, so the clip starts on the
    requested frame instead of the previous keyframe.

    Raises:
        RuntimeError: If ffmpeg fails, times out or produces no file.
    """
    timeout = FFMPEG_BASE_TIMEOUT_SECONDS + FFMPEG_SECONDS_PER_CLIP_SECOND * (end - start)
    try:
        code, _, err = await _run([
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{end - start:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-movflags", "+faststart", str(out_path),
        ], timeout=timeout)
    except RuntimeError:
        out_path.unlink(missing_ok=True)
        raise
    if code != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg could not cut {start:.1f}-{end:.1f}s: {err.strip()[:300]}")
    return out_path


def _clock(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


async def _read_clip(
    clip_path: Path,
    *,
    reason: str,
    instruction: str,
    thinking_level: str,
) -> DetailReading:
    if clip_path.stat().st_size < INLINE_MAX_BYTES:
        data = await asyncio.to_thread(clip_path.read_bytes)
        video = types.Part.from_bytes(data=data, mime_type="video/mp4")
    else:
        uri = await _upload_large_file(clip_path, "video/mp4", content_hash="")
        video = types.Part(file_data=types.FileData(file_uri=uri, mime_type="video/mp4"))
    prompt = _READ_PROMPT.format(reason=reason or "flagged by the first pass", instruction=instruction)
    contents = apply_video_options(
        types.Content(parts=[video, types.Part(text=prompt)]),
        {"media_resolution": "high", "start_offset": None, "end_offset": None, "fps": None},
    )
    return await GeminiClient.generate_structured(
        contents, schema=DetailReading, thinking_level=thinking_level,
    )


async def read_detail_segments(
    source: Path,
    segments: list[dict],
    *,
    content_id: str,
    instruction: str,
    thinking_level: str = "medium",
    use_cache: bool = True,
) -> tuple[list[dict], list[str]]:
    """Cut each flagged segment from ``source`` and read it at high resolution.

    Returns:
        (readings, notes). Each reading has the requested range, the padded
        clip range actually read, the reasons, and the transcription or an
        error for that clip. Notes explain anything skipped or truncated.
    """
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return [], ["auto_detail needs ffmpeg and ffprobe to cut clips; neither re-read ran"]

    cfg = get_config()
    try:
        duration = await probe_duration(source)
    except RuntimeError as exc:
        return [], [f"Could not read the video duration to plan clips: {exc}"]
    clips, notes = plan_clips(segments, duration)
    if not clips:
        return [], notes

    clip_dir = Path(cfg.cache_dir) / "clips" / content_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async def _one(clip: dict) -> dict:
        reason = "; ".join(r for r in clip["reasons"] if r)
        entry = {
            "clip_start": _clock(clip["start"]),
            "clip_end": _clock(clip["end"]),
            "reason": reason,
        }
        key = f"{clip['start']:.3f}-{clip['end']:.3f}|{thinking_level}|{instruction}"
        if use_cache:
            cached = cache_load(content_id, "video_detail_read", cfg.default_model, instruction=key)
            if cached:
                return {**entry, **cached, "cached": True}
        out_path: Path | None = None
        async with semaphore:
            try:
                # A unique name per call: two concurrent reads of the same video
                # must not overwrite (ffmpeg -y) or unlink each other's clip.
                # Re-create the directory first: a concurrent call that just
                # finished may have removed it while it was empty.
                clip_dir.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(
                    prefix=f"clip-{clip['start']:.0f}-{clip['end']:.0f}-", suffix=".mp4", dir=clip_dir,
                )
                os.close(fd)
                out_path = Path(name)
                await trim_clip(source, clip["start"], clip["end"], out_path)
                reading = await _read_clip(
                    out_path, reason=reason, instruction=instruction, thinking_level=thinking_level,
                )
            except Exception as exc:
                logger.warning("High-resolution read failed for %s: %s", entry, exc)
                return {**entry, "error": str(exc)}
            finally:
                if out_path is not None:
                    out_path.unlink(missing_ok=True)
        data = reading.model_dump(mode="json")
        if use_cache:
            cache_save(content_id, "video_detail_read", cfg.default_model, data, instruction=key)
        return {**entry, **data}

    try:
        readings = await asyncio.gather(*[_one(c) for c in clips])
    finally:
        try:
            clip_dir.rmdir()  # only succeeds when empty; another call may still use it
        except OSError:
            pass
    return list(readings), notes
