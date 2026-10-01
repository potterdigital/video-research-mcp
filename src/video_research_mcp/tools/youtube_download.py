"""YouTube video download via yt-dlp subprocess.

Downloads YouTube videos to a local cache directory for File API upload.
No Python yt-dlp dependency — calls the binary directly.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from ..config import get_config

logger = logging.getLogger(__name__)



def _format_selector(max_height: int, *, can_merge: bool) -> str:
    """yt-dlp format string for the best MP4 at or below ``max_height``.

    With ffmpeg available (``can_merge``), prefers separate H.264 video and
    m4a audio streams merged into one MP4, which is how YouTube serves
    anything above 360p. Without ffmpeg, yt-dlp cannot merge, so only
    single-file MP4 formats are requested; on YouTube these are usually
    360p. A source below ``max_height`` downloads at its native height.
    """
    h = int(max_height)
    single = f"b[height<={h}][ext=mp4]/b[ext=mp4]"
    if not can_merge:
        return single
    return f"bv*[height<={h}][vcodec^=avc1]+ba[ext=m4a]/{single}"


def _download_dir() -> Path:
    """Return the download cache directory, creating it if needed."""
    d = Path(get_config().cache_dir) / "downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def download_youtube_video(
    video_id: str,
    target_dir: Path | None = None,
    max_height: int | None = None,
) -> Path:
    """Download a YouTube video via yt-dlp to local cache.

    Reuses cached download if the file already exists. Downloads at
    ``max_height`` (default ``YOUTUBE_MAX_HEIGHT``, 1080) or the source's
    native resolution when that is lower. Without ffmpeg only single-file
    MP4 formats are available (see ``_format_selector``).

    Args:
        video_id: YouTube video ID (e.g. "dQw4w9WgXcQ").
        target_dir: Optional download destination. Defaults to cache dir.
        max_height: Maximum frame height in pixels. Defaults to config.

    Returns:
        Path to the downloaded .mp4 file.

    Raises:
        RuntimeError: If yt-dlp is not installed or download fails.
    """
    if not shutil.which("yt-dlp"):
        raise RuntimeError(
            "yt-dlp not found. Install it: brew install yt-dlp (macOS) "
            "or pip install yt-dlp"
        )

    if target_dir is None:
        target_dir = _download_dir()
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    if max_height is None:
        max_height = get_config().youtube_max_height
    can_merge = shutil.which("ffmpeg") is not None
    # The height cap (not the actual height) is in the name, so a download
    # made under a lower cap is never reused for a higher one.
    cap = f"max{int(max_height)}" if can_merge else "single"
    output_path = target_dir / f"{video_id}-{cap}.mp4"

    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info("Download cache hit: %s", output_path.name)
        return output_path

    if not can_merge:
        logger.warning(
            "ffmpeg not found: downloading a single-file MP4 for %s, usually 360p "
            "on YouTube. Install ffmpeg for downloads up to %sp.", video_id, max_height,
        )

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "-f", _format_selector(max_height, can_merge=can_merge),
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        url,
    ]

    logger.info("Downloading %s via yt-dlp...", video_id)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        # Keep any .part files: yt-dlp resumes from them on the next attempt.
        output_path.unlink(missing_ok=True)
        err_msg = stderr.decode().strip() if stderr else "unknown error"
        raise RuntimeError(f"yt-dlp failed for {video_id}: {err_msg}")

    if not output_path.exists():
        # Exit 0 without the merged file: the separate streams
        # (<stem>.f137.mp4, <stem>.f140.m4a) were left unmerged. Remove them.
        for leftover in target_dir.glob(f"{output_path.stem}.f*"):
            if not leftover.name.endswith(".part"):
                leftover.unlink(missing_ok=True)
        raise RuntimeError(
            f"yt-dlp exited successfully but output file not found: {output_path}"
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Downloaded %s (%.1f MB)", output_path.name, size_mb)
    return output_path
