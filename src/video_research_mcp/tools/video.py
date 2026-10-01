"""Video analysis tools — 3 single-video tools on a FastMCP sub-server.

Batch analysis lives in video_batch.py, registered via side-effect import.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from video_research_mcp.tracing import trace

from ..client import GeminiClient
from ..retry import with_retry
from .. import context_cache
from ..config import get_config
from ..errors import make_tool_error
from ..models.video import SessionInfo, SessionResponse
from ..prompts.video import METADATA_OPTIMIZER, METADATA_PREAMBLE
from ..sessions import session_store
from ..types import ThinkingLevel, VideoFilePath, YouTubeUrl, coerce_json_param
from ..youtube import YouTubeClient
from .video_cache import ensure_session_cache, prewarm_cache, prepare_cached_request
from .video_core import analyze_video
from .video_file import _upload_large_file, _video_file_content, _video_file_uri
from .video_url import (
    _extract_video_id,
    _normalize_youtube_url,
    _video_content,
)
from .youtube_download import download_youtube_video
from .video_options import (
    FpsParam,
    MediaResolutionParam,
    OffsetParam,
    apply_video_options,
    cache_key_suffix,
    resolve_options,
)
from ..contract.pipeline import run_strict_pipeline

logger = logging.getLogger(__name__)
video_server = FastMCP("video")

_SHORT_VIDEO_THRESHOLD = 5 * 60  # 5 minutes
_LONG_VIDEO_THRESHOLD = 30 * 60  # 30 minutes

async def _youtube_metadata_pipeline(
    video_id: str, instruction: str
) -> tuple[str | None, float | None]:
    """Fetch YouTube metadata and build analysis context + fps override.

    Non-fatal: returns (None, None) on any failure so the caller falls back
    to the generic pipeline.

    Returns:
        (metadata_context, fps_override) — context string for the analysis
        prompt and optional fps sampling rate.
    """
    try:
        meta = await YouTubeClient.video_metadata(video_id)
        if not meta.title:
            return None, None
    except Exception:
        logger.debug("YouTube metadata fetch failed for %s", video_id)
        return None, None

    fps_override: float | None = None
    if meta.duration_seconds > 0:
        if meta.duration_seconds < _SHORT_VIDEO_THRESHOLD:
            fps_override = 2.0
        elif meta.duration_seconds > _LONG_VIDEO_THRESHOLD:
            fps_override = 1.0

    tags_str = ", ".join(meta.tags[:10]) if meta.tags else "none"
    desc_excerpt = (meta.description[:200] + "...") if len(meta.description) > 200 else meta.description

    preamble = METADATA_PREAMBLE.format(
        title=meta.title,
        channel=meta.channel_title,
        category=meta.category or "Unknown",
        duration=meta.duration_display,
        tags=tags_str,
    )

    try:
        cfg = get_config()
        optimizer_prompt = METADATA_OPTIMIZER.format(
            title=meta.title,
            channel=meta.channel_title,
            category=meta.category or "Unknown",
            duration=meta.duration_display,
            description_excerpt=desc_excerpt,
            tags=tags_str,
            instruction=instruction,
        )
        optimized = await GeminiClient.generate(
            optimizer_prompt, model=cfg.flash_model, thinking_level="low"
        )
        context = f"{preamble}\n\nOptimized extraction focus: {optimized.strip()}"
    except Exception:
        logger.debug("Flash optimizer failed, using preamble only")
        context = preamble

    return context, fps_override


@video_server.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
@trace(name="video_analyze", span_type="TOOL")
async def video_analyze(
    url: YouTubeUrl | None = None,
    file_path: VideoFilePath | None = None,
    instruction: Annotated[str, Field(
        description="What to analyze — e.g. 'summarize key points', "
        "'extract all CLI commands shown', 'list all recipes and ingredients'"
    )] = "Provide a comprehensive analysis of this video.",
    output_schema: Annotated[dict | None, Field(
        description="Optional JSON Schema for the response. "
        "If omitted, uses default VideoResult schema."
    )] = None,
    thinking_level: ThinkingLevel = "high",
    use_cache: Annotated[bool, Field(description="Use cached results")] = True,
    strict_contract: Annotated[bool, Field(
        description="Enable strict contract pipeline with quality gates, "
        "artifact rendering, and semantic validation. Produces richer output "
        "with strategy report, concept map, and HTML/Markdown artifacts."
    )] = False,
    media_resolution: MediaResolutionParam = None,
    start_offset: OffsetParam = None,
    end_offset: OffsetParam = None,
    fps: FpsParam = None,
) -> dict:
    """Analyze a video (YouTube URL or local file) with any instruction.

    Provide exactly one of url or file_path. Uses Gemini's structured output
    for reliable JSON responses. Pass a custom output_schema to control the
    response shape, or use the default VideoResult schema.

    When strict_contract=True, runs the full contract pipeline: analysis with
    strict Pydantic models, parallel strategy/concept-map generation, artifact
    rendering, and quality gates. Returns richer output but takes longer.

    media_resolution='high' only takes effect on uploaded files, so with
    YOUTUBE_VIDEO_SOURCE=auto (the default) a YouTube URL is downloaded (up
    to YOUTUBE_MAX_HEIGHT, default 1080) and uploaded for 'high' requests
    only. If that fails, or the source mode is 'url', the call runs at the
    default resolution and ``video_source_warning`` says so. Clipping and fps
    work on URL parts directly.

    Args:
        url: YouTube video URL.
        file_path: Path to a local video file.
        instruction: What to analyze or extract from the video.
        output_schema: Optional JSON Schema dict for custom output shape.
        thinking_level: Gemini thinking depth.
        use_cache: Whether to use cached results.
        strict_contract: Run strict contract pipeline with quality gates.
        media_resolution: 'low' or 'high' frame detail.
        start_offset: Clip start ('90', '1:30', '01:01:30').
        end_offset: Clip end.
        fps: Frames sampled per second.

    Returns:
        Dict matching VideoResult schema (default), custom output_schema,
        or strict contract output with analysis, strategy, concept_map, artifacts.
    """
    output_schema = coerce_json_param(output_schema, dict)

    if strict_contract and output_schema is not None:
        return {
            "error": "strict_contract and output_schema are mutually exclusive. "
            "Strict mode uses its own schema (StrictVideoResult); "
            "omit output_schema or set strict_contract=False.",
            "category": "API_INVALID_ARGUMENT",
            "hint": "Remove output_schema when using strict_contract=True.",
            "retryable": False,
        }

    try:
        sources = sum(x is not None for x in (url, file_path))
        if sources == 0:
            raise ValueError("Provide exactly one of: url or file_path")
        if sources > 1:
            raise ValueError("Provide exactly one of: url or file_path — got both")
    except ValueError as exc:
        return make_tool_error(exc)

    try:
        cfg = get_config()
        options = resolve_options(
            media_resolution, start_offset, end_offset, fps,
            default_resolution=cfg.video_media_resolution,
            model=cfg.default_model,
        )
    except ValueError as exc:
        return make_tool_error(exc)

    try:
        metadata_context = None
        local_filepath = ""
        screenshot_dir = ""
        source_warning = ""
        file_uri = ""
        applied = options
        if url:
            clean_url = _normalize_youtube_url(url)
            content_id = _extract_video_id(url)
            source_label = clean_url

            meta_ctx, fps_override = await _youtube_metadata_pipeline(
                content_id, instruction
            )
            if meta_ctx:
                metadata_context = meta_ctx

            # Gemini ignores per-part media resolution on YouTube URL parts,
            # so 'high' needs the video downloaded and uploaded. Clipping and
            # fps work on URL parts and need no download.
            video_uri = clean_url
            source_mode = "youtube_url"
            mode = cfg.youtube_video_source
            if mode == "download" or (mode == "auto" and options["media_resolution"] == "high"):
                try:
                    local_path = await download_youtube_video(content_id)
                    file_uri = await _upload_large_file(
                        local_path, "video/mp4", content_hash=local_path.stem
                    )
                    video_uri = file_uri
                    local_filepath = str(local_path)
                    # Report what was downloaded, not what was asked for.
                    if local_path.stem.endswith("-single"):
                        source_mode = "download@single"
                        source_warning = (
                            "ffmpeg not found: downloaded a single-file MP4 (usually 360p on "
                            "YouTube), so small on-screen text may stay unreadable even at "
                            "'high'. Install ffmpeg for downloads up to "
                            f"{cfg.youtube_max_height}p."
                        )
                    else:
                        source_mode = f"download@{cfg.youtube_max_height}"
                except Exception as exc:
                    logger.warning("Download path failed for %s: %s", content_id, exc)
                    source_mode = "youtube_url_fallback"
                    source_warning = f"Download failed ({exc}); analyzed the YouTube URL instead."
            if not source_mode.startswith("download") and options["media_resolution"] == "high":
                applied = {**options, "media_resolution": "low"}
                source_warning = (
                    f"{source_warning} media_resolution='high' was not applied: Gemini "
                    "ignores it on YouTube URL parts, so this ran at the default resolution."
                ).strip()
            contents = apply_video_options(
                _video_content(video_uri, instruction), applied, fps_fallback=fps_override
            )
        else:
            contents, content_id, file_uri = await _video_file_content(file_path, instruction)
            contents = apply_video_options(contents, options)
            source_label = file_path
            local_filepath = str(Path(file_path).expanduser().resolve())
            source_mode = "local_file"

        def _annotate(result: dict) -> dict:
            result["local_filepath"] = local_filepath
            result["screenshot_dir"] = screenshot_dir
            result["video_source"] = source_mode
            result["video_options"] = applied
            if source_warning:
                result["video_source_warning"] = source_warning
            return result

        if strict_contract:
            result = await run_strict_pipeline(
                contents,
                instruction=instruction,
                content_id=content_id,
                source_label=source_label,
                thinking_level=thinking_level,
                metadata_context=metadata_context,
            )
            return _annotate(result)

        result = await analyze_video(
            contents,
            instruction=instruction,
            content_id=content_id,
            source_label=source_label,
            output_schema=output_schema,
            thinking_level=thinking_level,
            use_cache=use_cache,
            metadata_context=metadata_context,
            local_filepath=local_filepath,
            screenshot_dir=screenshot_dir,
            cache_instruction=instruction + cache_key_suffix(applied, source_mode),
        )
        _annotate(result)
        if result.get("detail_segments") and applied["media_resolution"] != "high":
            result["detail_hint"] = (
                "On-screen detail in detail_segments may not be fully legible at this "
                "resolution. Re-run with media_resolution='high' and start_offset/end_offset "
                "for each segment."
            )

        # Pre-warm the session context cache for whole-video, default-option
        # requests only, as before: YouTube URLs are skipped by prewarm_cache,
        # and a clipped or high-detail part must not become the session cache.
        is_default = applied["media_resolution"] == "low" and not (
            applied["start_offset"] or applied["end_offset"] or applied["fps"] is not None
        )
        if content_id and is_default:
            cache_uri = clean_url if url else file_uri
            if cache_uri:
                prewarm_cache(content_id, cache_uri)

        return result

    except (ValueError, FileNotFoundError) as exc:
        return make_tool_error(exc)
    except Exception as exc:
        return make_tool_error(exc)


async def _download_and_cache(
    video_id: str,
) -> tuple[str, str, str, str, str]:
    """Download YouTube video, upload to File API, and create context cache.

    Args:
        video_id: YouTube video ID.

    Returns:
        (cache_name, model, download_status, file_api_uri, local_filepath) where
        download_status is "downloaded" on success or "failed"/"unavailable".
    """
    try:
        local_path = await download_youtube_video(video_id)
    except Exception as exc:
        status = "unavailable" if "not found" in str(exc).lower() else "failed"
        logger.warning("Download failed for %s: %s", video_id, exc)
        return "", "", status, "", ""

    try:
        # Keyed by file stem ("<id>-max<cap>" or "<id>-single") so an upload
        # of a download made under a different cap is never reused.
        file_uri = await _upload_large_file(
            local_path, "video/mp4", content_hash=local_path.stem
        )
    except Exception as exc:
        logger.warning("File API upload failed for %s: %s", video_id, exc)
        return "", "", "failed", "", str(local_path)

    from google.genai import types

    cfg = get_config()
    try:
        file_part = types.Part(file_data=types.FileData(file_uri=file_uri))
        cache_name = await context_cache.get_or_create(
            video_id, [file_part], cfg.default_model
        )
        if cache_name:
            return cache_name, cfg.default_model, "downloaded", file_uri, str(local_path)
    except Exception:
        logger.debug("Cache creation failed for %s, session will use File API URI", video_id)

    # Cache creation failed but upload succeeded — session can still use the
    # File API URI (uncached but avoids re-fetching YouTube URL each turn)
    return "", "", "downloaded", file_uri, str(local_path)


@video_server.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
@trace(name="video_create_session", span_type="TOOL")
async def video_create_session(
    url: YouTubeUrl | None = None,
    file_path: VideoFilePath | None = None,
    description: Annotated[str, Field(description="Session purpose or focus area")] = "",
    download: Annotated[bool, Field(
        description="Download YouTube video locally for cached multi-turn sessions. "
        "Slower startup but faster and cheaper per turn. "
        "Requires yt-dlp (and ffmpeg for downloads above 360p)."
    )] = False,
) -> dict:
    """Create a persistent session for multi-turn video exploration.

    Provide exactly one of url or file_path. When ``download=True`` and the
    source is YouTube, the video is downloaded via yt-dlp, uploaded to the
    Gemini File API, and context-cached for fast multi-turn use.

    Args:
        url: YouTube video URL.
        file_path: Path to a local video file.
        description: Optional focus area for the session.
        download: Download YouTube video for cached sessions.

    Returns:
        Dict with session_id, status, video_title, source_type, cache/download status,
        and optional local_filepath when a local file is available.
    """
    try:
        sources = sum(x is not None for x in (url, file_path))
        if sources == 0:
            raise ValueError("Provide exactly one of: url or file_path")
        if sources > 1:
            raise ValueError("Provide exactly one of: url or file_path — got both")
    except ValueError as exc:
        return make_tool_error(exc)

    try:
        if url:
            clean_url = _normalize_youtube_url(url)
            source_type = "youtube"
            content_id = ""
            local_filepath = ""
        else:
            uri, content_id = await _video_file_uri(file_path)
            clean_url = uri
            source_type = "local"
            local_filepath = str(Path(file_path).expanduser().resolve())
    except (ValueError, FileNotFoundError) as exc:
        return make_tool_error(exc)

    title = ""
    video_id = ""
    if source_type == "youtube":
        video_id = _extract_video_id(url)
        try:
            meta = await YouTubeClient.video_metadata(video_id)
            title = meta.title
        except Exception:
            logger.debug("YouTube API title fetch failed, falling back to Gemini")

    if not title:
        try:
            title_content = _video_content(
                clean_url,
                "What is the title of this video? Reply with just the title.",
            )
            resp = await GeminiClient.generate(title_content, thinking_level="low")
            title = resp.strip()
        except Exception:
            title = Path(file_path).stem if file_path else ""

    cache_name, cache_model, cache_reason, download_status = "", "", "", ""

    if download and source_type == "youtube":
        cache_name, cache_model, download_status, file_uri, local_filepath = (
            await _download_and_cache(video_id)
        )
        if file_uri:
            # Session URL becomes the File API URI for multi-turn replay
            clean_url = file_uri
    elif source_type == "local" and content_id:
        cache_name, cache_model, cache_reason = await ensure_session_cache(
            content_id, clean_url
        )

    session = session_store.create(
        clean_url, "general",
        video_title=title,
        cache_name=cache_name,
        model=cache_model,
        local_filepath=local_filepath,
    )
    return SessionInfo(
        session_id=session.session_id,
        status="created",
        video_title=title,
        source_type=source_type,
        cache_status="cached" if cache_name else "uncached",
        download_status=download_status,
        cache_reason=cache_reason,
        local_filepath=local_filepath,
    ).model_dump(mode="json")


@video_server.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
@trace(name="video_continue_session", span_type="TOOL")
async def video_continue_session(
    session_id: Annotated[str, Field(min_length=1, description="Session ID from video_create_session")],
    prompt: Annotated[str, Field(min_length=1, description="Follow-up question or instruction")],
) -> dict:
    """Continue analysis within an existing video session.

    Args:
        session_id: Session ID returned by video_create_session.
        prompt: Follow-up question about the video.

    Returns:
        Dict with response text and turn_count.
    """
    session = session_store.get(session_id)
    if session is None:
        return {
            "error": f"Session {session_id} not found or expired",
            "category": "API_NOT_FOUND",
            "hint": "Create a new session with video_create_session",
        }

    from google.genai import types

    use_cache, contents, config_kwargs = await prepare_cached_request(session, prompt)
    user_content = contents[-1]  # last entry is the user message we just built
    model = config_kwargs.pop("_model")

    try:
        client = GeminiClient.get()

        response = await with_retry(
            lambda: client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        )
        parts = response.candidates[0].content.parts if response.candidates else []
        text = "\n".join(p.text for p in parts if p.text and not getattr(p, "thought", False))

        model_content = types.Content(
            role="model",
            parts=[types.Part(text=text)],
        )
        turn = session_store.add_turn(session_id, user_content, model_content)
        from ..weaviate_store import store_session_turn
        await store_session_turn(
            session_id,
            session.video_title,
            turn,
            prompt,
            text,
            local_filepath=session.local_filepath,
        )
        return SessionResponse(response=text, turn_count=turn).model_dump(mode="json")
    except Exception as exc:
        return make_tool_error(exc)


# Register batch tool on video_server (side-effect import + re-export)
from .video_batch import video_batch_analyze  # noqa: F401, E402
