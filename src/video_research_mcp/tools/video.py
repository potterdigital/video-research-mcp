"""Video analysis tools — 3 single-video tools on a FastMCP sub-server.

Batch analysis lives in video_batch.py, registered via side-effect import.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal

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
from .video_agentic import agentic_video_block, check_agentic_model
from .video_file import (
    _upload_large_file,
    _video_file_content,
    _video_file_uri,
    _video_mime_type,
)
from .video_url import (
    _extract_video_id,
    _normalize_youtube_url,
    _video_content,
)
from .youtube_download import download_youtube_video
from .video_detail import read_detail_segments
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
    auto_detail: Annotated[bool, Field(
        description="After the main pass, cut each returned detail_segment out of "
        "the video with ffmpeg and read it at media_resolution='high', returning "
        "verbatim on-screen text in detail_readings. Needs ffmpeg, a Gemini 3 "
        "model and the default output schema. Costs extra only for the seconds "
        "re-read."
    )] = False,
    processing: Annotated[Literal["static", "agentic"], Field(
        description="'static' (default) samples the whole video at a fixed frame "
        "rate. 'agentic' lets Gemini navigate the timeline and load only the "
        "transcript, frames or audio it needs: far fewer tokens and faster on long "
        "videos. Agentic needs gemini-3.8/3.7/3.6-flash or gemini-3.5-flash-lite, "
        "always uploads local files, and does not accept start_offset, end_offset, "
        "fps or strict_contract. Combines with media_resolution and auto_detail."
    )] = "static",
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
        auto_detail: Re-read detail_segments at high resolution from local clips.
        processing: 'static' or 'agentic' video processing.

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

    if auto_detail and (output_schema is not None or strict_contract):
        return make_tool_error(ValueError(
            "auto_detail needs the default output schema: it reads the detail_segments "
            "field, which custom output_schema and strict_contract results do not carry."
        ))

    agentic = processing == "agentic"
    if agentic:
        unsupported = [
            name for name, value in (
                ("start_offset", start_offset), ("end_offset", end_offset), ("fps", fps),
            ) if value is not None
        ]
        if unsupported:
            return make_tool_error(ValueError(
                f"processing='agentic' does not accept {', '.join(unsupported)}: Gemini "
                "supports clipping and fps only in static processing. Omit them, or use "
                "processing='static'."
            ))
        if strict_contract:
            return make_tool_error(ValueError(
                "processing='agentic' does not support strict_contract: the strict "
                "pipeline runs on static generateContent requests. Use processing='static'."
            ))
        try:
            check_agentic_model(get_config().default_model)
        except ValueError as exc:
            return make_tool_error(exc)
    elif processing != "static":
        return make_tool_error(ValueError(
            f"processing must be 'static' or 'agentic', got {processing!r}"
        ))

    try:
        cfg = get_config()
        options = resolve_options(
            media_resolution, start_offset, end_offset, fps,
            default_resolution=cfg.video_media_resolution,
            model=cfg.default_model,
        )
        if auto_detail:
            # Validates that 'high' is available on the configured model.
            resolve_options(
                "high", None, None, None,
                default_resolution=cfg.video_media_resolution, model=cfg.default_model,
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
        agentic_video: dict | None = None
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
            if agentic:
                # A YouTube URL works as the agentic uri (verified live); fps
                # from metadata is a static-only setting, so it is dropped.
                contents = None
                agentic_video = agentic_video_block(
                    video_uri,
                    mime_type="video/mp4" if file_uri else "",
                    resolution=applied["media_resolution"],
                )
            else:
                contents = apply_video_options(
                    _video_content(video_uri, instruction), applied, fps_fallback=fps_override
                )
        elif agentic:
            # Agentic is documented with File API uploads only, so local
            # files are always uploaded, even under the inline-size threshold.
            file_uri, content_id = await _video_file_uri(file_path)
            contents = None
            agentic_video = agentic_video_block(
                file_uri,
                mime_type=_video_mime_type(Path(file_path)),
                resolution=options["media_resolution"],
            )
            source_label = file_path
            local_filepath = str(Path(file_path).expanduser().resolve())
            source_mode = "local_file"
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
            if agentic:
                result["processing"] = "agentic"
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
            cache_instruction=instruction + cache_key_suffix(applied, source_mode)
            + ("\n[processing=agentic]" if agentic else ""),
            agentic_video=agentic_video,
        )
        _annotate(result)
        if auto_detail:
            await _attach_detail_readings(
                result,
                source_path=Path(local_filepath) if local_filepath else None,
                content_id=content_id,
                youtube_id=content_id if url else None,
                instruction=instruction,
                use_cache=use_cache,
            )
        if (
            result.get("detail_segments")
            and applied["media_resolution"] != "high"
            and not any("on_screen_text" in r for r in result.get("detail_readings", []))
        ):
            result["detail_hint"] = (
                "On-screen detail in detail_segments may not be fully legible at this "
                "resolution. Re-run with media_resolution='high' and start_offset/end_offset "
                "for each segment."
            )

        # Pre-warm the session context cache for whole-video, default-option
        # requests only, as before: YouTube URLs are skipped by prewarm_cache,
        # and a clipped or high-detail part must not become the session cache.
        # Agentic runs never prewarm: the session cache serves static requests.
        is_default = not agentic and applied["media_resolution"] == "low" and not (
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


async def _attach_detail_readings(
    result: dict,
    *,
    source_path: Path | None,
    content_id: str,
    youtube_id: str | None,
    instruction: str,
    use_cache: bool,
) -> None:
    """Re-read the result's detail_segments at high resolution from local clips.

    Adds ``detail_readings`` (one entry per clip) and, when anything was
    skipped or failed before reading, ``detail_notes``.
    """
    segments = result.get("detail_segments") or []
    result["detail_readings"] = []
    if not segments:
        return
    notes: list[str] = []
    if source_path is None and youtube_id:
        try:
            source_path = await download_youtube_video(youtube_id)
        except Exception as exc:
            notes.append(f"Could not download the video to cut clips: {exc}")
    if source_path is None:
        result["detail_notes"] = notes or ["No local copy of the video to cut clips from"]
        return
    readings, plan_notes = await read_detail_segments(
        source_path,
        segments,
        content_id=content_id,
        instruction=instruction,
        use_cache=use_cache,
    )
    result["detail_readings"] = readings
    if notes or plan_notes:
        result["detail_notes"] = notes + plan_notes


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

    try:
        _, contents, config_kwargs = await prepare_cached_request(session, prompt)
        user_content = contents[-1]
        model = config_kwargs.pop("_model")
        client = GeminiClient.get()

        response = await with_retry(
            lambda: client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        )
        if not response.candidates or response.candidates[0].content is None:
            raise ValueError("Gemini returned no content for this session turn")
        model_content = response.candidates[0].content
        parts = model_content.parts or []
        text = "\n".join(p.text for p in parts if p.text and not getattr(p, "thought", False))
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
