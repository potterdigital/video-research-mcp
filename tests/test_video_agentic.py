"""Tests for processing='agentic' on video_analyze (Gemini Interactions API)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import errors as genai_errors

import video_research_mcp.config as cfg_mod
from video_research_mcp.models.video import VideoResult
from video_research_mcp.tools.video import video_analyze
from video_research_mcp.tools.video_agentic import (
    require_all_properties,
    supports_agentic,
)

URL = "https://www.youtube.com/watch?v=GLvFTMtw4Jk"
FILE_URI = "https://generativelanguage.googleapis.com/v1beta/files/abc"


def _interaction(payload: dict | None = None, *, status: str = "completed", errors=None):
    text = json.dumps(payload if payload is not None else {"title": "T", "summary": "S"})
    usage = SimpleNamespace(
        total_tokens=58000, total_input_tokens=120, total_output_tokens=900,
        total_thought_tokens=300, total_tool_use_tokens=56000, total_cached_tokens=0,
    )
    return SimpleNamespace(status=status, output_text=text, usage=usage, errors=errors)


@pytest.fixture()
def interactions():
    """Patch GeminiClient.get() so client.aio.interactions.create is an AsyncMock."""
    client = MagicMock()
    client.aio.interactions.create = AsyncMock(return_value=_interaction())
    with patch("video_research_mcp.client.GeminiClient.get", return_value=client):
        yield client.aio.interactions.create


@pytest.fixture()
def no_metadata():
    with patch(
        "video_research_mcp.tools.video._youtube_metadata_pipeline",
        new_callable=AsyncMock, return_value=(None, None),
    ):
        yield


def _small_video(tmp_path):
    path = tmp_path / "talk.mp4"
    path.write_bytes(b"\x00" * 1024)
    return path


class TestModelGate:
    @pytest.mark.parametrize("model", [
        "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
        "gemini-3.5-flash-lite", "models/gemini-3.8-flash",
    ])
    def test_documented_models_pass(self, model):
        assert supports_agentic(model)

    @pytest.mark.parametrize("model", [
        "gemini-3.1-pro-preview", "gemini-3.5-flash", "gemini-3.6-flash-lite",
        "gemini-2.5-flash", "gemini-flash-latest",
    ])
    def test_undocumented_models_fail(self, model):
        # Catches: a loose prefix match sending agentic requests to models
        # Google does not document it for (billable 400s at best).
        assert not supports_agentic(model)

    async def test_tool_rejects_unsupported_model_before_any_call(
        self, interactions, no_metadata, monkeypatch,
    ):
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
        cfg_mod._config = None
        result = await video_analyze(url=URL, processing="agentic", use_cache=False)
        assert "gemini-3.8-flash" in result["error"]
        interactions.assert_not_awaited()


class TestRejectedCombinations:
    @pytest.mark.parametrize("kwargs, name", [
        ({"start_offset": "1:00"}, "start_offset"),
        ({"end_offset": 90}, "end_offset"),
        ({"fps": 2.0}, "fps"),
    ])
    async def test_static_only_options_rejected(self, interactions, kwargs, name):
        # Catches: clipping/fps silently ignored under agentic (docs: static only).
        result = await video_analyze(url=URL, processing="agentic", **kwargs)
        assert name in result["error"]
        interactions.assert_not_awaited()

    async def test_strict_contract_rejected(self, interactions):
        result = await video_analyze(url=URL, processing="agentic", strict_contract=True)
        assert "strict_contract" in result["error"]
        interactions.assert_not_awaited()


class TestRequestShape:
    async def test_local_file_is_uploaded_and_sent_as_agentic_block(self, tmp_path, interactions):
        # Catches: a small file going inline (undocumented for agentic), the
        # text preceding the video, server-side storage left on, or a missing
        # JSON schema; any of these changes cost, retention or the result shape.
        path = _small_video(tmp_path)
        with patch(
            "video_research_mcp.tools.video_file._upload_large_file",
            new_callable=AsyncMock, return_value=FILE_URI,
        ) as upload, patch("video_research_mcp.tools.video.prewarm_cache") as prewarm:
            result = await video_analyze(
                file_path=str(path), processing="agentic", use_cache=False,
                thinking_level="low", instruction="Find the slides",
            )
        upload.assert_awaited_once()
        prewarm.assert_not_called()
        kwargs = interactions.call_args.kwargs
        video, text = kwargs["input"]
        assert video == {
            "type": "video", "uri": FILE_URI, "processing": "agentic", "mime_type": "video/mp4",
        }
        assert text["type"] == "text"
        assert "Never guess at on-screen text" in text["text"]
        assert text["text"].endswith("User instruction: Find the slides")
        assert kwargs["store"] is False
        assert kwargs["generation_config"] == {"thinking_level": "low"}
        fmt = kwargs["response_format"]
        assert fmt["type"] == "text" and fmt["mime_type"] == "application/json"
        assert set(fmt["schema"]["required"]) == set(VideoResult.model_fields)
        assert kwargs["model"] == "gemini-3.8-flash"
        assert result["processing"] == "agentic"
        assert result["title"] == "T" and result["summary"] == "S"
        assert result["usage"]["total_tokens"] == 58000
        assert result["video_source"] == "local_file"

    async def test_media_resolution_high_is_sent_as_resolution(self, tmp_path, interactions):
        # Catches: media_resolution dropped on the agentic path (docs: independent of processing).
        path = _small_video(tmp_path)
        with patch(
            "video_research_mcp.tools.video_file._upload_large_file",
            new_callable=AsyncMock, return_value=FILE_URI,
        ):
            await video_analyze(
                file_path=str(path), processing="agentic", media_resolution="high",
                use_cache=False,
            )
        assert interactions.call_args.kwargs["input"][0]["resolution"] == "high"

    async def test_custom_output_schema_is_passed_through(self, interactions, no_metadata):
        interactions.return_value = _interaction({"slides": ["a"]})
        schema = {"type": "object", "properties": {"slides": {"type": "array"}}}
        result = await video_analyze(
            url=URL, processing="agentic", output_schema=schema, use_cache=False,
            instruction="List slides",
        )
        kwargs = interactions.call_args.kwargs
        assert kwargs["response_format"]["schema"] == schema
        assert kwargs["input"][1]["text"] == "List slides"
        assert result["slides"] == ["a"]

    def test_require_all_properties_reaches_nested_defs(self):
        # Catches: the measured Interactions behavior (optional properties
        # omitted) returning for nested segment objects.
        schema = require_all_properties(VideoResult.model_json_schema())
        assert set(schema["$defs"]["DetailSegment"]["required"]) == {"start", "end", "reason"}
        assert "required" not in VideoResult.model_json_schema()

    def test_require_all_properties_leaves_a_property_named_properties_intact(self):
        # Catches: the walker treating a properties map as a schema because one
        # property is literally named "properties", which injected a bogus
        # "required" property into the caller's map.
        schema = {
            "type": "object",
            "properties": {
                "properties": {"type": "string"},
                "required": {"type": "array", "items": {"type": "string"}},
                "nested": {
                    "type": "object",
                    "properties": {"properties": {"type": "object", "properties": {"a": {"type": "string"}}}},
                },
                "list": {"type": "array", "items": {"anyOf": [
                    {"type": "object", "properties": {"b": {"type": "string"}}},
                    {"type": "null"},
                ]}},
            },
        }
        out = require_all_properties(schema)
        assert set(out["properties"]) == {"properties", "required", "nested", "list"}
        assert out["properties"]["properties"] == {"type": "string"}
        assert out["required"] == ["properties", "required", "nested", "list"]
        nested = out["properties"]["nested"]
        assert set(nested["properties"]) == {"properties"}
        assert nested["required"] == ["properties"]
        assert nested["properties"]["properties"]["required"] == ["a"]
        assert out["properties"]["list"]["items"]["anyOf"][0]["required"] == ["b"]
        assert "required" not in schema  # input not mutated


class TestYouTubeSource:
    async def test_youtube_url_is_the_agentic_uri(self, interactions, monkeypatch):
        # Catches: needless download+upload of YouTube videos; agentic accepts
        # the URL directly (verified live). Metadata fps is static-only.
        monkeypatch.setenv("YOUTUBE_VIDEO_SOURCE", "auto")
        cfg_mod._config = None
        with patch(
            "video_research_mcp.tools.video._youtube_metadata_pipeline",
            new_callable=AsyncMock, return_value=("META CONTEXT", 2.0),
        ), patch(
            "video_research_mcp.tools.video.download_youtube_video", new_callable=AsyncMock,
        ) as dl:
            result = await video_analyze(url=URL, processing="agentic", use_cache=False)
        dl.assert_not_awaited()
        video, text = interactions.call_args.kwargs["input"]
        assert video == {"type": "video", "uri": URL, "processing": "agentic"}
        assert "META CONTEXT" in text["text"]
        assert result["video_source"] == "youtube_url"
        assert result["video_options"]["fps"] is None

    async def test_high_downloads_and_uploads_like_static(
        self, interactions, no_metadata, monkeypatch, tmp_path,
    ):
        # Catches: 'high' silently ineffective on a URL part under agentic.
        monkeypatch.setenv("YOUTUBE_VIDEO_SOURCE", "auto")
        cfg_mod._config = None
        local = tmp_path / "GLvFTMtw4Jk-max1080.mp4"
        local.write_bytes(b"\x00")
        with patch(
            "video_research_mcp.tools.video.download_youtube_video",
            new_callable=AsyncMock, return_value=local,
        ), patch(
            "video_research_mcp.tools.video._upload_large_file",
            new_callable=AsyncMock, return_value=FILE_URI,
        ):
            result = await video_analyze(
                url=URL, processing="agentic", media_resolution="high", use_cache=False,
            )
        assert interactions.call_args.kwargs["input"][0] == {
            "type": "video", "uri": FILE_URI, "processing": "agentic",
            "mime_type": "video/mp4", "resolution": "high",
        }
        assert result["video_source"] == "download@1080"


class TestCacheSeparation:
    async def test_agentic_and_static_use_different_cache_keys(
        self, interactions, no_metadata, mock_gemini_client,
    ):
        # Catches: a cached static result served for an agentic request or
        # the reverse; also pins the static key so existing caches still hit.
        mock_gemini_client["get"].return_value.aio.interactions.create = interactions
        mock_gemini_client["generate_structured"].return_value = VideoResult(title="static")
        keys: list[str] = []
        with patch(
            "video_research_mcp.tools.video_core.cache_load", return_value=None,
        ), patch(
            "video_research_mcp.tools.video_core.cache_save",
            side_effect=lambda *a, **k: keys.append(k["instruction"]),
        ):
            await video_analyze(url=URL, instruction="x")
            await video_analyze(url=URL, instruction="x", processing="agentic")
        static_key, agentic_key = keys
        assert static_key != agentic_key
        assert "processing" not in static_key
        assert agentic_key.endswith("[processing=agentic]")

    async def test_cache_hit_skips_the_interaction(self, interactions, no_metadata):
        cached = {"title": "cached", "processing": "agentic"}
        with patch("video_research_mcp.tools.video_core.cache_load", return_value=dict(cached)):
            result = await video_analyze(url=URL, processing="agentic")
        interactions.assert_not_awaited()
        assert result["cached"] is True
        assert "usage" not in result


class TestErrors:
    async def test_client_error_maps_to_invalid_argument(self, interactions, no_metadata):
        interactions.side_effect = genai_errors.ClientError(
            400, {"error": {"code": 400, "message": "bad video", "status": "INVALID_ARGUMENT"}},
        )
        result = await video_analyze(url=URL, processing="agentic", use_cache=False)
        assert result["category"] == "API_INVALID_ARGUMENT"
        assert "bad video" in result["error"]

    async def test_quota_error_is_retryable(self, interactions, no_metadata):
        interactions.side_effect = genai_errors.ClientError(
            429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}},
        )
        result = await video_analyze(url=URL, processing="agentic", use_cache=False)
        assert result["category"] == "API_QUOTA_EXCEEDED"
        assert result["retryable"] is True

    async def test_incomplete_interaction_is_an_error_not_an_empty_result(
        self, interactions, no_metadata,
    ):
        # Catches: a budget_exceeded or failed run returned as an empty VideoResult.
        interactions.return_value = _interaction(
            status="budget_exceeded", errors=[SimpleNamespace(message="token budget")],
        )
        result = await video_analyze(url=URL, processing="agentic", use_cache=False)
        assert "budget_exceeded" in result["error"]
        assert "token budget" in result["error"]

    async def test_non_json_output_is_an_error(self, interactions, no_metadata):
        interactions.return_value = SimpleNamespace(
            status="completed", output_text="not json", usage=None, errors=None,
        )
        result = await video_analyze(url=URL, processing="agentic", use_cache=False)
        # Catches: unparseable agentic output returned as an empty VideoResult
        # or reported as a retryable API failure.
        assert "Invalid JSON" in result["error"]
        assert "VideoResult" in result["error"]
        assert result["retryable"] is False
        assert "title" not in result


class TestSdkRetries:
    async def test_transient_429_and_503_are_retried_to_one_success(self, monkeypatch):
        # Catches: generate_agentic relying on SDK retries that are not there.
        # Interactions calls go through google-genai's generated NextGen client
        # (google/genai/_gaos/google_genai.py, _translate_retry_config), which
        # retries 408/409/429/5XX with backoff when http_options carries no
        # retry_options. If an SDK upgrade or a GeminiClient change drops that,
        # the first 503 surfaces as an error and this test fails.
        import httpx

        from video_research_mcp.client import GeminiClient
        from video_research_mcp.tools.video_agentic import generate_agentic

        monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
        cfg_mod._config = None
        statuses = iter([503, 429])
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            status = next(statuses, 200)
            if status != 200:
                return httpx.Response(status, json={"error": {"code": status, "message": "busy"}})
            return httpx.Response(200, json={
                "id": "i1", "status": "completed",
                "steps": [{"type": "model_output",
                           "content": [{"type": "text", "text": '{"title": "T"}'}]}],
            })

        monkeypatch.setattr(GeminiClient, "_clients", {})
        client = GeminiClient.get()
        client._api_client._async_httpx_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )
        raw, _usage = await generate_agentic(
            {"type": "video", "uri": URL, "processing": "agentic"}, "Summarize",
            schema={"type": "object", "properties": {"title": {"type": "string"}}},
        )
        assert json.loads(raw) == {"title": "T"}
        assert len(paths) == 3
        assert all(p.endswith("/interactions") for p in paths)


class TestAutoDetailOnAgentic:
    async def test_agentic_segments_feed_the_high_resolution_reread(
        self, interactions, no_metadata, tmp_path, monkeypatch,
    ):
        # Catches: auto_detail not working on top of an agentic first pass.
        monkeypatch.setenv("YOUTUBE_VIDEO_SOURCE", "auto")
        cfg_mod._config = None
        interactions.return_value = _interaction({
            "title": "t",
            "detail_segments": [{"start": "07:47", "end": "08:04", "reason": "lettered notes"}],
        })
        local = tmp_path / "GLvFTMtw4Jk-max1080.mp4"
        reading = [{"clip_start": "07:44", "clip_end": "08:07", "reason": "lettered notes",
                    "on_screen_text": "A. text"}]
        with patch(
            "video_research_mcp.tools.video.download_youtube_video",
            new_callable=AsyncMock, return_value=local,
        ) as dl, patch(
            "video_research_mcp.tools.video.read_detail_segments",
            new_callable=AsyncMock, return_value=(reading, []),
        ) as rd:
            result = await video_analyze(
                url=URL, processing="agentic", auto_detail=True, use_cache=False,
            )
        dl.assert_awaited_once_with("GLvFTMtw4Jk")
        assert rd.call_args.args[0] == local
        assert rd.call_args.args[1][0]["start"] == "07:47"
        assert result["detail_readings"] == reading
        assert result["processing"] == "agentic"
        assert "detail_hint" not in result
