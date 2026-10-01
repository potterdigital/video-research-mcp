"""Tests for per-request video options: media resolution, clipping, fps, YouTube source modes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

import video_research_mcp.config as cfg_mod
from video_research_mcp.models.video import DetailSegment, VideoResult
from video_research_mcp.tools import video_file
from video_research_mcp.tools.video import video_analyze
from video_research_mcp.tools.video_options import (
    apply_video_options,
    cache_key_suffix,
    parse_offset,
    resolve_options,
)
from video_research_mcp.tools.youtube_download import _format_selector, download_youtube_video

FILE_URI = "https://generativelanguage.googleapis.com/v1beta/files/abc"
URL = "https://www.youtube.com/watch?v=GLvFTMtw4Jk"
HIGH = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH


def _content(uri: str = FILE_URI) -> types.Content:
    return types.Content(parts=[
        types.Part(file_data=types.FileData(file_uri=uri)),
        types.Part(text="prompt"),
    ])


def _opts(**kw) -> dict:
    return resolve_options(
        kw.get("media_resolution"), kw.get("start_offset"), kw.get("end_offset"),
        kw.get("fps"), default_resolution=kw.get("default", "low"),
        model=kw.get("model", "gemini-3.5-flash"),
    )


def _video_part(contents: types.Content) -> types.Part:
    return next(p for p in contents.parts if p.file_data is not None)


def _sent_part(mock_gemini_client) -> types.Part:
    call = mock_gemini_client["generate_structured"].call_args
    contents = call.args[0] if call.args else call.kwargs["contents"]
    return _video_part(contents)


def _set_env(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cfg_mod._config = None


class TestParseOffset:
    @pytest.mark.parametrize("raw,expected", [
        ("90", "90s"), ("90s", "90s"), (90, "90s"), (7.5, "7.5s"), ("1:30", "90s"),
        ("90:00", "5400s"), ("01:01:30", "3690s"), ("10:15:30.75", "36930.75s"),
        (None, None), ("", None),
    ])
    def test_accepted_forms(self, raw, expected):
        # Catches: "1:30" or a JSON number reaching Gemini unconverted, and
        # precision loss on long videos (an earlier :g format gave 36930.8s).
        assert parse_offset(raw) == expected

    @pytest.mark.parametrize("raw", ["abc", "1:75", "-5", -5, "1:2:3:4", "1:60:00", True])
    def test_rejects_malformed(self, raw):
        with pytest.raises(ValueError):
            parse_offset(raw)


class TestResolveOptions:
    def test_default_resolution_comes_from_config_value(self):
        assert _opts(default="high")["media_resolution"] == "high"

    def test_rejects_unknown_resolution(self):
        # 'medium' is the same 70 tokens/frame as 'low' for video on Gemini 3.
        with pytest.raises(ValueError, match="'low' or 'high'"):
            _opts(media_resolution="medium")

    def test_accepts_models_prefixed_name(self):
        assert _opts(media_resolution="high", model="models/gemini-3.5-flash")["media_resolution"] == "high"

    def test_high_requires_gemini_3(self):
        # Catches: a raw 400 INVALID_ARGUMENT from per-part resolution on
        # Gemini 2.x instead of a clear message.
        with pytest.raises(ValueError, match="Gemini 3"):
            _opts(media_resolution="high", model="gemini-2.5-flash")
        assert _opts(model="gemini-2.5-flash")["media_resolution"] == "low"

    def test_rejects_end_before_start(self):
        with pytest.raises(ValueError, match="after start_offset"):
            _opts(start_offset="2:00", end_offset="1:00")


class TestApplyVideoOptions:
    def test_default_request_leaves_part_untouched(self):
        # Catches: attaching media_resolution on default requests, which makes
        # every call on a Gemini 2.x model fail with 400 INVALID_ARGUMENT.
        out = apply_video_options(_content(), _opts())
        assert _video_part(out) == _content().parts[0]

    def test_high_sets_resolution_on_video_part_only(self):
        # Catches: 'high' never reaching the request (the original defect).
        out = apply_video_options(_content(), _opts(media_resolution="high"))
        assert _video_part(out).media_resolution.level == HIGH
        assert out.parts[1].media_resolution is None

    def test_sets_clip_and_fps(self):
        out = apply_video_options(_content(), _opts(start_offset="7:45", end_offset="8:00", fps=2))
        vm = _video_part(out).video_metadata
        assert (vm.start_offset, vm.end_offset, vm.fps) == ("465s", "480s", 2)

    def test_metadata_fps_is_fallback_not_override(self):
        explicit = apply_video_options(_content(), _opts(fps=5), fps_fallback=2.0)
        fallback = apply_video_options(_content(), _opts(), fps_fallback=2.0)
        assert _video_part(explicit).video_metadata.fps == 5
        assert _video_part(fallback).video_metadata.fps == 2.0

    def test_does_not_mutate_input(self):
        original = _content()
        apply_video_options(original, _opts(media_resolution="high"))
        assert original.parts[0].media_resolution is None

    def test_cache_key_separates_options_and_sources(self):
        # Catches: a 'high' request served a cached 'low' result, or a result
        # from a 720-capped download served for a 1080 cap.
        keys = {
            cache_key_suffix(_opts(), "download@1080"),
            cache_key_suffix(_opts(media_resolution="high"), "download@1080"),
            cache_key_suffix(_opts(start_offset="10"), "download@1080"),
            cache_key_suffix(_opts(), "download@720"),
            cache_key_suffix(_opts(), "youtube_url"),
        }
        assert len(keys) == 5


class TestDownloader:
    def test_merge_format_caps_height_and_prefers_h264(self):
        fmt = _format_selector(1080, can_merge=True)
        assert fmt.split("/")[0] == "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]"
        assert fmt.endswith("b[ext=mp4]")

    def test_without_ffmpeg_requests_no_merge(self):
        # Catches: requesting video+audio merges without ffmpeg, which leaves
        # unmerged stream files and re-downloads on every call.
        fmt = _format_selector(1080, can_merge=False)
        assert "+" not in fmt
        assert all("ext=mp4" in alt for alt in fmt.split("/"))

    async def test_missing_output_cleans_unmerged_streams(self, tmp_path):
        # Catches: yt-dlp exiting 0 without producing the merged file, leaving
        # .fNNN stream files behind forever.
        tmp_path = tmp_path / "dl"
        tmp_path.mkdir()
        proc = MagicMock(returncode=0, communicate=AsyncMock(return_value=(b"", b"")))

        async def fake_exec(*args, **kwargs):
            (tmp_path / "vid-max1080.f137.mp4").write_bytes(b"v")
            (tmp_path / "vid-max1080.f140.m4a").write_bytes(b"a")
            return proc

        with (
            patch("shutil.which", return_value="/usr/bin/x"),
            patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
            pytest.raises(RuntimeError, match="output file not found"),
        ):
            await download_youtube_video("vid", target_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []

    async def test_without_ffmpeg_uses_separate_cache_name(self, tmp_path):
        # Catches: a low-resolution single-file download being reused later
        # when ffmpeg is available and a 1080 download was asked for.
        (tmp_path / "vid-single.mp4").write_bytes(b"x")

        def which(name):
            return None if name == "ffmpeg" else "/usr/bin/yt-dlp"

        with patch("shutil.which", side_effect=which):
            assert (await download_youtube_video("vid", target_dir=tmp_path)).name == "vid-single.mp4"
        with (
            patch("shutil.which", return_value="/usr/bin/x"),
            patch("asyncio.create_subprocess_exec",
                  side_effect=RuntimeError("would download")),
            pytest.raises(RuntimeError, match="would download"),
        ):
            await download_youtube_video("vid", target_dir=tmp_path)


class TestDownloaderResume:
    async def test_failed_download_keeps_part_files(self, tmp_path):
        # Catches: cleanup deleting yt-dlp's .part files, which makes every
        # retry of a large download start from zero.
        tmp_path = tmp_path / "dl"
        tmp_path.mkdir()
        part = tmp_path / "vid-max1080.f137.mp4.part"
        part.write_bytes(b"partial")
        proc = MagicMock(returncode=1, communicate=AsyncMock(return_value=(b"", b"network")))
        with (
            patch("shutil.which", return_value="/usr/bin/x"),
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(RuntimeError, match="network"),
        ):
            await download_youtube_video("vid", target_dir=tmp_path)
        assert part.exists()


class TestUpload:
    async def test_rejects_files_over_file_api_limit(self, tmp_path):
        f = tmp_path / "big.mp4"
        f.write_bytes(b"x")
        with (
            patch.object(video_file, "FILE_API_MAX_BYTES", 0),
            pytest.raises(ValueError, match="up to 2 GB"),
        ):
            await video_file._upload_large_file(f, "video/mp4", content_hash="big")

    async def test_expired_cached_upload_is_replaced(self, tmp_path, mock_gemini_client):
        # Catches: a cached upload past the File API's 48 h retention raising
        # the API's not-found error and failing every later request.
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")
        video_file._save_upload_cache("v", "files/old-uri", "files/old")
        client = mock_gemini_client["client"]
        client.aio.files.upload = AsyncMock(
            return_value=SimpleNamespace(uri="files/new-uri", name="files/new", state="ACTIVE")
        )
        waits = AsyncMock(side_effect=[Exception("404 NOT_FOUND"), None])
        with patch.object(video_file, "_wait_for_active", waits):
            uri = await video_file._upload_large_file(f, "video/mp4", content_hash="v")
        assert uri == "files/new-uri"


@pytest.fixture()
def no_metadata():
    with patch(
        "video_research_mcp.tools.video._youtube_metadata_pipeline",
        new_callable=AsyncMock, return_value=(None, None),
    ):
        yield


@pytest.fixture()
def fake_download(tmp_path):
    local = tmp_path / "GLvFTMtw4Jk-max1080.mp4"
    local.write_bytes(b"x")
    with (
        patch("video_research_mcp.tools.video.download_youtube_video",
              new_callable=AsyncMock, return_value=local) as dl,
        patch("video_research_mcp.tools.video._upload_large_file",
              new_callable=AsyncMock, return_value=FILE_URI) as up,
    ):
        yield dl, up


class TestVideoAnalyzeSources:
    async def test_auto_low_url_does_not_download(
        self, mock_gemini_client, monkeypatch, no_metadata, fake_download,
    ):
        # Catches: default requests paying for a download that buys nothing
        # (resolution is the default either way).
        _set_env(monkeypatch, YOUTUBE_VIDEO_SOURCE="auto")
        mock_gemini_client["generate_structured"].return_value = VideoResult(title="t")
        result = await video_analyze(url=URL, use_cache=False)
        fake_download[0].assert_not_awaited()
        assert result["video_source"] == "youtube_url"
        assert _sent_part(mock_gemini_client).media_resolution is None

    async def test_auto_high_url_downloads_and_uploads(
        self, mock_gemini_client, monkeypatch, no_metadata, fake_download,
    ):
        # Catches: 'high' sent on a YouTube URL part, where Gemini ignores it.
        _set_env(monkeypatch, YOUTUBE_VIDEO_SOURCE="auto")
        mock_gemini_client["generate_structured"].return_value = VideoResult(title="t")
        result = await video_analyze(url=URL, use_cache=False, media_resolution="high")
        assert result["video_source"] == "download@1080"
        assert fake_download[1].call_args.kwargs["content_hash"] == "GLvFTMtw4Jk-max1080"
        part = _sent_part(mock_gemini_client)
        assert part.file_data.file_uri == FILE_URI
        assert part.media_resolution.level == HIGH

    async def test_no_ffmpeg_download_is_labelled_and_warned(
        self, mock_gemini_client, monkeypatch, no_metadata, tmp_path,
    ):
        # Catches: a single-file (usually 360p) download reported as
        # download@1080, which also keys the cache as if it were 1080p.
        _set_env(monkeypatch, YOUTUBE_VIDEO_SOURCE="auto")
        local = tmp_path / "GLvFTMtw4Jk-single.mp4"
        local.write_bytes(b"x")
        mock_gemini_client["generate_structured"].return_value = VideoResult(title="t")
        with (
            patch("video_research_mcp.tools.video.download_youtube_video",
                  new_callable=AsyncMock, return_value=local),
            patch("video_research_mcp.tools.video._upload_large_file",
                  new_callable=AsyncMock, return_value=FILE_URI),
        ):
            result = await video_analyze(url=URL, use_cache=False, media_resolution="high")
        assert result["video_source"] == "download@single"
        assert "ffmpeg not found" in result["video_source_warning"]

    async def test_auto_clip_on_url_needs_no_download(
        self, mock_gemini_client, monkeypatch, no_metadata, fake_download,
    ):
        # Clipping works on URL parts (measured: 20 s clip 1,823 tokens vs 57,780 unclipped).
        _set_env(monkeypatch, YOUTUBE_VIDEO_SOURCE="auto")
        mock_gemini_client["generate_structured"].return_value = VideoResult(title="t")
        await video_analyze(url=URL, use_cache=False, start_offset="7:45", end_offset="8:00")
        fake_download[0].assert_not_awaited()
        assert _sent_part(mock_gemini_client).video_metadata.start_offset == "465s"

    async def test_download_failure_reports_high_not_applied(
        self, mock_gemini_client, monkeypatch, no_metadata,
    ):
        # Catches: reporting media_resolution 'high' for a run that fell back
        # to a URL part, where it was ignored.
        _set_env(monkeypatch, YOUTUBE_VIDEO_SOURCE="auto")
        mock_gemini_client["generate_structured"].return_value = VideoResult(title="t")
        with patch("video_research_mcp.tools.video.download_youtube_video",
                   new_callable=AsyncMock, side_effect=RuntimeError("HTTP 403")):
            result = await video_analyze(url=URL, use_cache=False, media_resolution="high")
        assert result["video_source"] == "youtube_url_fallback"
        assert result["video_options"]["media_resolution"] == "low"
        assert "HTTP 403" in result["video_source_warning"]
        assert "not applied" in result["video_source_warning"]
        assert _sent_part(mock_gemini_client).media_resolution is None

    async def test_url_mode_high_warns_without_download(
        self, mock_gemini_client, no_metadata, fake_download,
    ):
        mock_gemini_client["generate_structured"].return_value = VideoResult(title="t")
        result = await video_analyze(url=URL, use_cache=False, media_resolution="high")
        fake_download[0].assert_not_awaited()
        assert "not applied" in result["video_source_warning"]

    async def test_download_mode_does_not_prewarm_paid_cache(
        self, mock_gemini_client, monkeypatch, no_metadata, fake_download,
    ):
        # Catches: every default YouTube analyze creating a billed context
        # cache from the uploaded file.
        _set_env(monkeypatch, YOUTUBE_VIDEO_SOURCE="download")
        mock_gemini_client["generate_structured"].return_value = VideoResult(title="t")
        with patch("video_research_mcp.tools.video.prewarm_cache") as prewarm:
            result = await video_analyze(url=URL, use_cache=False)
        assert result["video_source"] == "download@1080"
        assert all(c.args[1] != FILE_URI for c in prewarm.call_args_list)

    async def test_detail_hint_only_below_high(
        self, mock_gemini_client, monkeypatch, no_metadata, fake_download,
    ):
        # Catches: the two-pass workflow losing its signal to re-read at high.
        _set_env(monkeypatch, YOUTUBE_VIDEO_SOURCE="auto")
        mock_gemini_client["generate_structured"].return_value = VideoResult(
            title="t", detail_segments=[DetailSegment(start="07:45", end="08:00", reason="slide")],
        )
        low = await video_analyze(url=URL, use_cache=False)
        high = await video_analyze(url=URL, use_cache=False, media_resolution="high")
        assert "media_resolution='high'" in low["detail_hint"]
        assert "detail_hint" not in high

    async def test_high_on_gemini_2_returns_tool_error(self, mock_gemini_client, monkeypatch):
        _set_env(monkeypatch, GEMINI_MODEL="gemini-2.5-flash")
        result = await video_analyze(url=URL, media_resolution="high")
        assert "Gemini 3" in result["error"]
        mock_gemini_client["generate_structured"].assert_not_called()

    async def test_invalid_offset_returns_tool_error(self, mock_gemini_client):
        result = await video_analyze(url=URL, start_offset="nonsense")
        assert "error" in result
        mock_gemini_client["generate_structured"].assert_not_called()

    async def test_local_file_gets_high(self, mock_gemini_client, tmp_path):
        mock_gemini_client["generate_structured"].return_value = VideoResult(title="t")
        with patch(
            "video_research_mcp.tools.video._video_file_content",
            new_callable=AsyncMock, return_value=(_content(), "hash123", FILE_URI),
        ):
            result = await video_analyze(file_path=str(Path(tmp_path) / "a.mp4"),
                                         use_cache=False, media_resolution="high")
        assert result["video_source"] == "local_file"
        assert _sent_part(mock_gemini_client).media_resolution.level == HIGH
