"""Tests for auto_detail: planning, trimming and high-resolution re-reads of detail segments."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from google.genai import types

import video_research_mcp.config as cfg_mod
from video_research_mcp.models.video import DetailSegment, VideoResult
from video_research_mcp.tools import video_detail
from video_research_mcp.tools.video import video_analyze
from video_research_mcp.tools.video_detail import (
    DetailReading,
    plan_clips,
    probe_duration,
    read_detail_segments,
    trim_clip,
)

HIGH = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
URL = "https://www.youtube.com/watch?v=GLvFTMtw4Jk"
HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")


def _seg(start, end, reason="slide"):
    return {"start": start, "end": end, "reason": reason}


@pytest.fixture()
def sample_video(tmp_path) -> Path:
    """A real 12 second H.264 + AAC file, so trimming runs the actual ffmpeg path."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not installed")
    out = tmp_path / "sample.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=12:size=320x240:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(out),
    ], check=True)
    return out


@pytest.fixture()
def clip_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_CACHE_DIR", str(tmp_path / "cache"))
    cfg_mod._config = None
    yield tmp_path / "cache"
    cfg_mod._config = None


class TestPlanClips:
    def test_pads_each_segment(self):
        # Catches: reading exactly the model's timestamps, which drift by a few
        # seconds (it reported 07:47 for a slide that appears at 07:45).
        clips, _ = plan_clips([_seg("07:47", "08:04")], duration=None)
        assert (clips[0]["start"], clips[0]["end"]) == (464.0, 487.0)

    def test_clamps_to_zero_and_duration(self):
        clips, _ = plan_clips([_seg("00:01", "00:09")], duration=10.0)
        assert (clips[0]["start"], clips[0]["end"]) == (0.0, 10.0)

    def test_merges_overlapping_after_padding(self):
        # Catches: paying twice for the same seconds when two flagged ranges touch.
        clips, _ = plan_clips([_seg("01:00", "01:10", "a"), _seg("01:14", "01:20", "b")], duration=None)
        assert len(clips) == 1
        assert clips[0]["reasons"] == ["a", "b"]

    def test_skips_unusable_times_with_a_note(self):
        clips, notes = plan_clips([_seg("soon", "later"), _seg("02:00", "01:00")], duration=None)
        assert clips == []
        assert len(notes) == 2

    def test_caps_length_and_count(self):
        # Catches: an unbounded high-resolution bill from a long or noisy first pass.
        many = [_seg(f"{m:02d}:00", f"{m:02d}:10") for m in range(10)]
        clips, notes = plan_clips(many, duration=None, max_segments=3)
        assert len(clips) == 3
        long, notes2 = plan_clips([_seg("00:00", "10:00")], duration=None, max_seconds=60)
        assert long[0]["end"] - long[0]["start"] == 60
        assert notes and notes2

    def test_segment_past_the_end_is_noted_not_dropped_silently(self):
        # Catches: a flagged range beyond the video's duration vanishing with
        # no explanation in detail_notes.
        clips, notes = plan_clips([_seg("05:00", "05:10")], duration=60.0)
        assert clips == []
        assert "starts after the end of the video" in notes[0]

    def test_truncation_drops_reasons_no_longer_in_the_clip(self):
        # Catches: a reason reported for a clip that was cut before its range,
        # so the reading claims to cover content it never saw.
        clips, _ = plan_clips(
            [_seg("00:10", "01:00", "early"), _seg("01:02", "03:00", "late")],
            duration=None, max_seconds=60,
        )
        assert (clips[0]["start"], clips[0]["end"]) == (7.0, 67.0)
        assert clips[0]["reasons"] == ["early", "late"]
        clips, _ = plan_clips(
            [_seg("00:10", "01:00", "early"), _seg("01:02", "03:00", "late")],
            duration=None, max_seconds=50,
        )
        assert clips[0]["reasons"] == ["early"]


def test_inline_limit_fits_the_request_cap_after_base64():
    # Catches: an inline clip whose base64 body alone passes Gemini's 20 MB
    # request limit (15 MB encodes to 20 MB), failing the read with a 400.
    encoded = -(-video_detail.INLINE_MAX_BYTES // 3) * 4
    assert encoded + 2_000_000 <= 20_000_000


async def test_run_kills_a_command_that_outlives_its_timeout():
    # Catches: a hung ffmpeg or ffprobe blocking the tool call forever.
    with pytest.raises(RuntimeError, match="did not finish within"):
        await video_detail._run(["sleep", "5"], timeout=0.2)


@needs_ffmpeg
class TestTrim:
    async def test_trim_produces_requested_length(self, sample_video, tmp_path):
        # Catches: stream-copy style cuts that start at the previous keyframe
        # and so return the wrong seconds.
        out = await trim_clip(sample_video, 4.0, 7.0, tmp_path / "c.mp4")
        assert abs(await probe_duration(out) - 3.0) < 0.25

    async def test_trim_failure_leaves_no_file(self, tmp_path):
        bad = tmp_path / "not-a-video.mp4"
        bad.write_bytes(b"nope")
        with pytest.raises(RuntimeError, match="ffmpeg could not cut"):
            await trim_clip(bad, 0, 1, tmp_path / "c.mp4")
        assert not (tmp_path / "c.mp4").exists()


@needs_ffmpeg
class TestReadDetailSegments:
    async def test_reads_each_clip_inline_at_high(self, sample_video, clip_cache_dir, mock_gemini_client):
        # Catches: re-reads sent at the default resolution, or the whole video
        # uploaded instead of the short clip.
        mock_gemini_client["generate_structured"].return_value = DetailReading(on_screen_text="A. note")
        readings, notes = await read_detail_segments(
            sample_video, [_seg("00:04", "00:06")], content_id="vid", instruction="i", use_cache=False,
        )
        assert readings[0]["on_screen_text"] == "A. note"
        assert (readings[0]["clip_start"], readings[0]["clip_end"]) == ("00:01", "00:09")
        contents = mock_gemini_client["generate_structured"].call_args.args[0]
        video = contents.parts[0]
        assert video.inline_data is not None and video.inline_data.mime_type == "video/mp4"
        assert video.media_resolution.level == HIGH
        assert contents.parts[1].text  # prompt after the video, per Gemini's video guidance
        assert not (clip_cache_dir / "clips" / "vid").exists()  # clips and their dir cleaned up

    async def test_one_failed_clip_does_not_sink_the_rest(self, sample_video, clip_cache_dir, mock_gemini_client):
        mock_gemini_client["generate_structured"].side_effect = [
            RuntimeError("quota"), DetailReading(on_screen_text="ok"),
        ]
        readings, _ = await read_detail_segments(
            sample_video, [_seg("00:00", "00:01"), _seg("00:08", "00:10")],
            content_id="vid", instruction="i", use_cache=False,
        )
        assert sorted("error" in r for r in readings) == [False, True]

    async def test_concurrent_reads_of_one_video_use_distinct_clip_files(
        self, sample_video, clip_cache_dir, mock_gemini_client,
    ):
        # Catches: two calls for the same video and range writing the same
        # clip path, so one ffmpeg -y overwrites or one finally deletes the
        # other's clip mid-read.
        mock_gemini_client["generate_structured"].return_value = DetailReading(on_screen_text="x")
        paths: list[Path] = []
        real_trim = video_detail.trim_clip

        async def recording_trim(source, start, end, out_path):
            paths.append(out_path)
            return await real_trim(source, start, end, out_path)

        args = dict(content_id="vid", instruction="i", use_cache=False)
        with patch.object(video_detail, "trim_clip", recording_trim):
            results = await asyncio.gather(
                read_detail_segments(sample_video, [_seg("00:04", "00:06")], **args),
                read_detail_segments(sample_video, [_seg("00:04", "00:06")], **args),
            )
        assert len(paths) == 2 and paths[0] != paths[1]
        assert all("error" not in r[0][0] for r in results)
        assert not (clip_cache_dir / "clips" / "vid").exists()

    async def test_ffmpeg_timeout_is_a_clip_error_and_leaves_no_file(
        self, sample_video, clip_cache_dir, mock_gemini_client, monkeypatch,
    ):
        # Catches: a stalled encode hanging the call or leaving a partial clip.
        monkeypatch.setattr(video_detail, "FFMPEG_BASE_TIMEOUT_SECONDS", 0.0)
        monkeypatch.setattr(video_detail, "FFMPEG_SECONDS_PER_CLIP_SECOND", 0.0)
        readings, _ = await read_detail_segments(
            sample_video, [_seg("00:04", "00:06")], content_id="vid", instruction="i", use_cache=False,
        )
        assert "ffmpeg did not finish within" in readings[0]["error"]
        mock_gemini_client["generate_structured"].assert_not_awaited()
        assert not (clip_cache_dir / "clips" / "vid").exists()

    async def test_ffprobe_timeout_becomes_a_note(self, sample_video, clip_cache_dir, monkeypatch):
        # Catches: a hung ffprobe raising out of auto_detail and discarding the
        # main analysis result.
        monkeypatch.setattr(video_detail, "FFPROBE_TIMEOUT_SECONDS", 0.0)
        readings, notes = await read_detail_segments(
            sample_video, [_seg("00:04", "00:06")], content_id="vid", instruction="i",
        )
        assert readings == []
        assert "ffprobe did not finish within" in notes[0]

    async def test_cache_key_includes_thinking_level(
        self, sample_video, clip_cache_dir, mock_gemini_client,
    ):
        # Catches: a low-thinking reading served from cache for a high-thinking request.
        mock_gemini_client["generate_structured"].return_value = DetailReading(on_screen_text="x")
        args = dict(content_id="vid", instruction="i", use_cache=True)
        await read_detail_segments(sample_video, [_seg("00:04", "00:06")], thinking_level="low", **args)
        again, _ = await read_detail_segments(
            sample_video, [_seg("00:04", "00:06")], thinking_level="high", **args,
        )
        assert "cached" not in again[0]
        assert mock_gemini_client["generate_structured"].await_count == 2

    async def test_cached_reading_skips_gemini(self, sample_video, clip_cache_dir, mock_gemini_client):
        mock_gemini_client["generate_structured"].return_value = DetailReading(on_screen_text="x")
        args = dict(content_id="vid", instruction="i", use_cache=True)
        await read_detail_segments(sample_video, [_seg("00:04", "00:06")], **args)
        again, _ = await read_detail_segments(sample_video, [_seg("00:04", "00:06")], **args)
        assert again[0]["cached"] is True
        assert mock_gemini_client["generate_structured"].await_count == 1


async def test_without_ffmpeg_returns_a_note(tmp_path):
    with patch.object(video_detail.shutil, "which", return_value=None):
        readings, notes = await read_detail_segments(
            tmp_path / "x.mp4", [_seg("00:01", "00:02")], content_id="v", instruction="i",
        )
    assert readings == [] and "ffmpeg" in notes[0]


class TestVideoAnalyzeAutoDetail:
    async def test_rejects_custom_schema(self, mock_gemini_client):
        result = await video_analyze(url=URL, auto_detail=True, output_schema={"type": "object"})
        assert "auto_detail needs the default output schema" in result["error"]

    async def test_rejects_non_gemini_3(self, mock_gemini_client, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
        cfg_mod._config = None
        result = await video_analyze(url=URL, auto_detail=True)
        assert "Gemini 3" in result["error"]

    async def test_youtube_segments_are_downloaded_and_read(
        self, mock_gemini_client, tmp_path, monkeypatch,
    ):
        # Catches: auto_detail doing nothing for YouTube sources analyzed by URL.
        monkeypatch.setenv("YOUTUBE_VIDEO_SOURCE", "auto")
        cfg_mod._config = None
        mock_gemini_client["generate_structured"].return_value = VideoResult(
            title="t", detail_segments=[DetailSegment(start="07:47", end="08:04", reason="slide")],
        )
        local = tmp_path / "GLvFTMtw4Jk-max1080.mp4"
        reading = [{"clip_start": "07:44", "clip_end": "08:07", "reason": "slide", "on_screen_text": "A."}]
        with (
            patch("video_research_mcp.tools.video._youtube_metadata_pipeline",
                  new_callable=AsyncMock, return_value=(None, None)),
            patch("video_research_mcp.tools.video.download_youtube_video",
                  new_callable=AsyncMock, return_value=local) as dl,
            patch("video_research_mcp.tools.video.read_detail_segments",
                  new_callable=AsyncMock, return_value=(reading, [])) as rd,
        ):
            result = await video_analyze(url=URL, use_cache=False, auto_detail=True)
        dl.assert_awaited_once_with("GLvFTMtw4Jk")
        assert rd.call_args.args[0] == local
        assert result["detail_readings"] == reading
        assert "detail_hint" not in result

    async def test_no_segments_means_no_download(self, mock_gemini_client):
        mock_gemini_client["generate_structured"].return_value = VideoResult(title="t")
        with (
            patch("video_research_mcp.tools.video._youtube_metadata_pipeline",
                  new_callable=AsyncMock, return_value=(None, None)),
            patch("video_research_mcp.tools.video.download_youtube_video", new_callable=AsyncMock) as dl,
        ):
            result = await video_analyze(url=URL, use_cache=False, auto_detail=True)
        dl.assert_not_awaited()
        assert result["detail_readings"] == []

    async def test_url_mode_never_downloads_for_auto_detail(self, mock_gemini_client):
        # Catches: auto_detail downloading a YouTube video although
        # YOUTUBE_VIDEO_SOURCE=url promises never to download (conftest sets url).
        mock_gemini_client["generate_structured"].return_value = VideoResult(
            title="t", detail_segments=[DetailSegment(start="01:00", end="01:10", reason="code")],
        )
        with (
            patch("video_research_mcp.tools.video._youtube_metadata_pipeline",
                  new_callable=AsyncMock, return_value=(None, None)),
            patch("video_research_mcp.tools.video.download_youtube_video",
                  new_callable=AsyncMock) as dl,
            patch("video_research_mcp.tools.video.read_detail_segments",
                  new_callable=AsyncMock) as rd,
        ):
            result = await video_analyze(url=URL, use_cache=False, auto_detail=True)
        dl.assert_not_awaited()
        rd.assert_not_awaited()
        assert result["detail_readings"] == []
        assert "YOUTUBE_VIDEO_SOURCE=url forbids downloading" in result["detail_notes"][0]
        assert "detail_hint" in result

    async def test_download_failure_is_reported_not_raised(self, mock_gemini_client, monkeypatch):
        monkeypatch.setenv("YOUTUBE_VIDEO_SOURCE", "auto")
        cfg_mod._config = None
        mock_gemini_client["generate_structured"].return_value = VideoResult(
            title="t", detail_segments=[DetailSegment(start="01:00", end="01:10", reason="code")],
        )
        with (
            patch("video_research_mcp.tools.video._youtube_metadata_pipeline",
                  new_callable=AsyncMock, return_value=(None, None)),
            patch("video_research_mcp.tools.video.download_youtube_video",
                  new_callable=AsyncMock, side_effect=RuntimeError("HTTP 403")),
        ):
            result = await video_analyze(url=URL, use_cache=False, auto_detail=True)
        assert "HTTP 403" in result["detail_notes"][0]
        assert result["detail_readings"] == []
        assert "detail_hint" in result
