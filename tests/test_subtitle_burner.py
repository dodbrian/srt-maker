import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srt_maker.subtitle_burner import NVENC_HQ_ARGS, SubtitleBurner


class TestSubtitleBurner:
    @patch("subprocess.run")
    def test_burn_subtitles_success_with_default_output(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        video_path = tmp_path / "sample.mp4"
        srt_path = tmp_path / "sample.srt"
        video_path.write_bytes(b"")
        srt_path.write_text("", encoding="utf-8")

        burner = SubtitleBurner()
        burner._probe_video_dimensions = MagicMock(return_value=(1920, 1080))

        output_path = burner.burn_subtitles(str(video_path), str(srt_path))

        assert output_path == str(tmp_path / "sample_subtitled.mp4")
        mock_run.assert_called_once()

    def test_burn_subtitles_missing_video_raises_value_error(self, tmp_path):
        srt_path = tmp_path / "sample.srt"
        srt_path.write_text("", encoding="utf-8")

        burner = SubtitleBurner()

        with pytest.raises(ValueError) as exc_info:
            burner.burn_subtitles(str(tmp_path / "missing.mp4"), str(srt_path))

        assert "Video file not found" in str(exc_info.value)

    def test_burn_subtitles_missing_srt_raises_value_error(self, tmp_path):
        video_path = tmp_path / "sample.mp4"
        video_path.write_bytes(b"")

        burner = SubtitleBurner()

        with pytest.raises(ValueError) as exc_info:
            burner.burn_subtitles(str(video_path), str(tmp_path / "missing.srt"))

        assert "SRT file not found" in str(exc_info.value)

    @patch("subprocess.run")
    def test_burn_subtitles_uses_custom_output(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        video_path = tmp_path / "sample.mp4"
        srt_path = tmp_path / "sample.srt"
        output_path = tmp_path / "custom-output.mp4"
        video_path.write_bytes(b"")
        srt_path.write_text("", encoding="utf-8")

        burner = SubtitleBurner()
        burner._probe_video_dimensions = MagicMock(return_value=(1920, 1080))

        result = burner.burn_subtitles(
            str(video_path), str(srt_path), output_path=str(output_path)
        )

        assert result == str(output_path)
        cmd = mock_run.call_args[0][0]
        assert cmd[-1] == str(output_path)

    @patch("subprocess.run")
    def test_burn_subtitles_builds_expected_ffmpeg_command(
        self, mock_run, tmp_path
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        video_path = tmp_path / "sample.mp4"
        srt_path = tmp_path / "sample.srt"
        video_path.write_bytes(b"")
        srt_path.write_text("", encoding="utf-8")

        burner = SubtitleBurner()
        burner._probe_video_dimensions = MagicMock(return_value=(3840, 2160))
        burner.burn_subtitles(str(video_path), str(srt_path))

        cmd = mock_run.call_args[0][0]
        filter_value = cmd[cmd.index("-vf") + 1]

        assert cmd[0] == "ffmpeg"
        assert cmd[1:3] == ["-i", str(video_path)]
        assert "subtitles='" in filter_value
        assert "force_style='" in filter_value
        assert "FontSize=14" in filter_value
        assert "MarginV=120" in filter_value
        assert "Outline=2" in filter_value
        assert "Shadow=0" in filter_value
        assert "Alignment=2" in filter_value
        assert "sample.srt" in filter_value
        assert cmd[cmd.index("-map") + 1] == "0:v:0"
        assert "0:a?" in cmd
        assert "-map_metadata" in cmd
        assert "-map_chapters" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "libx264"
        assert cmd[cmd.index("-c:a") + 1] == "copy"
        assert "-sn" in cmd

    @patch("subprocess.run")
    def test_burn_subtitles_uses_nvenc_when_gpu_requested(
        self, mock_run, tmp_path
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        video_path = tmp_path / "sample.mp4"
        srt_path = tmp_path / "sample.srt"
        video_path.write_bytes(b"")
        srt_path.write_text("", encoding="utf-8")

        burner = SubtitleBurner()
        burner._probe_video_dimensions = MagicMock(return_value=(1920, 1080))
        burner.burn_subtitles(str(video_path), str(srt_path), use_gpu=True)

        cmd = mock_run.call_args[0][0]
        assert cmd[cmd.index("-c:v") + 1] == "h264_nvenc"
        for index in range(0, len(NVENC_HQ_ARGS), 2):
            flag = NVENC_HQ_ARGS[index]
            value = NVENC_HQ_ARGS[index + 1]
            assert cmd[cmd.index(flag) + 1] == value

    @patch("subprocess.run")
    def test_style_flags_affect_subtitles_filter(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        video_path = tmp_path / "sample.mp4"
        srt_path = tmp_path / "sample.srt"
        video_path.write_bytes(b"")
        srt_path.write_text("", encoding="utf-8")

        burner = SubtitleBurner()
        burner._probe_video_dimensions = MagicMock(return_value=(3840, 2160))
        burner.burn_subtitles(
            str(video_path),
            str(srt_path),
            font_size=28,
            bottom_margin=36,
            primary_color="#112233",
        )

        cmd = mock_run.call_args[0][0]
        filter_value = cmd[cmd.index("-vf") + 1]

        assert "force_style='" in filter_value
        assert "FontSize=28" in filter_value
        assert "MarginV=36" in filter_value
        assert "Outline=2" in filter_value
        assert "Shadow=0" in filter_value
        assert "Alignment=2" in filter_value
        assert "PrimaryColour=&H00332211" in filter_value

    def test_build_subtitles_filter_uses_default_uhd_style(self):
        burner = SubtitleBurner()
        burner._probe_video_dimensions = MagicMock(return_value=(3840, 2160))

        filter_value = burner._build_subtitles_filter(
            "/tmp/subtitles.srt",
            video_path="/tmp/video_uhd.mp4",
        )

        assert filter_value.startswith("subtitles='")
        assert "force_style='" in filter_value
        assert "FontSize=14" in filter_value
        assert "MarginV=120" in filter_value
        assert "Outline=2" in filter_value
        assert "Shadow=0" in filter_value
        assert "Alignment=2" in filter_value

    def test_build_subtitles_filter_without_styles_for_non_uhd_video(self):
        burner = SubtitleBurner()
        burner._probe_video_dimensions = MagicMock(return_value=(1920, 1080))

        filter_value = burner._build_subtitles_filter(
            "/tmp/subtitles.srt",
            video_path="/tmp/video_hd.mp4",
        )

        assert filter_value.startswith("subtitles='")
        assert "force_style" not in filter_value

    @patch("subprocess.run")
    def test_default_style_for_video_uses_uhd_preset_for_4k_input(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="3840x2160\n",
            stderr="",
        )
        burner = SubtitleBurner()

        style = burner._default_style_for_video("/tmp/video.mp4")

        assert style == {
            "FontSize": 14,
            "MarginV": 120,
            "Outline": 2,
            "Shadow": 0,
            "Alignment": 2,
        }

    @patch("subprocess.run")
    def test_default_style_for_video_skips_preset_for_hd_input(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="1920x1080\n",
            stderr="",
        )
        burner = SubtitleBurner()

        style = burner._default_style_for_video("/tmp/video.mp4")

        assert style == {}

    def test_ffprobe_path_uses_sibling_binary_for_custom_ffmpeg_path(self):
        burner = SubtitleBurner(ffmpeg_path="/custom/bin/ffmpeg")

        assert burner._ffprobe_path() == "/custom/bin/ffprobe"

    def test_ffprobe_path_preserves_executable_suffix(self):
        burner = SubtitleBurner(ffmpeg_path="C:/ffmpeg/bin/ffmpeg.exe")

        assert burner._ffprobe_path() == "C:/ffmpeg/bin/ffprobe.exe"

    def test_normalize_ass_color_accepts_ass_format(self):
        burner = SubtitleBurner()

        assert burner._normalize_ass_color("&H00FFFFFF") == "&H00FFFFFF"

    def test_normalize_ass_color_rejects_invalid_ass_format(self):
        burner = SubtitleBurner()

        with pytest.raises(ValueError) as exc_info:
            burner._normalize_ass_color("&HXYZ")

        assert "Primary color must be" in str(exc_info.value)

    def test_normalize_ass_color_rejects_invalid_hex(self):
        burner = SubtitleBurner()

        with pytest.raises(ValueError) as exc_info:
            burner._normalize_ass_color("not-a-color")

        assert "Primary color must be" in str(exc_info.value)

    def test_burn_subtitles_rejects_same_input_and_output(self, tmp_path):
        video_path = tmp_path / "sample.mp4"
        srt_path = tmp_path / "sample.srt"
        video_path.write_bytes(b"")
        srt_path.write_text("", encoding="utf-8")

        burner = SubtitleBurner()

        with pytest.raises(ValueError) as exc_info:
            burner.burn_subtitles(
                str(video_path), str(srt_path), output_path=str(video_path)
            )

        assert "Output video path must be different" in str(exc_info.value)

    def test_burn_subtitles_rejects_non_positive_font_size(self, tmp_path):
        video_path = tmp_path / "sample.mp4"
        srt_path = tmp_path / "sample.srt"
        video_path.write_bytes(b"")
        srt_path.write_text("", encoding="utf-8")

        burner = SubtitleBurner()

        with pytest.raises(ValueError) as exc_info:
            burner.burn_subtitles(str(video_path), str(srt_path), font_size=0)

        assert "Font size must be a positive integer" in str(exc_info.value)

    def test_burn_subtitles_rejects_negative_bottom_margin(self, tmp_path):
        video_path = tmp_path / "sample.mp4"
        srt_path = tmp_path / "sample.srt"
        video_path.write_bytes(b"")
        srt_path.write_text("", encoding="utf-8")

        burner = SubtitleBurner()

        with pytest.raises(ValueError) as exc_info:
            burner.burn_subtitles(
                str(video_path), str(srt_path), bottom_margin=-1
            )

        assert "Bottom margin must be zero or greater" in str(exc_info.value)

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_burn_subtitles_ffmpeg_not_found(self, mock_run, tmp_path):
        video_path = tmp_path / "sample.mp4"
        srt_path = tmp_path / "sample.srt"
        video_path.write_bytes(b"")
        srt_path.write_text("", encoding="utf-8")

        burner = SubtitleBurner()

        with pytest.raises(RuntimeError) as exc_info:
            burner.burn_subtitles(str(video_path), str(srt_path))

        assert "ffmpeg not found" in str(exc_info.value)

    @patch("subprocess.run")
    def test_burn_subtitles_ffmpeg_failure_surfaces_stderr(
        self, mock_run, tmp_path
    ):
        video_path = tmp_path / "sample.mp4"
        srt_path = tmp_path / "sample.srt"
        video_path.write_bytes(b"")
        srt_path.write_text("", encoding="utf-8")
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr="subtitles filter failed"
        )

        burner = SubtitleBurner()

        with pytest.raises(RuntimeError) as exc_info:
            burner.burn_subtitles(str(video_path), str(srt_path))

        assert "Failed to burn subtitles" in str(exc_info.value)
        assert "subtitles filter failed" in str(exc_info.value)

    @patch("subprocess.run")
    def test_custom_ffmpeg_path(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        video_path = tmp_path / "sample.mp4"
        srt_path = tmp_path / "sample.srt"
        video_path.write_bytes(b"")
        srt_path.write_text("", encoding="utf-8")

        burner = SubtitleBurner(ffmpeg_path="/custom/path/to/ffmpeg")
        burner.burn_subtitles(str(video_path), str(srt_path))

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/custom/path/to/ffmpeg"

    def test_default_output_path_preserves_container(self):
        burner = SubtitleBurner()

        output_path = burner._default_output_path(Path("/tmp/video.mkv"))

        assert output_path == Path("/tmp/video_subtitled.mkv")

    def test_select_video_codec_uses_webm_safe_codec(self):
        burner = SubtitleBurner()

        assert burner._select_video_codec("/tmp/output.webm") == "libvpx-vp9"

    def test_select_video_codec_uses_ogg_safe_codec(self):
        burner = SubtitleBurner()

        assert burner._select_video_codec("/tmp/output.ogv") == "libtheora"

    def test_select_video_codec_uses_nvenc_when_gpu_requested(self):
        burner = SubtitleBurner()

        assert (
            burner._select_video_codec("/tmp/output.mp4", use_gpu=True)
            == "h264_nvenc"
        )

    def test_build_video_encoding_args_includes_nvenc_quality_defaults(self):
        burner = SubtitleBurner()

        assert burner._build_video_encoding_args(
            "/tmp/output.mp4", use_gpu=True
        ) == ["-c:v", "h264_nvenc", *NVENC_HQ_ARGS]
