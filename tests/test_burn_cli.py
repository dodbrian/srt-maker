from unittest.mock import MagicMock, patch

import pytest

from srt_maker.burn_cli import main, parse_args


class TestBurnCLI:
    def test_parse_args_defaults(self):
        args = parse_args(["video.mp4", "subs.srt"])

        assert args.video_file == "video.mp4"
        assert args.srt_file == "subs.srt"
        assert args.output is None
        assert args.font_size is None
        assert args.bottom_margin is None
        assert args.primary_color is None
        assert args.verbose is False

    def test_parse_args_with_options(self):
        args = parse_args(
            [
                "video.mp4",
                "subs.srt",
                "-o",
                "output.mp4",
                "--font-size",
                "24",
                "--bottom-margin",
                "28",
                "--primary-color",
                "#FFFFFF",
                "-v",
            ]
        )

        assert args.video_file == "video.mp4"
        assert args.srt_file == "subs.srt"
        assert args.output == "output.mp4"
        assert args.font_size == 24
        assert args.bottom_margin == 28
        assert args.primary_color == "#FFFFFF"
        assert args.verbose is True

    def test_help_message(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(["--help"])

        captured = capsys.readouterr()
        assert "Render a video with burned-in subtitles" in captured.out
        assert "video_file" in captured.out
        assert "srt_file" in captured.out

    def test_no_args_shows_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            parse_args([])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Render a video with burned-in subtitles" in captured.out

    def test_parse_args_rejects_non_positive_font_size(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(["video.mp4", "subs.srt", "--font-size", "0"])

        captured = capsys.readouterr()
        assert "positive integer" in captured.err

    def test_parse_args_rejects_negative_bottom_margin(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(["video.mp4", "subs.srt", "--bottom-margin", "-1"])

        captured = capsys.readouterr()
        assert "zero or greater" in captured.err

    @patch("srt_maker.burn_cli.SubtitleBurner")
    def test_main_success(self, mock_burner_class, capsys):
        mock_burner = MagicMock()
        mock_burner.burn_subtitles.return_value = "video_subtitled.mp4"
        mock_burner_class.return_value = mock_burner

        with patch("sys.argv", ["srt-burn", "video.mp4", "subs.srt"]):
            main()

        mock_burner.burn_subtitles.assert_called_once_with(
            video_path="video.mp4",
            srt_path="subs.srt",
            output_path=None,
            font_size=None,
            bottom_margin=None,
            primary_color=None,
        )
        captured = capsys.readouterr()
        assert "Subtitled video saved to: video_subtitled.mp4" in captured.out

    @patch("srt_maker.burn_cli.SubtitleBurner")
    def test_main_runtime_error_exits_with_code_one(
        self, mock_burner_class, capsys
    ):
        mock_burner = MagicMock()
        mock_burner.burn_subtitles.side_effect = RuntimeError("ffmpeg failed")
        mock_burner_class.return_value = mock_burner

        with patch("sys.argv", ["srt-burn", "video.mp4", "subs.srt"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "ffmpeg failed" in captured.out

    @patch("srt_maker.burn_cli.SubtitleBurner")
    def test_main_value_error_exits_with_code_one(self, mock_burner_class, capsys):
        mock_burner = MagicMock()
        mock_burner.burn_subtitles.side_effect = ValueError("SRT file not found")
        mock_burner_class.return_value = mock_burner

        with patch("sys.argv", ["srt-burn", "video.mp4", "subs.srt"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "SRT file not found" in captured.out
