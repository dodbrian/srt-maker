import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
import tempfile

from srt_maker.cli import main, parse_args, setup_logging


class TestCLI:
    def test_parse_args_defaults(self):
        args = parse_args(["video.mp4"])

        assert args.video_file == "video.mp4"
        assert args.output is None
        assert args.model == "base"
        assert args.language is None
        assert args.precision == 0
        assert args.offset == 0.0
        assert args.device is None
        assert args.verbose is False

    def test_parse_args_with_options(self):
        args = parse_args(
            [
                "video.mp4",
                "-o",
                "output.srt",
                "-m",
                "tiny",
                "-l",
                "en",
                "-p",
                "100",
                "--offset",
                "1.5",
                "-d",
                "cpu",
                "-v",
            ]
        )

        assert args.video_file == "video.mp4"
        assert args.output == "output.srt"
        assert args.model == "tiny"
        assert args.language == "en"
        assert args.precision == 100
        assert args.offset == 1.5
        assert args.device == "cpu"
        assert args.verbose is True

    def test_parse_args_invalid_model(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(["video.mp4", "-m", "invalid"])

        captured = capsys.readouterr()
        assert "invalid choice" in captured.err

    def test_setup_logging(self):
        import logging

        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        setup_logging(verbose=True)

        logger = logging.getLogger()

        assert logger.level == logging.DEBUG

    def test_setup_logging_not_verbose(self):
        import logging

        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        setup_logging(verbose=False)

        logger = logging.getLogger()

        assert logger.level == logging.INFO

    @patch("srt_maker.cli.AudioExtractor")
    @patch("srt_maker.cli.Transcriber")
    @patch("srt_maker.cli.SRTGenerator")
    def test_main_success(
        self, mock_srt_gen, mock_transcriber_class, mock_audio_extractor_class
    ):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_file:
            video_path = video_file.name

        try:
            with patch("sys.argv", ["srt-maker", video_path]):
                mock_audio_extractor = MagicMock()
                mock_audio_extractor.extract_audio.return_value = "test_audio.wav"
                mock_audio_extractor_class.return_value = mock_audio_extractor

                mock_transcriber = MagicMock()
                mock_transcriber.get_segments.return_value = [
                    {"start": 0.0, "end": 2.5, "text": "Test"}
                ]
                mock_transcriber.language = "en"
                mock_transcriber_class.return_value = mock_transcriber

                mock_srt_generator = MagicMock()
                mock_srt_gen.return_value = mock_srt_generator

                main()

                mock_audio_extractor.extract_audio.assert_called_once()
                mock_transcriber.load_model.assert_called_once()
                mock_transcriber.get_segments.assert_called_once()
                mock_srt_generator.write_srt.assert_called_once()

        finally:
            Path(video_path).unlink(missing_ok=True)

    @patch("sys.argv", ["srt-maker", "nonexistent.mp4"])
    def test_main_file_not_found(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Video file not found" in captured.out

    @patch("srt_maker.cli.AudioExtractor")
    def test_main_runtime_error(self, mock_audio_extractor_class, capsys):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_file:
            video_path = video_file.name

        try:
            with patch("sys.argv", ["srt-maker", video_path]):
                mock_audio_extractor = MagicMock()
                mock_audio_extractor.extract_audio.side_effect = RuntimeError(
                    "ffmpeg not found"
                )
                mock_audio_extractor_class.return_value = mock_audio_extractor

                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 1
                captured = capsys.readouterr()
                assert "ffmpeg not found" in captured.out

        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_help_message(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(["--help"])

        captured = capsys.readouterr()
        assert "Generate SRT subtitles" in captured.out
        assert "video_file" in captured.out
        assert "--output" in captured.out
