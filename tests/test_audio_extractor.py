import pytest
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

from srt_maker.audio_extractor import AudioExtractor


class TestAudioExtractor:
    @patch("subprocess.run")
    def test_extract_audio_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        extractor = AudioExtractor()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_file:
            video_path = video_file.name

        try:
            result_path = extractor.extract_audio(video_path)

            expected_audio_path = Path(video_path).with_suffix(".wav")
            assert result_path == str(expected_audio_path)

            mock_run.assert_called_once()
            call_args = mock_run.call_args

            assert call_args[0][0][0] == "ffmpeg"
            assert "-i" in call_args[0][0]
            assert video_path in call_args[0][0]

        finally:
            Path(video_path).unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_audio_custom_output(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        extractor = AudioExtractor()

        with (
            tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_file,
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file,
        ):
            video_path = video_file.name
            audio_path = audio_file.name

        try:
            result_path = extractor.extract_audio(video_path, output_path=audio_path)

            assert result_path == audio_path
            call_args = mock_run.call_args
            assert audio_path in call_args[0][0]

        finally:
            Path(video_path).unlink(missing_ok=True)
            Path(audio_path).unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_audio_custom_sample_rate(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        extractor = AudioExtractor()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_file:
            video_path = video_file.name

        try:
            extractor.extract_audio(video_path, sample_rate=48000)

            call_args = mock_run.call_args
            cmd = call_args[0][0]
            ar_index = cmd.index("-ar") + 1
            assert cmd[ar_index] == "48000"

        finally:
            Path(video_path).unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_audio_custom_channels(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        extractor = AudioExtractor()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_file:
            video_path = video_file.name

        try:
            extractor.extract_audio(video_path, channels=2)

            call_args = mock_run.call_args
            cmd = call_args[0][0]
            ac_index = cmd.index("-ac") + 1
            assert cmd[ac_index] == "2"

        finally:
            Path(video_path).unlink(missing_ok=True)

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_extract_audio_ffmpeg_not_found(self, mock_run):
        extractor = AudioExtractor()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_file:
            video_path = video_file.name

        try:
            with pytest.raises(RuntimeError) as exc_info:
                extractor.extract_audio(video_path)

            assert "ffmpeg not found" in str(exc_info.value)

        finally:
            Path(video_path).unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_audio_ffmpeg_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="ffmpeg: Invalid data"
        )
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr="Invalid data"
        )

        extractor = AudioExtractor()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_file:
            video_path = video_file.name

        try:
            with pytest.raises(RuntimeError) as exc_info:
                extractor.extract_audio(video_path)

            assert "Failed to extract audio" in str(exc_info.value)
            assert "Invalid data" in str(exc_info.value)

        finally:
            Path(video_path).unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_custom_ffmpeg_path(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        extractor = AudioExtractor(ffmpeg_path="/custom/path/to/ffmpeg")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_file:
            video_path = video_file.name

        try:
            extractor.extract_audio(video_path)

            call_args = mock_run.call_args
            assert call_args[0][0][0] == "/custom/path/to/ffmpeg"

        finally:
            Path(video_path).unlink(missing_ok=True)
