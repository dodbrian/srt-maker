import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from srt_maker.transcriber import Transcriber


class TestTranscriber:
    def test_init_valid_model_size(self):
        for size in Transcriber.MODEL_SIZES:
            transcriber = Transcriber(model_size=size)
            assert transcriber.model_size == size

    def test_init_invalid_model_size(self):
        with pytest.raises(ValueError) as exc_info:
            Transcriber(model_size="invalid")

        assert "Invalid model size" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_init_with_language(self):
        transcriber = Transcriber(model_size="base", language="en")
        assert transcriber.language == "en"

    def test_init_with_device(self):
        transcriber = Transcriber(model_size="base", device="cuda")
        assert transcriber.device == "cuda"

    @patch("srt_maker.transcriber.whisper.load_model")
    def test_load_model(self, mock_load_model):
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        transcriber = Transcriber(model_size="base")
        transcriber.load_model()

        mock_load_model.assert_called_once_with("base", device=None)
        assert transcriber.model == mock_model

    @patch("srt_maker.transcriber.whisper.load_model")
    def test_load_model_with_device(self, mock_load_model):
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        transcriber = Transcriber(model_size="tiny", device="cpu")
        transcriber.load_model()

        mock_load_model.assert_called_once_with("tiny", device="cpu")

    @patch("srt_maker.transcriber.whisper.load_model")
    def test_transcribe_loads_model_if_none(self, mock_load_model):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 2.5, "text": "Hello"}],
            "language": "en",
        }
        mock_load_model.return_value = mock_model

        transcriber = Transcriber(model_size="base")
        result = transcriber.transcribe("test_audio.wav")

        mock_load_model.assert_called_once()
        assert result["language"] == "en"
        assert len(result["segments"]) == 1

    @patch("srt_maker.transcriber.whisper.load_model")
    def test_transcribe_with_existing_model(self, mock_load_model):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": [], "language": "es"}
        mock_load_model.return_value = mock_model

        transcriber = Transcriber(model_size="base")
        transcriber.load_model()

        mock_load_model.reset_mock()

        result = transcriber.transcribe("test_audio.wav")

        mock_load_model.assert_not_called()
        assert result["language"] == "es"

    @patch("srt_maker.transcriber.whisper.load_model")
    def test_transcribe_passes_language(self, mock_load_model):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": [], "language": "fr"}
        mock_load_model.return_value = mock_model

        transcriber = Transcriber(model_size="base", language="fr")
        transcriber.transcribe("test_audio.wav")

        mock_model.transcribe.assert_called_once()
        call_args = mock_model.transcribe.call_args
        assert call_args[1]["language"] == "fr"

    @patch("srt_maker.transcriber.whisper.load_model")
    def test_get_segments(self, mock_load_model):
        mock_model = MagicMock()
        mock_result = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "First"},
                {"start": 2.5, "end": 5.0, "text": "Second"},
            ],
            "language": "en",
        }
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model

        transcriber = Transcriber(model_size="base")
        segments = transcriber.get_segments("test_audio.wav")

        assert len(segments) == 2
        assert segments[0]["text"] == "First"
        assert segments[1]["text"] == "Second"
