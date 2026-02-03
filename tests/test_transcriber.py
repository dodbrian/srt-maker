import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from srt_maker.transcriber import (
    Transcriber,
    DEFAULT_NO_SPEECH_THRESHOLD,
    DEFAULT_LOGPROB_THRESHOLD,
    DEFAULT_MIN_DURATION,
    DEFAULT_MAX_REPETITIONS,
)


class TestTranscriber:
    def test_detect_gpu_returns_bool(self):
        result = Transcriber.detect_gpu()
        assert isinstance(result, bool)

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

    @patch("srt_maker.transcriber.whisper.load_model")
    def test_transcribe_preserves_user_language(self, mock_load_model):
        """Test that user-specified language is preserved even when Whisper detects a different language"""
        mock_model = MagicMock()
        # Simulate Whisper detecting English but user specified German
        mock_model.transcribe.return_value = {"segments": [], "language": "en"}
        mock_load_model.return_value = mock_model

        transcriber = Transcriber(model_size="base", language="de")
        result = transcriber.transcribe("test_audio.wav")

        # User specified 'de', so it should remain 'de' even though Whisper detected 'en'
        assert transcriber.language == "de"
        assert result["language"] == "en"  # Result still contains what Whisper detected

    @patch("srt_maker.transcriber.whisper.load_model")
    def test_transcribe_updates_language_when_not_specified(self, mock_load_model):
        """Test that detected language is stored when user didn't specify one"""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": [], "language": "ja"}
        mock_load_model.return_value = mock_model

        transcriber = Transcriber(model_size="base", language=None)
        result = transcriber.transcribe("test_audio.wav")

        # No language specified, so it should be updated with detected language
        assert transcriber.language == "ja"
        assert result["language"] == "ja"


class TestTranscriberFiltering:
    """Tests for the hallucination filtering functionality"""

    def test_init_default_thresholds(self):
        """Test that default thresholds are set correctly"""
        transcriber = Transcriber(model_size="base")
        assert transcriber.no_speech_threshold == DEFAULT_NO_SPEECH_THRESHOLD
        assert transcriber.logprob_threshold == DEFAULT_LOGPROB_THRESHOLD
        assert transcriber.min_segment_duration == DEFAULT_MIN_DURATION
        assert transcriber.max_repetitions == DEFAULT_MAX_REPETITIONS

    def test_init_custom_thresholds(self):
        """Test that custom thresholds can be set"""
        transcriber = Transcriber(
            model_size="base",
            no_speech_threshold=0.8,
            logprob_threshold=-0.5,
            min_segment_duration=0.5,
            max_repetitions=3,
        )
        assert transcriber.no_speech_threshold == 0.8
        assert transcriber.logprob_threshold == -0.5
        assert transcriber.min_segment_duration == 0.5
        assert transcriber.max_repetitions == 3

    def test_filter_segments_empty_list(self):
        """Test filtering with empty segment list"""
        transcriber = Transcriber(model_size="base")
        result = transcriber.filter_segments([])
        assert result == []

    def test_filter_segments_passes_good_segments(self):
        """Test that segments meeting all criteria are kept"""
        transcriber = Transcriber(model_size="base")
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world",
                "no_speech_prob": 0.1,
                "avg_logprob": -0.5,
            },
            {
                "start": 2.0,
                "end": 4.0,
                "text": "Goodbye world",
                "no_speech_prob": 0.2,
                "avg_logprob": -0.6,
            },
        ]
        result = transcriber.filter_segments(segments)
        assert len(result) == 2

    def test_filter_segments_removes_high_no_speech_prob(self):
        """Test that segments with high no_speech_prob are filtered"""
        transcriber = Transcriber(model_size="base", no_speech_threshold=0.6)
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Good segment",
                "no_speech_prob": 0.3,
                "avg_logprob": -0.5,
            },
            {
                "start": 2.0,
                "end": 4.0,
                "text": "Bad segment - hallucination",
                "no_speech_prob": 0.8,  # Above threshold
                "avg_logprob": -0.5,
            },
        ]
        result = transcriber.filter_segments(segments)
        assert len(result) == 1
        assert result[0]["text"] == "Good segment"

    def test_filter_segments_removes_low_logprob(self):
        """Test that segments with low avg_logprob are filtered"""
        transcriber = Transcriber(model_size="base", logprob_threshold=-1.0)
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Confident segment",
                "no_speech_prob": 0.1,
                "avg_logprob": -0.5,
            },
            {
                "start": 2.0,
                "end": 4.0,
                "text": "Low confidence segment",
                "no_speech_prob": 0.1,
                "avg_logprob": -1.5,  # Below threshold
            },
        ]
        result = transcriber.filter_segments(segments)
        assert len(result) == 1
        assert result[0]["text"] == "Confident segment"

    def test_filter_segments_removes_short_duration(self):
        """Test that very short segments are filtered"""
        transcriber = Transcriber(model_size="base", min_segment_duration=0.1)
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Normal segment",
                "no_speech_prob": 0.1,
                "avg_logprob": -0.5,
            },
            {
                "start": 2.0,
                "end": 2.05,  # Only 50ms - below threshold
                "text": "Micro segment",
                "no_speech_prob": 0.1,
                "avg_logprob": -0.5,
            },
        ]
        result = transcriber.filter_segments(segments)
        assert len(result) == 1
        assert result[0]["text"] == "Normal segment"

    def test_filter_segments_removes_consecutive_repetitions(self):
        """Test that consecutive repeated text is filtered"""
        transcriber = Transcriber(model_size="base", max_repetitions=2)
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Repeated text",
                "no_speech_prob": 0.1,
                "avg_logprob": -0.5,
            },
            {
                "start": 2.0,
                "end": 4.0,
                "text": "Repeated text",  # First repetition - allowed
                "no_speech_prob": 0.1,
                "avg_logprob": -0.5,
            },
            {
                "start": 4.0,
                "end": 6.0,
                "text": "Repeated text",  # Second repetition - filtered
                "no_speech_prob": 0.1,
                "avg_logprob": -0.5,
            },
            {
                "start": 6.0,
                "end": 8.0,
                "text": "Repeated text",  # Third repetition - filtered
                "no_speech_prob": 0.1,
                "avg_logprob": -0.5,
            },
        ]
        result = transcriber.filter_segments(segments)
        assert len(result) == 2  # Only first two should remain

    def test_filter_segments_resets_repetition_count_on_new_text(self):
        """Test that repetition counter resets when text changes"""
        transcriber = Transcriber(model_size="base", max_repetitions=2)
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "First text",
                "no_speech_prob": 0.1,
                "avg_logprob": -0.5,
            },
            {
                "start": 2.0,
                "end": 4.0,
                "text": "First text",  # Repetition 1
                "no_speech_prob": 0.1,
                "avg_logprob": -0.5,
            },
            {
                "start": 4.0,
                "end": 6.0,
                "text": "Different text",  # New text - counter resets
                "no_speech_prob": 0.1,
                "avg_logprob": -0.5,
            },
            {
                "start": 6.0,
                "end": 8.0,
                "text": "Different text",  # Repetition 1 of new text
                "no_speech_prob": 0.1,
                "avg_logprob": -0.5,
            },
        ]
        result = transcriber.filter_segments(segments)
        assert len(result) == 4  # All should remain

    def test_filter_segments_removes_empty_text(self):
        """Test that segments with empty text are filtered"""
        transcriber = Transcriber(model_size="base")
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Has text",
                "no_speech_prob": 0.1,
                "avg_logprob": -0.5,
            },
            {
                "start": 2.0,
                "end": 4.0,
                "text": "",  # Empty text
                "no_speech_prob": 0.1,
                "avg_logprob": -0.5,
            },
            {
                "start": 4.0,
                "end": 6.0,
                "text": "   ",  # Whitespace only
                "no_speech_prob": 0.1,
                "avg_logprob": -0.5,
            },
        ]
        result = transcriber.filter_segments(segments)
        assert len(result) == 1
        assert result[0]["text"] == "Has text"

    def test_filter_segments_handles_missing_metadata(self):
        """Test that segments without metadata use defaults"""
        transcriber = Transcriber(model_size="base")
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "No metadata segment",
                # Missing no_speech_prob and avg_logprob
            },
        ]
        result = transcriber.filter_segments(segments)
        # Should pass because defaults (0.0) are within thresholds
        assert len(result) == 1

    @patch("srt_maker.transcriber.whisper.load_model")
    def test_get_segments_applies_filtering(self, mock_load_model):
        """Test that get_segments applies filtering to results"""
        mock_model = MagicMock()
        mock_result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Good segment",
                    "no_speech_prob": 0.1,
                    "avg_logprob": -0.5,
                },
                {
                    "start": 2.0,
                    "end": 4.0,
                    "text": "Hallucinated segment",
                    "no_speech_prob": 0.9,  # High no_speech_prob
                    "avg_logprob": -0.5,
                },
            ],
            "language": "en",
        }
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model

        transcriber = Transcriber(model_size="base", no_speech_threshold=0.6)
        segments = transcriber.get_segments("test_audio.wav")

        assert len(segments) == 1
        assert segments[0]["text"] == "Good segment"
