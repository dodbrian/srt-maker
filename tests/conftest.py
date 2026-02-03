"""
Test fixtures and configuration for srt_maker tests.
"""

import pytest
import os
from pathlib import Path


@pytest.fixture
def sample_video_path(tmp_path):
    """
    Path to a sample video file for testing.

    In a real scenario, this would be an actual video file.
    For now, we'll create a placeholder.
    """
    video_file = tmp_path / "sample.mp4"
    # Create an empty file as placeholder
    video_file.write_bytes(b"")
    return str(video_file)


@pytest.fixture
def sample_audio_path(tmp_path):
    """
    Path to a sample audio file for testing.
    """
    audio_file = tmp_path / "audio.wav"
    # Create an empty file as placeholder
    audio_file.write_bytes(b"")
    return str(audio_file)


@pytest.fixture
def output_srt_path(tmp_path):
    """
    Path to an output SRT file for testing.
    """
    return str(tmp_path / "output.srt")


@pytest.fixture
def mock_whisper_result():
    """
    Mock whisper transcription result for testing.
    """
    return {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "This is the first segment."},
            {"start": 2.8, "end": 5.2, "text": "This is the second segment."},
            {"start": 5.5, "end": 8.0, "text": "And this is the third segment."},
        ],
        "language": "en",
    }


@pytest.fixture
def sample_transcription():
    """
    Sample transcription data for testing SRT generation.
    """
    return [
        {"start": 0.0, "end": 2.5, "text": "This is the first segment."},
        {"start": 2.8, "end": 5.2, "text": "This is the second segment."},
        {"start": 5.5, "end": 8.0, "text": "And this is the third segment."},
    ]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
