import pytest
import tempfile
from pathlib import Path

from srt_maker.srt_generator import SRTGenerator


class TestSRTGenerator:
    def test_format_timestamp_basic(self):
        generator = SRTGenerator()

        assert generator.format_timestamp(0.0) == "00:00:00,000"
        assert generator.format_timestamp(1.5) == "00:00:01,500"
        assert generator.format_timestamp(60.0) == "00:01:00,000"
        assert generator.format_timestamp(3600.0) == "01:00:00,000"

    def test_format_timestamp_with_milliseconds(self):
        generator = SRTGenerator()

        assert generator.format_timestamp(1.234) == "00:00:01,234"
        assert generator.format_timestamp(0.001) == "00:00:00,001"
        assert generator.format_timestamp(0.999) == "00:00:00,999"

    def test_format_timestamp_large_values(self):
        generator = SRTGenerator()

        assert generator.format_timestamp(3661.5) == "01:01:01,500"
        assert generator.format_timestamp(7200.25) == "02:00:00,250"

    def test_format_timestamp_custom_precision(self):
        generator = SRTGenerator()

        result = generator.format_timestamp(1.5)
        assert "500" in result
        assert result.startswith("00:00:01,")
        assert result == "00:00:01,500"

    def test_generate_srt_empty_segments(self):
        generator = SRTGenerator()

        result = generator.generate_srt([])
        assert result == ""

    def test_generate_srt_single_segment(self):
        generator = SRTGenerator()

        segments = [{"start": 0.0, "end": 2.5, "text": "Hello world"}]

        result = generator.generate_srt(segments)

        expected = """1
00:00:00,000 --> 00:00:02,500
Hello world
"""
        assert result == expected

    def test_generate_srt_multiple_segments(self):
        generator = SRTGenerator()

        segments = [
            {"start": 0.0, "end": 2.5, "text": "First segment"},
            {"start": 2.5, "end": 5.0, "text": "Second segment"},
            {"start": 5.0, "end": 7.5, "text": "Third segment"},
        ]

        result = generator.generate_srt(segments)

        lines = result.split("\n")
        assert lines[0] == "1"
        assert "00:00:00,000 --> 00:00:02,500" in result
        assert "First segment" in result
        assert lines[4] == "2"
        assert "Second segment" in result
        assert "Third segment" in result

    def test_generate_srt_indexing(self):
        generator = SRTGenerator()

        segments = [
            {"start": 0.0, "end": 1.0, "text": "A"},
            {"start": 1.0, "end": 2.0, "text": "B"},
            {"start": 2.0, "end": 3.0, "text": "C"},
        ]

        result = generator.generate_srt(segments)

        assert result.startswith("1\n00:00:00,000")
        assert "3\n00:00:02,000" in result

    def test_generate_srt_missing_text(self):
        generator = SRTGenerator()

        segments = [{"start": 0.0, "end": 1.0}]

        result = generator.generate_srt(segments)

        assert "1\n00:00:00,000 --> 00:00:01,000\n\n" in result

    def test_generate_srt_whitespace_in_text(self):
        generator = SRTGenerator()

        segments = [{"start": 0.0, "end": 1.0, "text": "  Hello world  "}]

        result = generator.generate_srt(segments)

        assert "Hello world" in result
        assert "  Hello world  " not in result

    def test_write_srt(self):
        generator = SRTGenerator()

        segments = [{"start": 0.0, "end": 2.5, "text": "Test subtitle"}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            output_path = f.name

        try:
            generator.write_srt(segments, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert content == generator.generate_srt(segments)

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_write_srt_empty_segments(self):
        generator = SRTGenerator()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            output_path = f.name

        try:
            generator.write_srt([], output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert content == ""

        finally:
            Path(output_path).unlink(missing_ok=True)
