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

        assert result == ""

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

    def test_min_display_duration_extends_short_segment(self):
        """Test that short segments are extended to minimum display duration"""
        generator = SRTGenerator(min_display_duration=2.0)

        segments = [{"start": 0.0, "end": 0.5, "text": "Hi"}]

        result = generator.generate_srt(segments)

        # Should be extended from 0.5s to 2.0s
        assert "00:00:00,000 --> 00:00:02,000" in result

    def test_min_display_duration_no_effect_on_long_segment(self):
        """Test that long segments are not shortened by min_display_duration"""
        generator = SRTGenerator(min_display_duration=2.0)

        segments = [{"start": 0.0, "end": 5.0, "text": "This is a longer segment"}]

        result = generator.generate_srt(segments)

        # Should remain 5.0s
        assert "00:00:00,000 --> 00:00:05,000" in result

    def test_min_display_duration_prevents_overlap(self):
        """Test that extending duration doesn't cause overlap with next segment"""
        generator = SRTGenerator(min_display_duration=2.0)

        segments = [
            {"start": 0.0, "end": 0.5, "text": "Hello"},
            {"start": 1.0, "end": 1.2, "text": "World"},
        ]

        result = generator.generate_srt(segments)

        # First segment should be capped at 1.0s (start of next segment)
        assert "00:00:00,000 --> 00:00:01,000" in result
        # Second segment can extend to 2.0s (1.0 + 2.0 = 3.0 - 1.0 start = 2.0 duration)
        assert "00:00:01,000 --> 00:00:03,000" in result

    def test_min_display_duration_last_segment_extends_freely(self):
        """Test that the last segment can extend beyond without constraint"""
        generator = SRTGenerator(min_display_duration=3.0)

        segments = [
            {"start": 0.0, "end": 2.0, "text": "First"},
            {"start": 5.0, "end": 5.5, "text": "Last"},
        ]

        result = generator.generate_srt(segments)

        # Last segment should extend to full 3.0s duration
        assert "00:00:05,000 --> 00:00:08,000" in result

    def test_min_display_duration_zero_has_no_effect(self):
        """Test that min_display_duration=0.0 keeps original timing"""
        generator = SRTGenerator(min_display_duration=0.0)

        segments = [{"start": 0.0, "end": 0.3, "text": "Short"}]

        result = generator.generate_srt(segments)

        # Should keep original 0.3s duration
        assert "00:00:00,000 --> 00:00:00,300" in result

    def test_min_display_duration_with_time_offset(self):
        """Test that min_display_duration works correctly with time_offset"""
        generator = SRTGenerator(min_display_duration=2.0)

        segments = [
            {"start": 0.0, "end": 0.5, "text": "Hello"},
            {"start": 1.0, "end": 1.2, "text": "World"},
        ]

        result = generator.generate_srt(segments, time_offset=5.0)

        # With offset of 5.0s, first segment: 5.0 -> min(5.0+2.0, 6.0) = 6.0
        assert "00:00:05,000 --> 00:00:06,000" in result
        # Second segment: 6.0 -> 6.0+2.0 = 8.0
        assert "00:00:06,000 --> 00:00:08,000" in result

    def test_min_display_duration_consecutive_short_segments(self):
        """Test multiple consecutive short segments with gaps"""
        generator = SRTGenerator(min_display_duration=1.5)

        segments = [
            {"start": 0.0, "end": 0.3, "text": "One"},
            {"start": 0.8, "end": 1.0, "text": "Two"},
            {"start": 1.5, "end": 1.7, "text": "Three"},
        ]

        result = generator.generate_srt(segments)

        # First: 0.0 -> min(1.5, 0.8) = 0.8
        assert "00:00:00,000 --> 00:00:00,800" in result
        # Second: 0.8 -> min(0.8+1.5=2.3, 1.5) = 1.5
        assert "00:00:00,800 --> 00:00:01,500" in result
        # Third: 1.5 -> 1.5+1.5 = 3.0 (last segment)
        assert "00:00:01,500 --> 00:00:03,000" in result
