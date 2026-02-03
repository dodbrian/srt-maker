import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class SRTGenerator:
    def __init__(self, timestamp_precision: int = 0, min_display_duration: float = 0.0):
        self.timestamp_precision = timestamp_precision
        self.min_display_duration = min_display_duration

    def format_timestamp(self, seconds: float) -> str:
        total_seconds = seconds
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        secs = total_seconds % 60
        milliseconds = secs % 1
        secs = int(secs)

        if self.timestamp_precision > 0:
            ms_total_digits = 3 + self.timestamp_precision
            ms_format = f"{{0:0{ms_total_digits}d}}"
            ms_str = ms_format.format(int(milliseconds * 1000))
        else:
            ms_str = f"{int(milliseconds * 1000):03d}"

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms_str}"

    def generate_srt(
        self, segments: List[Dict[str, Any]], time_offset: float = 0.0
    ) -> str:
        if not segments:
            return ""

        srt_lines = []

        for index, segment in enumerate(segments, start=1):
            start_time = max(0.0, segment.get("start", 0.0) + time_offset)
            end_time = max(0.0, segment.get("end", 0.0) + time_offset)
            text = segment.get("text", "").strip()

            # Apply minimum display duration
            if self.min_display_duration > 0.0:
                desired_end = start_time + self.min_display_duration

                # Cap at the start of the next segment to avoid overlap
                if index < len(segments):
                    next_start = max(
                        0.0, segments[index].get("start", 0.0) + time_offset
                    )
                    end_time = max(end_time, min(desired_end, next_start))
                else:
                    # Last segment can extend freely
                    end_time = max(end_time, desired_end)

            start_str = self.format_timestamp(start_time)
            end_str = self.format_timestamp(end_time)

            if text:
                srt_lines.append(f"{index}")
                srt_lines.append(f"{start_str} --> {end_str}")
                srt_lines.append(text)
                srt_lines.append("")

        return "\n".join(srt_lines)

    def write_srt(
        self, segments: List[Dict[str, Any]], output_path: str, time_offset: float = 0.0
    ):
        srt_content = self.generate_srt(segments, time_offset)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logger.info(f"SRT file written to {output_path}")
