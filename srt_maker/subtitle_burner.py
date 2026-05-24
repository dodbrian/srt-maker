import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)
StyleValue = Union[int, str]

COLOR_ERROR_MESSAGE = (
    "Primary color must be in #RRGGBB format or ASS &HAABBGGRR format"
)
NVENC_HQ_ARGS = [
    "-preset",
    "p6",
    "-tune",
    "hq",
    "-rc",
    "vbr",
    "-cq",
    "23",
    "-b:v",
    "8M",
    "-maxrate",
    "18M",
    "-bufsize",
    "36M",
    "-spatial_aq",
    "1",
    "-temporal_aq",
    "1",
    "-aq-strength",
    "8",
]
UHD_SUBTITLE_STYLE: Dict[str, StyleValue] = {
    "FontSize": 14,
    "MarginV": 120,
    "Outline": 2,
    "Shadow": 0,
    "Alignment": 2,
}


class SubtitleBurner:
    def __init__(self, ffmpeg_path: Optional[str] = None):
        self.ffmpeg_path = ffmpeg_path or "ffmpeg"

    def burn_subtitles(
        self,
        video_path: str,
        srt_path: str,
        output_path: Optional[str] = None,
        font_size: Optional[int] = None,
        bottom_margin: Optional[int] = None,
        primary_color: Optional[str] = None,
        use_gpu: bool = False,
    ) -> str:
        source_video = Path(video_path)
        subtitle_file = Path(srt_path)

        self._validate_input_file(source_video, "Video file")
        self._validate_input_file(subtitle_file, "SRT file")
        self._validate_style_options(
            font_size=font_size,
            bottom_margin=bottom_margin,
        )

        if output_path is None:
            output_file = self._default_output_path(source_video)
        else:
            output_file = Path(output_path)

        self._validate_output_path(source_video, output_file)

        cmd = self._build_command(
            video_path=str(source_video),
            srt_path=str(subtitle_file),
            output_path=str(output_file),
            font_size=font_size,
            bottom_margin=bottom_margin,
            primary_color=primary_color,
            use_gpu=use_gpu,
        )

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Subtitled video saved to {output_file}")
            return str(output_file)
        except subprocess.CalledProcessError as error:
            logger.error(f"Failed to burn subtitles: {error.stderr}")
            raise RuntimeError(
                f"Failed to burn subtitles into {video_path}: {error.stderr}"
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg: "
                "https://ffmpeg.org/download.html"
            )

    def _build_command(
        self,
        video_path: str,
        srt_path: str,
        output_path: str,
        font_size: Optional[int] = None,
        bottom_margin: Optional[int] = None,
        primary_color: Optional[str] = None,
        use_gpu: bool = False,
    ) -> List[str]:
        video_encoding_args = self._build_video_encoding_args(
            output_path, use_gpu=use_gpu
        )

        return [
            self.ffmpeg_path,
            "-i",
            video_path,
            "-vf",
            self._build_subtitles_filter(
                srt_path=srt_path,
                video_path=video_path,
                font_size=font_size,
                bottom_margin=bottom_margin,
                primary_color=primary_color,
            ),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-map_metadata",
            "0",
            "-map_chapters",
            "0",
            *video_encoding_args,
            "-c:a",
            "copy",
            "-sn",
            "-y",
            output_path,
        ]

    def _build_video_encoding_args(
        self, output_path: str, use_gpu: bool = False
    ) -> List[str]:
        video_codec = self._select_video_codec(output_path, use_gpu=use_gpu)
        encoding_args = ["-c:v", video_codec]

        if video_codec == "h264_nvenc":
            encoding_args.extend(NVENC_HQ_ARGS)

        return encoding_args

    def _build_subtitles_filter(
        self,
        srt_path: str,
        video_path: Optional[str] = None,
        font_size: Optional[int] = None,
        bottom_margin: Optional[int] = None,
        primary_color: Optional[str] = None,
    ) -> str:
        escaped_path = self._escape_filter_path(srt_path)
        styles = self._default_style_for_video(video_path)

        if font_size is not None:
            styles["FontSize"] = font_size

        if bottom_margin is not None:
            styles["MarginV"] = bottom_margin

        if primary_color is not None:
            styles["PrimaryColour"] = self._normalize_ass_color(primary_color)

        filter_value = f"subtitles='{escaped_path}'"
        if styles:
            style = ",".join(f"{name}={value}" for name, value in styles.items())
            filter_value += f":force_style='{style}'"

        return filter_value

    def _default_style_for_video(
        self, video_path: Optional[str]
    ) -> Dict[str, StyleValue]:
        if video_path is None:
            return {}

        dimensions = self._probe_video_dimensions(video_path)
        if dimensions is None:
            return {}

        width, height = dimensions
        if width >= 3840 or height >= 2160:
            return UHD_SUBTITLE_STYLE.copy()

        return {}

    def _probe_video_dimensions(
        self, video_path: str
    ) -> Optional[Tuple[int, int]]:
        ffprobe_path = self._ffprobe_path()
        cmd = [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            video_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            logger.warning(f"Could not detect video dimensions with ffprobe: {error}")
            return None

        dimensions = result.stdout.strip()
        if not dimensions:
            return None

        try:
            width_text, height_text = dimensions.split("x", maxsplit=1)
            return int(width_text), int(height_text)
        except ValueError:
            logger.warning(f"Unexpected ffprobe dimension output: {dimensions}")
            return None

    def _ffprobe_path(self) -> str:
        ffmpeg_path = Path(self.ffmpeg_path)
        ffmpeg_name = ffmpeg_path.name
        if ffmpeg_name.startswith("ffmpeg"):
            ffprobe_name = f"ffprobe{ffmpeg_path.suffix}"
            if ffmpeg_path.parent != Path(".") or ffmpeg_path.suffix:
                return str(ffmpeg_path.with_name(ffprobe_name))

            return ffprobe_name

        return "ffprobe"

    def _default_output_path(self, video_path: Path) -> Path:
        return video_path.with_name(f"{video_path.stem}_subtitled{video_path.suffix}")

    def _validate_input_file(self, path: Path, label: str) -> None:
        if not path.exists():
            raise ValueError(f"{label} not found: {path}")

        if not path.is_file():
            raise ValueError(f"{label} is not a file: {path}")

    def _validate_output_path(self, source_video: Path, output_path: Path) -> None:
        if source_video.resolve() == output_path.resolve():
            raise ValueError("Output video path must be different from input video path")

    def _validate_style_options(
        self,
        font_size: Optional[int],
        bottom_margin: Optional[int],
    ) -> None:
        if font_size is not None and font_size <= 0:
            raise ValueError("Font size must be a positive integer")

        if bottom_margin is not None and bottom_margin < 0:
            raise ValueError("Bottom margin must be zero or greater")

    def _select_video_codec(self, output_path: str, use_gpu: bool = False) -> str:
        suffix = Path(output_path).suffix.lower()

        if suffix == ".webm":
            return "libvpx-vp9"

        if suffix in {".ogv", ".ogg"}:
            return "libtheora"

        if use_gpu:
            return "h264_nvenc"

        return "libx264"

    def _escape_filter_path(self, path: str) -> str:
        return (
            Path(path)
            .resolve()
            .as_posix()
            .replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
        )

    def _normalize_ass_color(self, color: str) -> str:
        normalized = color.strip()
        if normalized.upper().startswith("&H"):
            if len(normalized) != 10:
                raise ValueError(COLOR_ERROR_MESSAGE)

            try:
                int(normalized[2:], 16)
            except ValueError as error:
                raise ValueError(COLOR_ERROR_MESSAGE) from error

            return normalized

        if normalized.startswith("#"):
            normalized = normalized[1:]

        if len(normalized) != 6:
            raise ValueError(COLOR_ERROR_MESSAGE)

        try:
            red = normalized[0:2]
            green = normalized[2:4]
            blue = normalized[4:6]
            int(red, 16)
            int(green, 16)
            int(blue, 16)
        except ValueError as error:
            raise ValueError(COLOR_ERROR_MESSAGE) from error

        return f"&H00{blue}{green}{red}"
