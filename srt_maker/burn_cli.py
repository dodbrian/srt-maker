import argparse
import logging
import sys

from rich.console import Console

from .subtitle_burner import SubtitleBurner

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Render a video with burned-in subtitles from an SRT file."
    )

    parser.add_argument("video_file", help="Path to the input video file")
    parser.add_argument("srt_file", help="Path to the input SRT subtitle file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output video path (default: <video_name>_subtitled.<ext>)",
        default=None,
    )
    parser.add_argument(
        "--font-size",
        type=positive_int,
        default=None,
        help="Burned subtitle font size",
    )
    parser.add_argument(
        "--bottom-margin",
        type=non_negative_int,
        default=None,
        help="Bottom margin for burned subtitles",
    )
    parser.add_argument(
        "--primary-color",
        default=None,
        help="Primary subtitle color in #RRGGBB or ASS &H... format",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--use-gpu",
        action="store_true",
        default=None,
        help="Force NVIDIA NVENC for video encoding",
    )
    gpu_group.add_argument(
        "--no-gpu",
        action="store_false",
        dest="use_gpu",
        help="Disable GPU detection and use CPU encoding",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        parser.exit(1)

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    try:
        burner = SubtitleBurner()
        output_path = burner.burn_subtitles(
            video_path=args.video_file,
            srt_path=args.srt_file,
            output_path=args.output,
            font_size=args.font_size,
            bottom_margin=args.bottom_margin,
            primary_color=args.primary_color,
            use_gpu=args.use_gpu,
        )
        console.print(f"[green]✓ Subtitled video saved to: {output_path}[/green]")
    except (RuntimeError, ValueError) as error:
        console.print(f"[red]Error: {error}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as error:
        logger.exception("Unexpected error occurred")
        console.print(f"[red]Unexpected error: {error}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
