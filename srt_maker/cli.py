import argparse
import sys
import logging
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)

from .audio_extractor import AudioExtractor
from .transcriber import Transcriber
from .srt_generator import SRTGenerator

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate SRT subtitles from video audio using speech recognition."
    )

    parser.add_argument("video_file", help="Path to the input video file")

    parser.add_argument(
        "-o",
        "--output",
        help="Output SRT file path (default: <video_name>.srt)",
        default=None,
    )

    parser.add_argument(
        "-m",
        "--model",
        choices=Transcriber.MODEL_SIZES,
        default="base",
        help="Whisper model size (default: base)",
    )

    parser.add_argument(
        "-l",
        "--language",
        help="Language code (e.g., en, es, fr). Auto-detect if not specified.",
        default=None,
    )

    parser.add_argument(
        "-p",
        "--precision",
        type=int,
        default=0,
        help="Timestamp precision in milliseconds (default: 0)",
    )

    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Time offset in seconds to add to all timestamps (default: 0.0)",
    )

    parser.add_argument(
        "-d",
        "--device",
        choices=["cpu", "cuda", "auto"],
        default=None,
        help="Device to run the model on (default: auto)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args(argv)


def main():
    args = parse_args()
    setup_logging(args.verbose)

    gpu_detected = Transcriber.detect_gpu()
    if gpu_detected:
        console.print("[green]✓ GPU (CUDA) detected[/green]")
    else:
        console.print("[yellow]⚠ No GPU detected, using CPU[/yellow]")

    video_path = Path(args.video_file)

    if not video_path.exists():
        console.print(f"[red]Error: Video file not found: {args.video_file}[/red]")
        sys.exit(1)

    if not video_path.is_file():
        console.print(f"[red]Error: Not a file: {args.video_file}[/red]")
        sys.exit(1)

    output_path = args.output or str(video_path.with_suffix(".srt"))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            extract_task = progress.add_task(
                "[cyan]Extracting audio from video...", total=None
            )

            audio_extractor = AudioExtractor()
            audio_path = audio_extractor.extract_audio(str(video_path))

            progress.update(extract_task, completed=True)

            model_task = progress.add_task("[cyan]Loading Whisper model...", total=None)

            transcriber = Transcriber(
                model_size=args.model, language=args.language, device=args.device
            )
            transcriber.load_model()

            progress.update(model_task, completed=True)

            transcribe_task = progress.add_task(
                "[cyan]Transcribing audio...", total=None
            )

            segments = transcriber.get_segments(audio_path)

            progress.update(transcribe_task, completed=True)

            generate_task = progress.add_task(
                "[cyan]Generating SRT file...", total=None
            )

            srt_generator = SRTGenerator(timestamp_precision=args.precision)
            srt_generator.write_srt(segments, output_path, time_offset=args.offset)

            progress.update(generate_task, completed=True)

        console.print(f"\n{len(segments)} subtitle segments generated.")
        console.print(f"[green]✓ SRT file saved to: {output_path}[/green]")

        detected_language = transcriber.language or "auto-detected"
        console.print(f"Detected language: {detected_language}")

    except RuntimeError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unexpected error occurred")
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
