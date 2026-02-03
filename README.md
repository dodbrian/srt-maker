# srt-maker

CLI tool to generate SRT subtitles from video audio using speech recognition.

## Features

- Automatic speech recognition using OpenAI Whisper (local, offline)
- Language detection support
- Progress indicators for transcription
- Configurable output parameters
- Timestamp precision control
- Minimum subtitle display duration for better readability

## Requirements

- Python 3.9+
- ffmpeg (installed on system)

## Installation

### Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html

### Install Python Package

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

### Basic Usage

```bash
srt-maker video.mp4
```

This generates `video.srt` in the same directory.

### Options

```bash
srt-maker video.mp4 [OPTIONS]

Options:
  video_file                  Path to the input video file (required)
  -o, --output OUTPUT         Output SRT file path (default: <video_name>.srt)
  -m, --model MODEL           Whisper model size: tiny, base, small, medium, large, large-v1, large-v2, large-v3 (default: base)
  -l, --language LANG         Language code (e.g., en, es, fr). Auto-detect if not specified
  -p, --precision N           Timestamp precision in milliseconds (default: 0)
  -d, --device DEVICE         Device to run the model on: cpu, cuda, auto (default: auto)
  --min-display-duration N    Minimum display duration for subtitles in seconds (default: 0.0 - use actual speech duration)
  --no-speech-threshold N     Filter segments with no_speech_prob above this value (default: 0.6)
  --logprob-threshold N       Filter segments with avg_logprob below this value (default: -1.0)
  --min-duration N            Minimum segment duration in seconds (default: 0.1)
  --max-repetitions N         Max consecutive repetitions of same text (default: 2)
  --offset N                 Time offset in seconds to add to all timestamps (default: 0.0)
  -v, --verbose              Enable verbose logging
  --help                     Show help message
```

### Examples

Generate subtitles with custom output path:
```bash
srt-maker video.mp4 -o subtitles.srt
```

Use tiny model for faster transcription (less accurate):
```bash
srt-maker video.mp4 -m tiny
```

Use large model for better accuracy (slower):
```bash
srt-maker video.mp4 -m large
```

Specify language for better accuracy:
```bash
srt-maker video.mp4 -l en
```

Force CPU usage:
```bash
srt-maker video.mp4 -d cpu
```

Extend short subtitles for better readability (2 second minimum display duration):
```bash
srt-maker video.mp4 --min-display-duration 2.0
```

## Development

### Run Tests

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=srt_maker
```

Run specific test file:
```bash
pytest tests/test_audio_extractor.py
```

### Watch Mode

Run tests in watch mode for continuous feedback during development:
```bash
./test_runner.sh --watch
```

### Linting

Run linting checks:
```bash
pyflakes srt_maker/**/*.py
```

### Running the Test Runner

The `test_runner.sh` script provides automated testing with continuous feedback:

```bash
# Run all tests with coverage
./test_runner.sh

# Skip slow tests
./test_runner.sh --skip-slow

# Watch mode (re-runs on file changes)
./test_runner.sh --watch
```

## Project Structure

```
srt-maker/
├── srt_maker/
│   ├── __init__.py
│   ├── audio_extractor.py    # Audio extraction from video
│   ├── transcriber.py       # Whisper speech recognition
│   ├── srt_generator.py     # SRT file formatting
│   └── cli.py               # CLI interface
├── tests/
│   ├── conftest.py          # Test fixtures
│   ├── test_audio_extractor.py
│   ├── test_transcriber.py
│   ├── test_srt_generator.py
│   └── test_cli.py
├── test_runner.sh           # Automated test runner
└── pyproject.toml
```

## How It Works

1. **Audio Extraction**: Extract audio track from video using ffmpeg
2. **Speech Recognition**: Use OpenAI Whisper to transcribe audio segments
3. **Language Detection**: Automatically detect language (or use specified)
4. **SRT Generation**: Format segments into SRT subtitle format

## License

MIT