# srt-maker Development Plan

## Overview
Build a CLI tool that generates SRT subtitle files from video audio using speech recognition.

## Requirements
- **Language**: Python
- **Speech Recognition**: OpenAI Whisper (local, offline)
- **Audio Extraction**: ffmpeg
- **Progress Indicators**: Rich library
- **Features**: Language detection, configurable timestamp precision

## Development Steps

### Phase 1: Project Setup ✓
1. Create project structure
   - `srt_maker/` - Main package
   - `tests/` - Test suite
   - `test_runner.sh` - Automated test runner
   - `pyproject.toml` - Package configuration

2. Set up dependencies
   - openai-whisper
   - ffmpeg system binary
   - rich (for progress bars)
   - tqdm
   - pytest + pytest-cov + pytest-mock

3. Configure pytest
   - Test fixtures for video/audio files
   - Coverage reporting
   - Custom markers (slow, integration)

### Phase 2: Core Components ✓

#### Audio Extraction (`audio_extractor.py`)
- Extract audio from video using ffmpeg
- Configurable sample rate and channels
- Error handling for missing ffmpeg
- Tests: 7 tests, 100% coverage

#### Transcriber (`transcriber.py`)
- Whisper model loading
- Support for all Whisper model sizes (tiny, base, small, medium, large)
- Language detection/override
- Device selection (cpu/cuda)
- Tests: 10 tests, 100% coverage

#### SRT Generator (`srt_generator.py`)
- Timestamp formatting (HH:MM:SS,mmm)
- Configurable precision
- Segment ordering and indexing
- Text normalization
- File writing
- Tests: 12 tests, 94.7% coverage

#### CLI Interface (`cli.py`)
- Argument parsing
- Progress indicators with Rich
- Error handling
- Keyboard interrupt support
- Tests: 9 tests, 85.5% coverage

### Phase 3: Testing & Automation ✓

#### Test Infrastructure
- `tests/conftest.py` - Fixtures
- Unit tests for each module
- Mocked external dependencies (ffmpeg, whisper)
- Edge case coverage

#### Automated Test Runner (`test_runner.sh`)
- Linting (pyflakes)
- Unit tests with coverage
- Watch mode for development
- Continuous feedback

### Phase 4: Documentation ✓

#### README.md
- Installation instructions
- Usage examples
- Development guide
- Project structure

## Architecture

```
srt-maker/
├── srt_maker/
│   ├── __init__.py           # Package init
│   ├── audio_extractor.py    # ffmpeg audio extraction
│   ├── transcriber.py        # Whisper speech recognition
│   ├── srt_generator.py      # SRT file formatting
│   └── cli.py                # CLI interface
├── tests/
│   ├── conftest.py           # Test fixtures
│   ├── test_audio_extractor.py
│   ├── test_transcriber.py
│   ├── test_srt_generator.py
│   └── test_cli.py
├── test_runner.sh            # Automated test runner
├── pyproject.toml           # Package config
└── README.md                # Documentation
```

## Test Coverage

| Module | Statements | Missed | Coverage |
|--------|------------|-------|----------|
| __init__.py | 1 | 0 | 100% |
| audio_extractor.py | 22 | 0 | 100% |
| transcriber.py | 28 | 0 | 100% |
| srt_generator.py | 38 | 2 | 94.7% |
| cli.py | 69 | 10 | 85.5% |
| **Total** | **158** | **12** | **92%** |

## CLI Usage

```bash
# Basic usage
srt-maker video.mp4

# With options
srt-maker video.mp4 \
  --output subtitles.srt \
  --model base \
  --language en \
  --precision 0 \
  --device cpu \
  --verbose
```

## Features Implemented

✓ Audio extraction from video
✓ Whisper speech recognition
✓ Language detection
✓ Progress indicators
✓ Timestamp precision control
✓ Model size selection
✓ Device selection
✓ Comprehensive test suite
✓ Automated test runner
✓ Documentation

## Agentic Testing Strategy

### Continuous Feedback Loop
1. Write test → Implement code → Run tests → Fix failures
2. Unit tests for each component with mocks
3. `test_runner.sh` provides instant feedback
4. Watch mode (`--watch`) for TDD workflow

### Test Coverage Goals
- Aim for 90%+ coverage across all modules
- Mock external dependencies (ffmpeg, whisper)
- Test edge cases and error paths
- Integration tests for CLI

## Usage Example

```bash
# Generate subtitles for a video
srt-maker presentation.mp4 -o presentation.srt -m base -l en
```

Output:
```
⠋ Extracting audio from video...
⠋ Loading Whisper model...
⠋ Transcribing audio...
⠋ Generating SRT file...

42 subtitle segments generated.
✓ SRT file saved to: presentation.srt
Detected language: en
```

## Status: COMPLETE

All 12 todo items completed:
1. ✓ Project structure
2. ✓ Automated test runner
3. ✓ Pytest infrastructure
4. ✓ Dependencies installed
5. ✓ Audio extraction + tests
6. ✓ Whisper transcriber + tests
7. ✓ SRT generator + tests
8. ✓ CLI + tests
9. ✓ Progress indicators
10. ✓ Timestamp precision
11. ✓ Test fixtures
12. ✓ README documentation

## Next Steps (Optional Enhancements)

- Integration test with actual video file
- Support for multiple languages in one video
- Subtitle formatting options (font size, position)
- Batch processing for multiple videos
- Web version/API
