# AGENTS.md

Guidelines for agentic coding systems (AI agents, Cursor, Copilot, etc.) working in this repository.

## Build, Lint & Test Commands

### Install Dependencies
```bash
# Install package and dependencies
pip install -e .

# Install with dev dependencies for testing
pip install -e ".[dev]"
```

### Linting
```bash
# Run pyflakes on all source files
pyflakes srt_maker/**/*.py
```

### Testing
```bash
# Run all tests with coverage
pytest -v --cov=srt_maker tests/

# Run a single test file
pytest tests/test_cli.py -v

# Run a specific test function
pytest tests/test_cli.py::TestCLI::test_parse_args_defaults -v

# Run tests matching a pattern
pytest -k "test_parse" -v

# Skip slow tests
pytest -v -m "not slow"

# Run with coverage report
pytest --cov=srt_maker --cov-report=term-missing
```

### Test Runner Script
```bash
# Run all tests with linting (recommended)
./test_runner.sh

# Run in watch mode (auto-rerun on file changes)
./test_runner.sh --watch

# Skip slow tests
./test_runner.sh --skip-slow
```

## Code Style Guidelines

### Imports
- Standard library imports first, then third-party, then local imports
- Imports organized alphabetically within groups
- Use explicit imports; avoid star imports
- Example:
  ```python
  import logging
  import subprocess
  from pathlib import Path
  from typing import List, Dict, Optional
  
  import whisper
  from rich.console import Console
  
  from .audio_extractor import AudioExtractor
  ```

### Formatting
- Line length: follow PEP 8 (79 chars for code, 72 for docstrings)
- Use 4 spaces for indentation (never tabs)
- Linted with pyflakes (syntax and import checks)

### Type Annotations
- Use type hints for function parameters and return types
- Use `Optional[T]` for nullable values; avoid `Union[T, None]`
- Use `typing` module imports: `List`, `Dict`, `Optional`, `Any`, etc.
- Example:
  ```python
  def extract_audio(
      self,
      video_path: str,
      output_path: Optional[str] = None,
      sample_rate: int = 16000,
  ) -> str:
  ```

### Naming Conventions
- **Functions/Methods**: `lowercase_with_underscores`
- **Classes**: `PascalCase`
- **Constants**: `UPPERCASE_WITH_UNDERSCORES`
- **Private members**: prefix with `_` (e.g., `_internal_method`)
- **Modules**: `lowercase_with_underscores`

### Error Handling
- Catch specific exceptions; avoid bare `except:`
- Raise `RuntimeError` for operational failures
- Raise `ValueError` for invalid arguments
- Log errors with `logger.error()` or `logger.exception()`
- Provide helpful error messages to users via console
- Example:
  ```python
  except subprocess.CalledProcessError as e:
      logger.error(f"Failed to extract audio: {e.stderr}")
      raise RuntimeError(f"Failed to extract audio from {video_path}: {e.stderr}")
  ```

### Logging
- Get logger at module level: `logger = logging.getLogger(__name__)`
- Use appropriate levels: `debug`, `info`, `warning`, `error`
- Log important state changes and operations
- Use `logger.exception()` in exception handlers to include traceback

### Documentation
- Docstrings for functions and classes using triple quotes
- Concise docstring format (one-line or summary + details)
- Describe parameters and return values when non-obvious

### Testing Patterns
- Test file naming: `test_*.py`
- Test class naming: `Test*`
- Test function naming: `test_*`
- Use pytest fixtures for setup (see `tests/conftest.py`)
- Mock external dependencies (whisper, ffmpeg, file operations)
- Use `pytest.raises()` for exception testing
- Mark slow tests with `@pytest.mark.slow`
- Mark integration tests with `@pytest.mark.integration`

### Project Structure
```
srt-maker/
├── srt_maker/              # Main package
│   ├── __init__.py
│   ├── cli.py              # CLI interface & entry point
│   ├── audio_extractor.py  # Video to audio conversion
│   ├── transcriber.py      # Whisper-based speech recognition
│   └── srt_generator.py    # SRT format output
├── tests/                  # Test suite
│   ├── conftest.py         # Pytest fixtures & configuration
│   ├── test_cli.py
│   ├── test_audio_extractor.py
│   ├── test_transcriber.py
│   └── test_srt_generator.py
├── pyproject.toml          # Package configuration
├── test_runner.sh          # Automated test runner
└── AGENTS.md               # This file
```

## Key Dependencies
- **whisper**: OpenAI Whisper for speech recognition
- **ffmpeg-python**: Audio/video processing
- **rich**: Terminal styling and progress bars
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting

## Notes for Agents
- Always run tests before committing changes
- Maintain 100% test coverage for critical paths
- Use the test_runner.sh script for fast feedback loops
- Check for type correctness; use type hints throughout
- Keep error messages user-friendly (via `rich.console.Console`)
- Run `./test_runner.sh` (not just `pytest`) to catch import/lint issues
