#!/bin/bash
set -e

echo "========================================="
echo "SRT-Maker Agentic Test Runner"
echo "========================================="
echo ""
echo "This script provides continuous feedback during development:"
echo "  - Runs linting (pyflakes)"
echo "  - Runs unit tests with coverage"
echo "  - Provides detailed failure reports"
echo "  - Can run in watch mode for development"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
WATCH_MODE=false
SKIP_SLOW=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --watch|-w)
      WATCH_MODE=true
      shift
      ;;
    --skip-slow)
      SKIP_SLOW=true
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --watch, -w     Run in watch mode (re-run tests on file changes)"
      echo "  --skip-slow     Skip tests marked as slow"
      echo "  --help, -h      Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Function to run linting
run_lint() {
    echo -e "${YELLOW}Running linting...${NC}"
    if command -v pyflakes &> /dev/null; then
        if pyflakes srt_maker/**/*.py; then
            echo -e "${GREEN}✓ Linting passed${NC}"
        else
            return 1
        fi
    else
        echo -e "${YELLOW}⚠ pyflakes not found, skipping linting${NC}"
    fi
}

# Function to run tests
run_tests() {
    local pytest_args="-v --tb=short --cov=srt_maker --cov-report=term-missing"
    
    if [ "$SKIP_SLOW" = true ]; then
        pytest_args="$pytest_args -m 'not slow'"
    fi
    
    echo -e "${YELLOW}Running tests...${NC}"
    if pytest $pytest_args tests/; then
        echo -e "${GREEN}✓ All tests passed${NC}"
        return 0
    else
        return 1
    fi
}

# Function to show test results summary
show_summary() {
    exit_code=$1
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo -e "${GREEN}=========================================${NC}"
        echo -e "${GREEN}All checks passed! ✓${NC}"
        echo -e "${GREEN}=========================================${NC}"
    else
        echo ""
        echo -e "${RED}=========================================${NC}"
        echo -e "${RED}Some checks failed ✗${NC}"
        echo -e "${RED}=========================================${NC}"
    fi
    
    return $exit_code
}

# Main execution
if [ "$WATCH_MODE" = true ]; then
    echo -e "${YELLOW}Starting watch mode...${NC}"
    echo "Press Ctrl+C to stop"
    echo ""
    
    # Check if pytest-watch is installed
    if ! command -v ptw &> /dev/null; then
        echo -e "${RED}pytest-watch not found. Install it with: pip install pytest-watch${NC}"
        exit 1
    fi
    
    run_tests
    
    ptw --runner "pytest -v --tb=short --cov=srt_maker tests/"
else
    run_lint || show_summary 1
    run_tests || show_summary 1
    show_summary 0
fi