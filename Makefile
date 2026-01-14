.PHONY: lint format test check all clean ui help

# Default target
help:
	@echo "Available commands:"
	@echo "  make lint     - Run ruff linter"
	@echo "  make format   - Auto-format code with ruff"
	@echo "  make test     - Run unit tests (mocked, no API keys)"
	@echo "  make check    - Run lint + format check (CI equivalent)"
	@echo "  make all      - Run check + test"
	@echo "  make ui       - Start Streamlit UI"
	@echo "  make clean    - Remove cache and build artifacts"

# Linting
lint:
	poetry run ruff check .

# Auto-format code
format:
	poetry run ruff format .

# Run unit tests (mocked only)
test:
	poetry run pytest tests/ -v --tb=short -m "not requires_api"

# Run all tests including integration (requires API keys)
test-all:
	poetry run pytest tests/ -v --tb=short

# CI-equivalent check (lint + format check)
check: lint
	poetry run ruff format --check .

# Run everything
all: check test

# Start Streamlit UI
ui:
	poetry run streamlit run ui/app.py

# Clean up
clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache .ruff_cache
	rm -rf *.egg-info dist build
	rm -rf .coverage htmlcov
