.PHONY: help sync lock run test lint format ppt

help:
	@echo "make sync    - install project and dev dependencies with uv"
	@echo "make lock    - refresh uv.lock"
	@echo "make run     - run the project entrypoint"
	@echo "make test    - run tests"
	@echo "make lint    - run ruff checks"
	@echo "make format  - run ruff formatter"
	@echo "make ppt     - launch local PPT preview server at http://localhost:8000"

sync:
	uv sync --group dev

lock:
	uv lock

run:
	uv run python main.py

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

ppt:
	@echo "🎬 PPT preview at http://localhost:8000  (Ctrl+C to stop)"
	@cd docs/PPT && python3 -m http.server 8000
