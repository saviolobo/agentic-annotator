.PHONY: install test lint format eval-smoke

install:
	uv sync --all-groups

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .
	uv run mypy agents/ eval/ api/ mcp_servers/ --ignore-missing-imports

format:
	uv run ruff format .
	uv run ruff check --fix .

eval-smoke:
	uv run python eval/run_eval.py --smoke --n 10
