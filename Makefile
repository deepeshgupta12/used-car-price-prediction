.PHONY: setup fmt lint typecheck test

setup:
	uv pip install -e ".[dev]"

fmt:
	ruff format .

lint:
	ruff check .

typecheck:
	mypy src

test:
	pytest -q