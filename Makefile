.PHONY: install style

install:
	uv sync --all-groups
	uv run pre-commit install

style:
	uv run ruff check --fix .
	uv run ruff format .