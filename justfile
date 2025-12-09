# just manual: https://github.com/casey/just#readme

# Default: list available recipes
_default:
    just --list

# Bootstrap: create venv with chosen Python (default 3.12) and sync deps
bootstrap default="3.12":
    uv venv --python {{default}}
    just sync

# Start async Python REPL
shell:
    uv run python -m asyncio

# Sync dependencies with environment (reads pyproject/requirements)
sync:
    uv sync


# Run the app via Docker Compose
run-compose *args:
    docker compose up {{args}}

# Apply all database migrations
migrate:
    uv run alembic upgrade head

# Create a new migration with a name
makemigration name:
    uv run alembic revision --autogenerate -m "{{name}}"

# Run tests
test *args:
    uv run pytest tests/ {{args}}

# Run CI tests (with coverage)
ci-test coverage_dir='./coverage':
    uv run pytest --cov=./ --cov-report=html:{{coverage_dir}} tests/

# Clean venv and caches
clean:
    rm -rf .venv
    rm -rf __pycache__ .pytest_cache .mypy_cache
    echo "Cleaned venv and caches"

# ---------------------------
# Ruff (lint/format) recipes
# ---------------------------

# Ensure ruff is present in the environment (optional helper)
ruff-add:
    uv add ruff

# Lint only
ruff:
    uv run ruff check .

# Lint and auto-fix (safe fixes)
ruff-fix:
    uv run ruff check . --fix

# Code formatter (like black, via ruff)
format:
    uv run ruff format .

# Alias for linters (expand later if you add mypy, etc.)
lint: ruff

# Full hygiene: fix + format
tidy:
    just ruff-fix
    just format

golden:
    pytest -vv tests/golden_tests/test_golden_panel.py
