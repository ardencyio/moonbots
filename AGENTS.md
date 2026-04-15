# AGENTS.md

## Overview

This repository follows a standardized Python development workflow optimized for agent-based coding and automation (Pi, Opencode, Goose, and other compatible assistants). All Python environment, dependency, and management tasks **must** use `uv` instead of legacy tools like `pip`, `poetry`, or `pipenv`.

---

## Python Workflow

### Environment Management

- Use [`uv`](https://github.com/astral-sh/uv) for all environment creation, dependency installation, and package management.
- Do **not** use `pip`, `poetry`, or `venv` directly.
- Typical workflow:

```bash
uv venv
uv add <package-name>
uv run python <script>.py
```

### Dependency Installation

- Add dependencies with `uv add`.
- Update dependencies with `uv sync`.
- Lock dependencies automatically via `uv.lock`.

Example:

```bash
uv add requests ruff ty
uv sync
```

---

## Linting and Formatting

### Linter: Ruff

- Use [Ruff](https://docs.astral.sh/ruff/) for linting, formatting, and code style.
- Replace `flake8`, `black`, and similar tools with Ruff.
- Configuration should live in `pyproject.toml` under `[tool.ruff]`.

Common commands:

```bash
uv run ruff check .
uv run ruff format .
```

---

## Type Checking

### Type Checker: Ty

- Use [Ty](https://docs.astral.sh/ty/) for static type analysis and enforcement.
- Replace tools like `mypy` or `pyright` with Ty.
- Run type checks via uv:

```bash
uv run ty .
```

---

## Agent Workflow Integration

Coding agents interacting with this repo should:

1. Use `uv` for all Python environment and dependency tasks.
2. Run Ruff before code submission or commit.
3. Run Ty for type verification after code generation.
4. Avoid using or suggesting legacy package managers (`pip`, `poetry`, etc.).

---

## Example Agent Setup

For automated workflows:

```bash
# Initialize environment
uv venv

# Install dev tools
uv add -d ruff ty

# Check code
uv run ruff check .
uv run ty .
```

---

## Notes

- Target Python version: `>=3.11`
- Agents may use `uv run` for all Python invocations.
- Keep scripts in `scripts/` or `src/` unless specified otherwise.
- All generated code should follow Ruff formatting and pass Ty checks before submission.
