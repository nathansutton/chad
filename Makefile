# The verification gate. CONTRIBUTING.md, RELEASING.md and CI all run exactly this.
.PHONY: gate lint typecheck slop test slop-baseline slop-review
gate: lint typecheck slop test

lint:
	uv run ruff check src tests benchmarks

typecheck:
	uv run mypy src/chad benchmarks/polyglot

# anti-slop is vendored and stdlib-only but needs Python 3.12+, so it runs on its own
# interpreter, outside the 3.11 project env (config: [tool.anti-slop] in pyproject.toml).
slop:
	uv run --python 3.12 --no-project python tools/anti_slop

# Re-record today's findings as accepted debt. Only after a deliberate review — never to
# make a new finding go away.
slop-baseline:
	uv run --python 3.12 --no-project python tools/anti_slop --generate-baseline

# The findings on the lines YOUR change touched, grouped by confidence, each with the
# recipe that replaces it. Run it on a branch before `make gate`.
slop-review:
	uv run --python 3.12 --no-project python tools/anti_slop review --base main --preset agent-strict

test:
	uv run pytest -q
