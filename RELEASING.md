# Releasing chad

Maintainer checklist. The pipeline is tag-driven: pushing a version tag runs
`.github/workflows/publish.yml`, which builds and — after the manual `pypi-env`
environment approval — publishes `chad-code` to PyPI.

## Checklist

1. **Gates green, locally and in CI** (none of these load model weights):
   ```bash
   uv run pytest -q
   uv run ruff check src tests
   uv run mypy src/chad
   ```
2. **Behavior changes eval'd.** Anything model-visible since the last release
   (prompts, tool schemas, guardrails, engine, compaction) has been through the
   private eval rig (see CONTRIBUTING.md — the unit gate alone is not enough).
3. **CHANGELOG.md**: move `[Unreleased]` under the new version heading with the
   date. If the model itself was bumped, say so explicitly — a *model* bump is
   what tells users a re-download is coming and old snapshots can be freed.
4. **Regenerate the demo GIF** on this Mac (CI can't — it needs the model):
   ```bash
   vhs docs/demo.tape
   ```
   Watch it once. The wifi-off moment must survive the cut — it's the offline
   claim, demonstrated. Expect to tune `Sleep`/`Wait` values if the TUI or
   model timings changed; the tape (`docs/demo.tape`) is the source of truth
   and ships with the repo so the demo can't silently rot.
   Wifi comes back on at the end of the tape; check it did
   (`networksetup -getairportpower en0`).
5. **Version bump** in BOTH `pyproject.toml` and `src/chad/__init__.py`
   (`__version__` drives `--version`; the pyproject version drives the wheel and
   the ATIF trajectory's agent.version). They must match.
6. **Tag and push**:
   ```bash
   git tag v<X.Y.Z> && git push origin v<X.Y.Z>
   ```
   Approve the `pypi-env` gate when the publish workflow pauses on it.
7. **Cold-install check** (or wait for the weekly canary,
   `.github/workflows/canary.yml`):
   ```bash
   uvx --refresh chad-code -- --version
   ```
   `uvx` caches resolves — `--refresh` is what a real upgrade looks like.
8. **Benchmark claims stay honest.** If the verified Terminal-Bench number
   landed since the last release, the provisional labels flip in their own
   dedicated commit per the pre-committed landing rules — never mixed into a
   release commit.
