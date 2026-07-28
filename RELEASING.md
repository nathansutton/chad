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
   Watch it once, all the way through. Two things must survive the cut: the
   banner (logo, version, model, context, cwd) and the fix landing with its diff
   on screen. The tape (`docs/demo.tape`) is the source of truth and ships with
   the repo so the demo can't silently rot.

   Two things about recording it, both learned the hard way:
   - **The turn is timed, not matched.** `Wait+Screen` stops seeing the screen
     once chad's TUI starts a turn, so the tape sleeps through it. Overshooting
     costs a few dull frames of a finished transcript; undershooting types
     `/exit` into a running turn. If the model or the prompt changed, re-measure
     the offsets before recording — the tape header says how.
   - **Expect to re-roll.** Sampling is not deterministic: a take where the model
     fumbles (invents a path, re-reads, wanders) is a bad demo, not a bad tape.
     Watch the frames, and if it wandered, just run it again.
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
