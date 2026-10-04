# Releasing chad

Maintainer checklist. Pushing a version tag runs `.github/workflows/publish.yml`, which
builds and, after the manual `pypi` environment approval, publishes `chad-code` to PyPI.

1. **Gate green**, locally and in CI: `make gate`.
2. **Model-visible changes eval'd.** Anything since the last release that touches prompts,
   tool schemas, guardrails, the engine or compaction has a paired polyglot comparison
   against the last release (`benchmarks/polyglot/stats.py compare`; see CONTRIBUTING.md).
3. **CHANGELOG.md**: move `[Unreleased]` under the new version with its date. Say
   explicitly whether the model changed: a model bump is what tells users a re-download
   is coming and old snapshots can be freed.
4. **Hardware tables.** If the model or the engine changed, re-run `chad-bench` and
   `benchmarks/stock/` on the release build and refresh `docs/benchmarks.md` in the same
   commit as the version bump. The docs carry measurements a reader can reproduce on their
   own Mac and nothing else: no pass rates, no leaderboards, no numbers against other
   agents or hosted models.
5. **Version bump** in both `pyproject.toml` and `src/chad/__init__.py`; they must match.
   Do it on a `release/X.Y.Z` branch and open the release PR from there.
6. **Demo GIF**, regenerated on this Mac (CI has no model): `vhs docs/demo.tape`. Watch it
   through once. The banner (version, model, context, cwd) and the fix landing with its
   diff must both survive the cut. Three things about the tape:
   - The turn is timed, not matched: `Wait+Screen` cannot see the TUI mid-turn, so the
     tape sleeps through it. If the model or the prompt changed, re-measure the offsets
     (the tape header says how).
   - Keep the priming run. It warms the prefix checkpoint off camera; without it the GIF
     records a ~75 s cold prefill no returning user pays.
   - Expect to re-roll. Sampling is not deterministic, and a take where the model wanders
     is a bad demo, not a bad tape.
7. **Tag main after the PR merges**, then approve the `pypi` gate when the workflow
   pauses on it (the environment name must match the trusted-publisher config on PyPI; see
   the comment at the top of `publish.yml`):
   ```bash
   git tag v<X.Y.Z> && git push origin v<X.Y.Z>
   ```
8. **Cold-install check**, or wait for the weekly canary (`.github/workflows/canary.yml`):
   ```bash
   uvx --refresh chad-code -- --version
   ```
