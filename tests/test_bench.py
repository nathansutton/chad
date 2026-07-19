"""Tier-1 tests for chad-bench's CLI wiring (bench.py). No model load.

Regression origin: a dogfood replace_symbol rewrite of `main` silently dropped the
--context-tokens argparse line while _run_agentic still read args.context_tokens, so
`chad-bench --agentic` crashed with AttributeError. The rewrite parsed cleanly — the
edit gate can't see semantic drift — so the arg↔consumer contract gets a real test.
"""
import os

from chad import bench


def _stub_agentic(monkeypatch, seen):
    monkeypatch.setattr(bench, "_pick_model", lambda: ("stub-model", "test"))
    monkeypatch.setattr(bench, "_ensure_model", lambda mid: None)
    monkeypatch.setattr(bench, "_run_agentic",
                        lambda mid, why, ctx: seen.update(mid=mid, ctx=ctx) or 0)


def test_agentic_wires_context_tokens(monkeypatch):
    """`chad-bench --agentic` must reach _run_agentic with the parsed --context-tokens
    (default 24000) — the exact call that regressed."""
    seen: dict = {}
    _stub_agentic(monkeypatch, seen)
    assert bench.main(["--agentic"]) == 0
    assert seen == {"mid": "stub-model", "ctx": 24000}
    seen.clear()
    assert bench.main(["--agentic", "--context-tokens", "12000"]) == 0
    assert seen["ctx"] == 12000


def test_chunk_arg_exports_env(monkeypatch):
    """--chunk reaches the engine via CHAD_PREFILL_CHUNK before any benchmark runs.

    bench.main writes os.environ directly, so clean up with save/restore — a
    monkeypatch.delenv AFTER the write would make teardown restore the leaked value
    (delenv records the current value to put back), poisoning later engine tests."""
    seen: dict = {}
    _stub_agentic(monkeypatch, seen)
    prior = os.environ.pop("CHAD_PREFILL_CHUNK", None)
    try:
        assert bench.main(["--agentic", "--chunk", "128"]) == 0
        assert os.environ.get("CHAD_PREFILL_CHUNK") == "128"
    finally:
        if prior is None:
            os.environ.pop("CHAD_PREFILL_CHUNK", None)
        else:
            os.environ["CHAD_PREFILL_CHUNK"] = prior
