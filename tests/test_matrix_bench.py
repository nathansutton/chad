"""benchmarks/matrix: the scorecard derivation, and the committed run's hygiene.

Three things a reader of the harness x engine grid has to be able to trust without
re-running 20 hours of model time:

  1. the derived columns mean what the docstring says (hand-built turns, every column);
  2. the tables committed under `_runs/` are exactly what `scorecard.py` and `run.py
     table` print from the committed rows — no hand edits between the data and the post;
  3. nothing committed under `benchmarks/matrix/` names the machine it ran on.

Stdlib only; runs on the Linux CI matrix, where there is no MLX and no model.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import socket
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(ROOT, "benchmarks", "matrix")
RUNS = os.path.join(MATRIX, "_runs")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(MATRIX, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sc():
    return _load("scorecard")


def _turn(rep, arm, task, t, prompt_n, cache_n, predicted_n, pms, gms, ttft, sys_sha,
          tools_sha="T1", n_tools=7, sys_chars=4200, t_end=None, **extra):
    return {"arm": arm, "task": task, "rep": rep, "t_arrive": t, "ttft_s": ttft,
            "t_last_byte": t_end if t_end is not None else t + 30,
            "shape": "chat", "sys_sha": sys_sha, "tools_sha": tools_sha,
            "n_tools": n_tools, "sys_chars": sys_chars, "status": 200,
            "timings": {"prompt_n": prompt_n, "cache_n": cache_n,
                        "predicted_n": predicted_n, "prompt_ms": pms,
                        "predicted_ms": gms}, **extra}


# -- 1. derivation -------------------------------------------------------------

def test_three_turn_arm(sc, tmp_path):
    rows = [{"arm": "x+llama", "task": "t", "rep": 0, "wall_s": 100.0, "passed": True,
             "timed_out": False, "generated": 600, "prefill": 3000,
             "tests_passed": 5, "tests_total": 5}]
    turns = [
        # turn 1: cold, everything prefilled
        _turn(0, "x+llama", "t", 1.0, 2000, 0, 200, 20000, 10000, 20.0, "S1"),
        # turn 2: 90% reuse, prefix stable
        _turn(0, "x+llama", "t", 40.0, 250, 2250, 200, 2500, 10000, 2.5, "S1"),
        # turn 3: 50% reuse, system prompt changed -> one churn event
        _turn(0, "x+llama", "t", 80.0, 1500, 1500, 200, 15000, 10000, 15.0, "S2"),
    ]
    per = sc.per_run(rows, turns, str(tmp_path))
    assert len(per) == 1
    r = per[0]
    assert r["round_trips"] == 3 and r["side_requests"] == 0
    assert r["tax"] == 2000
    assert r["wait_first"] == 20.0
    assert r["uncached_later"] == [250, 1500]
    assert r["wait_later"] == [2.5, 15.0]
    assert [round(x, 2) for x in r["reuse"]] == [0.9, 0.5]
    assert r["churn"] == 1 and r["churn_of"] == 2
    assert abs(r["prefill_s"] - 37.5) < 1e-9
    assert abs(r["prefill_share"] - 37500 / 67500) < 1e-9
    assert abs(r["model_busy"] - 0.675) < 1e-9
    assert r["ctx_exit"] == 3200
    assert abs(r["exp_toks"] - 6.0) < 1e-9
    assert r["ttft_lt_prefill"] == 0

    agg = sc.aggregate(per)[0]
    assert agg["reuse"] == 0.7           # median of [0.9, 0.5]
    assert agg["churn"] == (1, 2)
    assert agg["wait_later"] == 8.75 and agg["wait_later_p90"] == 15.0
    assert agg["uncached_later"] == 875
    text = sc.render([agg], per, ["t"])
    assert "x+llama" in text and "70%" in text and "1/2" in text and "8.8 s · 15 s" in text


def test_side_requests_are_classified_and_excluded(sc, tmp_path):
    """A title call (no tools) before turn 1 and another beside turn 3: the tax is the
    first AGENT request, the per-turn columns skip the side calls, the concurrency and
    the abandoned counts see them, and the after-side comparison only looks at turn 3+."""
    rows = [{"arm": "h+llama", "task": "t", "rep": 0, "wall_s": 200.0, "passed": False,
             "timed_out": True, "generated": 400, "prefill": 9000}]
    A = "h+llama"
    turns = [
        _turn(0, A, "t", 0.0, 300, 0, 20, 3000, 1000, 3.0, "TITLE", None, None, 400, t_end=120.0),
        _turn(0, A, "t", 0.5, 9000, 0, 100, 100000, 10000, 30.0, "S", "T", 18, 24000,
              t_end=110.0),
        _turn(0, A, "t", 120.0, 3000, 9100, 100, 30000, 10000, 30.0, "S", "T", 18, 24000,
              t_end=160.0),
        # abandoned summary call, concurrent with turn 3
        {"arm": A, "task": "t", "rep": 0, "t_arrive": 161.0, "t_last_byte": 170.0,
         "ttft_s": 9.0, "shape": "chat", "sys_sha": "SUM", "tools_sha": None,
         "n_tools": None, "sys_chars": 350, "status": 200, "usage": None,
         "client_disconnected": True},
        _turn(0, A, "t", 162.0, 2500, 12200, 100, 25000, 10000, 25.0, "S", "T", 18, 24000,
              t_end=197.0),
        _turn(0, A, "t", 198.0, 40, 14800, 100, 400, 10000, 0.4, "S", "T", 18, 24000,
              t_end=199.0),
    ]
    per = sc.per_run(rows, turns, str(tmp_path))
    r = per[0]
    assert r["round_trips"] == 4
    assert r["side_requests"] == 2 and r["side_concurrent"] == 2 and r["side_abandoned"] == 1
    assert r["unmeasured"] == 1
    assert r["tax"] == 9000 and r["wait_first"] == 100.0
    assert r["n_tools"] == 18 and r["sys_chars"] == 24000
    assert r["uncached_later"] == [3000, 2500, 40]
    assert r["uncached_after_side"] == [2500] and r["uncached_after_main"] == [40]
    # side-request prefill still counts toward the run's prefill seconds and busy time
    assert abs(r["prefill_s"] - (3000 + 100000 + 30000 + 25000 + 400) / 1000) < 1e-9
    assert r["churn"] == 0 and r["churn_of"] == 3
    agg = sc.aggregate(per)[0]
    assert agg["side_requests"] == 2 and agg["uncached_after_side"] == 2500
    text = sc.render([agg], per, ["t"])
    assert "2 (2 · 1)" in text


def test_no_tools_harness_has_no_side_requests(sc):
    turns = [{"n_tools": None}, {"n_tools": None}, {"n_tools": 0}]
    sc.classify(turns)
    assert not any(t["side"] for t in turns)


def test_ttft_instrument_check(sc, tmp_path):
    """The proxy's first-byte stamp is earlier than the server's own prefill time:
    counted, and never rendered as a wait."""
    rows = [{"arm": "x+llama", "task": "t", "rep": 0, "wall_s": 300.0, "passed": True,
             "timed_out": False, "generated": 100, "prefill": 18000}]
    turns = [_turn(0, "x+llama", "t", 0.0, 18000, 0, 100, 214000, 10000, 30.1, "S")]
    r = sc.per_run(rows, turns, str(tmp_path))[0]
    assert r["ttft_lt_prefill"] == 1 and r["ttft_checked"] == 1
    assert r["wait_first"] == 214.0 and r["proxy_ttft_cold"] == 30.1
    text = sc.render(sc.aggregate([r]), [r], ["t"])
    assert "214.0 s" in text and "30.1" not in text and "1 of 1 requests" in text


def test_usage_fallback_and_mlx_trace(sc, tmp_path):
    t = {"arm": "c+llama", "task": "t", "rep": 0, "t_arrive": 1.0, "ttft_s": 3.0,
         "shape": "chat", "sys_sha": "S", "tools_sha": "T", "status": 200,
         "usage": {"input_tokens": 1000, "output_tokens": 50,
                   "input_tokens_details": {"cached_tokens": 800}}}
    n = sc.norm_turn(t)
    assert n["prompt_n"] == 200 and n["cache_n"] == 800 and n["predicted_n"] == 50
    assert n["prompt_ms"] is None and n["measured"]
    # the MLX arm reads its prefill trace from the row's (repo-relative) path
    trace = tmp_path / "prefill_trace.jsonl"
    trace.write_text(
        '{"seq":1,"step":1,"prompt_tokens":73,"cached_tokens":2486,"prefill_s":1.2,'
        '"gen_tokens":100,"gen_s":5.0}\n'
        '{"seq":2,"step":2,"prompt_tokens":300,"cached_tokens":2659,"prefill_s":3.0,'
        '"gen_tokens":100,"gen_s":5.0}\n')
    rows = [{"arm": "chad+mlx", "task": "t", "rep": 0, "wall_s": 50.0, "passed": False,
             "timed_out": True, "generated": 200, "prefill": 373,
             "prefill_trace": str(trace)}]
    r = sc.per_run(rows, [], str(tmp_path))[0]
    assert r["round_trips"] == 2 and r["tax"] == 2559 and r["wait_first"] == 1.2
    assert abs(r["reuse"][0] - 2659 / 2959) < 1e-9
    assert "churn" not in r            # nothing to hash on the raw path
    assert abs(r["model_busy"] - 14.2 / 50.0) < 1e-9
    text = sc.render(sc.aggregate([r]), [r], ["t"])
    assert "1.2 s †" in text


# -- 2. the committed tables are what the scripts print --------------------------

@pytest.mark.skipif(not os.path.exists(os.path.join(RUNS, "grid.json")),
                    reason="no committed run")
def test_committed_scorecard_is_reproducible(sc):
    text, data = sc.build(RUNS)
    with open(os.path.join(RUNS, "scorecard.md")) as f:
        assert f.read() == text
    arms = {a["arm"]: a for a in data["arms"]}
    # The sampler was verified for this run, and the grid is complete.
    assert os.path.exists(os.path.join(RUNS, "sampler_audit_summary.json"))
    assert sum(a["n"] for a in arms.values()) == 88
    # The instrument caveat the post rests on: llama-server's early first byte.
    assert sum(a["ttft_lt_prefill"] for a in arms.values()) > 50


@pytest.mark.skipif(not os.path.exists(os.path.join(RUNS, "grid.json")),
                    reason="no committed run")
def test_committed_tables_md_is_reproducible():
    run = _load("run")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run.main(["table"])
    with open(os.path.join(RUNS, "tables.md")) as f:
        assert f.read().strip() == buf.getvalue().strip()


def test_tasks_are_committed_pristine():
    """`run.py` scaffolds from git, so every task file has to be in HEAD (or staged) —
    a working-tree-only task would refuse at run time, not here."""
    run = _load("run")
    for task in run.TASKS:
        mod = task.replace("-", "_")
        for name in (f"{mod}.py", f"{mod}_test.py", "instructions.md"):
            p = os.path.join(MATRIX, "tasks", task, name)
            assert os.path.exists(p), p
            r = subprocess.run(["git", "-C", ROOT, "ls-files", "--error-unmatch",
                                os.path.relpath(p, ROOT)], capture_output=True)
            assert r.returncode == 0, f"{p} is not tracked"


# -- 3. nothing committed names the machine ---------------------------------------

_PATTERNS = [
    re.compile(r"/Users/[A-Za-z0-9._-]+"),        # a macOS home directory
    re.compile(r"/home/[A-Za-z0-9._-]+/"),        # a Linux one
    re.compile(r"/var/folders/[A-Za-z0-9_]+/"),   # a per-user macOS temp dir
    re.compile(r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?:\.\d{1,3})?\b"),
    re.compile(r"\b[A-Za-z0-9-]+\.local\b"),      # a Bonjour hostname
]


def _tracked_matrix_files() -> list:
    r = subprocess.run(["git", "-C", ROOT, "ls-files", "--cached", "--others",
                        "--exclude-standard", "benchmarks/matrix"],
                       capture_output=True, text=True)
    return [os.path.join(ROOT, p) for p in r.stdout.split() if p]


def test_nothing_committed_names_the_machine():
    dynamic = []
    for s in (os.path.expanduser("~"), socket.gethostname(),
              socket.gethostname().split(".")[0]):
        if s and len(s) >= 4 and s != "/":
            dynamic.append(s)
    files = _tracked_matrix_files()
    assert files, "nothing under benchmarks/matrix is tracked"
    bad = []
    for p in files:
        if not os.path.isfile(p):
            continue
        with open(p, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8", "replace")
        for pat in _PATTERNS:
            m = pat.search(text)
            if m:
                bad.append(f"{os.path.relpath(p, ROOT)}: {m.group(0)!r}")
        for s in dynamic:
            if s in text:
                bad.append(f"{os.path.relpath(p, ROOT)}: <local identifier>")
    assert not bad, "\n".join(bad)


def test_redaction_and_relative_paths():
    run = _load("run")
    home = os.path.expanduser("~")
    host = socket.gethostname()
    out = run._redact(f"saved to {home}/x on {host}")
    assert home not in out and "~/x" in out
    if len(host) >= 4:
        assert host not in out
    assert run._rel(os.path.join(run.ROOT, "benchmarks", "x")) == "benchmarks/x"
    assert run._rel("/somewhere/else/file.json") == "file.json"
    assert run._rel(None) is None
