"""Model-free tests for `chad prove` (prove.py) — the bundled smoke test.

Nothing here loads weights. We test the parts a visitor's trust rides on: the
task fixtures are solvable and their verifiers actually discriminate, a tampered
check script cannot spoof a pass (the D27 hardening), the offline socket guard
blocks non-local connections, the scorecard renders honestly on failure and only
offers the share snippet on a 100% pass, and the wrong-backend preflight refuses
with exit code 2 before any model work.
"""
import argparse
import json
import os
import socket
import subprocess

import pytest

from chad import cli, prove

# ---- task fixtures: solvable, and the verifier discriminates --------------------

# The intended fix for each task, applied "by hand" — proves the bundled fixture
# is solvable and the checker passes exactly when the fix is real.
SOLUTIONS = {
    "casual_typo_fix": {
        "greet.py": "def greet(name):\n    return 'Hello, ' + name\n"},
    "add_function": {
        "mathx.py": "def add(a, b):\n    return a + b\n\n"
                    "def factorial(n):\n"
                    "    return 1 if n <= 1 else n * factorial(n - 1)\n"},
    "locate_and_fix": {
        "discount.py": "def apply_discount(price, pct):\n"
                       "    return price * (1 - pct / 100)\n"},
    "fix_bug_midtext": {
        "construct.py": "def construct_addendum(text):\n"
                        "    marker = 'ADDENDUM:'\n"
                        "    i = text.find(marker)\n"
                        "    if i != -1:\n"
                        "        return text[i + len(marker):].strip()\n"
                        "    return text\n"},
}


def _seed(task, tmp):
    os.chdir(tmp)
    for fname, content in task["files"].items():
        with open(fname, "w") as f:
            f.write(content)


@pytest.mark.parametrize("task", prove.TASKS, ids=lambda t: t["name"])
def test_task_fixture_unsolved_fails(task, tmp_path, monkeypatch):
    """The seeded (buggy/incomplete) fixture must FAIL its verifier as-is —
    a checker that passes before the agent does anything proves nothing."""
    monkeypatch.chdir(tmp_path)
    _seed(task, tmp_path)
    assert prove._verify(task) is False


@pytest.mark.parametrize("task", prove.TASKS, ids=lambda t: t["name"])
def test_task_fixture_solved_passes(task, tmp_path, monkeypatch):
    """Applying the intended fix by hand must PASS — the fixture is solvable."""
    monkeypatch.chdir(tmp_path)
    _seed(task, tmp_path)
    for fname, content in SOLUTIONS[task["name"]].items():
        with open(fname, "w") as f:
            f.write(content)
    assert prove._verify(task) is True


def test_tampered_check_cannot_spoof(tmp_path, monkeypatch):
    """D27 verifier hardening: an 'agent' that edits the seeded check.py to
    print the expected marker — without fixing anything — must still fail,
    because _verify re-seeds the checker from the read-only source first."""
    task = next(t for t in prove.TASKS if t["name"] == "add_function")
    monkeypatch.chdir(tmp_path)
    _seed(task, tmp_path)
    with open("check.py", "w") as f:  # the spoof: no fix, checker neutered
        f.write("print('ALL CHECKS PASS')\n")
    assert prove._verify(task) is False
    # And the re-seed restored the real checker on disk.
    assert "factorial" in open("check.py").read()


def test_first_task_is_the_fastest_survivor():
    """The first task carries the two-minute promise (design: lead with the
    fastest reliable survivor). Pin the ordering so a reshuffle is deliberate."""
    assert prove.TASKS[0]["name"] == "casual_typo_fix"
    assert 3 <= len(prove.TASKS) <= 5  # design budget: 3-5 tasks


# ---- offline socket guard --------------------------------------------------------

def test_socket_guard_blocks_remote_allows_local():
    uninstall = prove._install_socket_guard()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(OSError, match="blocked at the library level"):
            s.connect(("93.184.216.34", 80))
        s.close()
        # localhost is allowed through the guard: a connect to an unused local
        # port must fail with plain refusal, NEVER the guard's message.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        try:
            s.connect(("127.0.0.1", 1))  # nothing listens on port 1
        except OSError as e:
            assert "blocked at the library level" not in str(e)
        finally:
            s.close()
    finally:
        uninstall()
    # Uninstall restores the real connect (no guard message afterward).
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", 1))
    except OSError as e:
        assert "blocked at the library level" not in str(e)
    finally:
        s.close()


# ---- scorecard honesty -----------------------------------------------------------

def _fake_results(all_pass):
    rows = [
        {"name": "casual_typo_fix", "passed": True, "wall": 45.9,
         "timed_out": False, "tok_per_s": 41.5, "gen_tokens": 900, "ttft_s": 1.3},
        {"name": "add_function", "passed": all_pass, "wall": 47.0,
         "timed_out": False, "tok_per_s": 40.2, "gen_tokens": 850, "ttft_s": None},
    ]
    meta = {"model": "Ornith-1.0-9B-UD-Q4_K_XL-MLX",
            "download_mode": "model already cached, offline guard engaged",
            "hardware": "Apple M2, 24 GB", "load_s": 8.2, "big_ram_note": None}
    return rows, meta


def test_scorecard_full_pass_offers_share_snippet():
    rows, meta = _fake_results(all_pass=True)
    card = prove._scorecard(rows, meta)
    assert "2/2 tasks passed" in card
    assert "share it" in card
    assert "not a benchmark" in card          # the D22 disclosure, verbatim intent
    assert "development suite" in card        # provenance disclosure
    assert "library level" in card            # offline claim never overstated


def test_scorecard_partial_failure_prints_no_share_snippet():
    rows, meta = _fake_results(all_pass=False)
    card = prove._scorecard(rows, meta)
    assert "1/2 tasks passed" in card
    assert "share it" not in card             # D27: share only on 100% pass
    assert "session.log" in card              # failure line: transcript pointer
    assert "repeatable" in card               # the softened retry hint
    assert "FAIL" in card


# ---- a broken run still produces a scorecard -------------------------------------

def test_verify_treats_a_hung_or_unrunnable_check_as_a_failure(tmp_path, monkeypatch):
    """The verifier is the last call between a finished proof and its scorecard. A check
    that hangs (a server the agent left running) or cannot be spawned at all is a failed
    task, not a traceback in place of the whole report."""
    task = prove.TASKS[0]
    monkeypatch.chdir(tmp_path)
    _seed(task, tmp_path)

    def hang(check_path):
        raise subprocess.TimeoutExpired(cmd=check_path, timeout=60)

    assert prove._verify(task, run_check=hang) is False

    def unrunnable(check_path):
        raise OSError("no such interpreter")

    assert prove._verify(task, run_check=unrunnable) is False


def test_a_task_that_raises_becomes_a_failed_row(tmp_path, monkeypatch, capsys):
    """One task blowing up must not cost the user the scorecard and results.json — the
    two artifacts the command exists to produce."""

    class _FakeEngine:
        def load(self):
            return 1.0

    # A complete cache: the model check passes, no download. The shipped default is a hub
    # GGUF, so "complete" is the file plus its drafter/tokenizer sidecar; a packed repo's
    # single-file layout is here too so the check reads the same either way.
    from chad import gguf_pack
    snapshot = tmp_path / "hf-snapshot"
    (snapshot / "dflash").mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_text("x")
    (snapshot / gguf_pack.hub_spec(cli._HF_MODEL)[1]).write_text("x")
    for name in gguf_pack._SIDECAR_FILES:
        (snapshot / name).write_text("{}")

    def cached_file(repo_id, filename):
        path = snapshot / filename
        return str(path) if path.exists() else None

    host = cli.Host(platform_id=lambda: ("Darwin", "arm64"), ram_gb=lambda: 64.0,
                    cached_file=cached_file)
    invoking_dir = tmp_path / "invoking"
    invoking_dir.mkdir()
    monkeypatch.chdir(invoking_dir)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")      # run() sets it; restore after
    monkeypatch.delenv("CHAD_MODEL", raising=False)

    def boom(engine, task, capture_ttft=False):
        raise RuntimeError("the engine fell over")

    real_connect = socket.socket.connect
    rc = prove.run(cli._prove_parser().parse_args([]), host=host,
                   make_engine=lambda model_id: _FakeEngine(), run_one=boom)

    assert socket.socket.connect is real_connect  # the offline guard never outlives run()
    assert rc == 1
    rows = json.loads((invoking_dir / "results.json").read_text())["results"]
    assert len(rows) == len(prove.TASKS)
    assert all(r["passed"] is False and r["error"] == "RuntimeError" for r in rows), rows
    card = capsys.readouterr().out
    assert f"0/{len(prove.TASKS)} tasks passed" in card
    assert "task raised RuntimeError" in card
    assert "share it" not in card


# ---- preflight refusals ----------------------------------------------------------

def test_prove_rejects_remote_backends(capsys):
    for backend in ("llama",):
        args = argparse.Namespace(backend=backend)
        assert prove.run(args) == 2
        err = capsys.readouterr().err
        assert "nothing to prove" in err
