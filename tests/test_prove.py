"""Model-free tests for `chad prove` (prove.py) — the bundled smoke test.

Nothing here loads weights. We test the parts a visitor's trust rides on: the
task fixtures are solvable and their verifiers actually discriminate, a tampered
check script cannot spoof a pass (the D27 hardening), the offline socket guard
blocks non-local connections, the scorecard renders honestly on failure and only
offers the share snippet on a 100% pass, and the wrong-backend preflight refuses
with exit code 2 before any model work.
"""
import argparse
import os
import socket

import pytest

from chad import prove

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


# ---- preflight refusals ----------------------------------------------------------

def test_prove_rejects_remote_backends(capsys):
    for backend in ("openai", "llama"):
        args = argparse.Namespace(backend=backend)
        assert prove.run(args) == 2
        err = capsys.readouterr().err
        assert "nothing to prove" in err
