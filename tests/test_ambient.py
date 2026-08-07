"""The ambient state channel: env_manifest / session_ledger / bash_read_skeleton.

The design contract under test, in order of importance:

1. OFF is byte-identical — with the three levers disabled, `annotate` returns its
   input unchanged and the system prompt carries no manifest block (the 1.10.0
   bare-default contract).
2. Bookkeeping is lever-independent — state is true even while the levers are
   off, so one enabled mid-session (the eval harness flips CHAD_ENABLE between
   tasks in-process) sees real history, not history that started when it did.
3. Facts, with dedup — the ledger states what happened with provenance, elides
   when nothing changed, and every emission is a fired() event.
"""

import subprocess

import pytest

from chad import ambient, levers

# Captured at module import, BEFORE the conftest autouse fixture stubs it for the
# rest of the suite — the manifest tests below exercise the real builder.
_REAL_BUILD_MANIFEST = ambient._build_manifest


@pytest.fixture
def bare(monkeypatch):
    """Levers at the shipped default (all OFF), ambient state fresh."""
    monkeypatch.delenv("CHAD_ENABLE", raising=False)
    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    ambient.reset()


@pytest.fixture
def on(monkeypatch):
    """Only the ambient levers ON — isolates them from the other 48."""
    monkeypatch.setenv("CHAD_ENABLE",
                       "env_manifest,session_ledger,bash_read_skeleton")
    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    ambient.reset()


def test_levers_registered():
    for name in ("env_manifest", "session_ledger", "bash_read_skeleton"):
        assert name in levers.LEVERS
        assert levers.LEVERS[name].group == "ambient"
    # env_manifest's whole effect is prompt content; the instrumentation gate
    # enforces the annotation/code agreement, this pins the intent.
    assert levers.LEVERS["env_manifest"].fires == levers.PASSIVE


# ---------------------------------------------------------------------------
# contract 1: OFF is byte-identical
# ---------------------------------------------------------------------------

def test_off_is_byte_identical(bare):
    r1 = ambient.annotate("edit", {"path": "a.py"}, "[edited a.py]")
    r2 = ambient.annotate("bash", {"command": "cat a.py"}, "body")
    r3 = ambient.annotate("read", {"path": "a.py"}, "1  x = 1")
    assert (r1, r2, r3) == ("[edited a.py]", "body", "1  x = 1")
    assert ambient.ledger_suffix() == ""


def test_manifest_absent_when_off(bare, monkeypatch):
    monkeypatch.setattr(ambient, "_build_manifest", lambda: "- present: gcc 13")
    assert ambient.env_manifest() == ""
    from chad.prompt import build_system_prompt
    assert "Environment manifest" not in build_system_prompt()


# ---------------------------------------------------------------------------
# contract 2: bookkeeping runs while off
# ---------------------------------------------------------------------------

def test_state_tracked_while_off_then_visible_when_enabled(bare, monkeypatch):
    ambient.annotate("edit", {"path": "a.py"}, "[edited a.py]")
    ambient.annotate("bash", {"command": "python3 -m pytest -q"}, "[exit 1]\n1 failed")
    monkeypatch.setenv("CHAD_ENABLE", "session_ledger")
    suffix = ambient.ledger_suffix()
    assert "edited: a.py" in suffix
    assert "python3 → exit 1" in suffix  # head + exit status, with provenance
    assert "this call" in suffix  # the run was the most recent call at readout


# ---------------------------------------------------------------------------
# session ledger (E2)
# ---------------------------------------------------------------------------

def test_ledger_rides_landed_mutations_only(on):
    before = levers.fire_counts().get("session_ledger", 0)
    read = ambient.annotate("read", {"path": "a.py"}, "1  x = 1")
    assert "[session]" not in read
    missed = ambient.annotate("edit", {"path": "a.py"}, "[edit failed: no match]")
    assert "[session]" not in missed  # a rejected edit changed nothing
    landed = ambient.annotate("edit", {"path": "a.py"}, "[edited a.py]")
    assert "[session] edited: a.py" in landed
    assert "no verifying run yet" in landed
    assert levers.fire_counts()["session_ledger"] == before + 1


def test_ledger_symbols_writes_and_provenance(on):
    ambient.annotate("replace_symbol", {"name": "parse", "path": "x.py"},
                     "[replaced parse]")
    ambient.annotate("bash", {"command": "python3 check.py"}, "ok")
    out = ambient.annotate("write", {"path": "answer.txt"}, "[wrote 12 bytes]")
    assert "edited: x.py·parse()" in out
    assert "wrote: answer.txt" in out
    assert "(python3 → exit 0)" in out


def test_ledger_dedups_unchanged_state_but_not_new_state(on):
    first = ambient.annotate("edit", {"path": "a.py"}, "[edited a.py]")
    assert "[session]" in first
    again = ambient.annotate("edit", {"path": "a.py"}, "[edited a.py]")
    assert "[session]" not in again  # same file, same facts — elided
    other = ambient.annotate("edit", {"path": "b.py"}, "[edited b.py]")
    assert "[session]" in other and "b.py" in other
    # force= (the done-bounce path) restates current facts through the dedup
    assert "[session]" in ambient.ledger_suffix(force=True)


def test_runner_wrappers_count_as_verifying_runs(on):
    """`uv run pytest` is a pytest run — the wrapper is stripped before the
    executing check (ambient-only; the verify gate's regex is untouched)."""
    ambient.annotate("edit", {"path": "a.py"}, "[edited a.py]")
    out = ambient.annotate("bash", {"command": "uv run pytest -q"},
                           "[exit 1]\n2 failed")
    assert out == "[exit 1]\n2 failed"  # bash results never carry the ledger
    assert "(pytest → exit 1)" in ambient.ledger_suffix(force=True)


def test_trivial_and_nonexecuting_bash_is_not_a_verifying_run(on):
    ambient.annotate("edit", {"path": "a.py"}, "[edited a.py]")
    ambient.annotate("bash", {"command": "python3 --version"}, "Python 3.11")
    ambient.annotate("bash", {"command": "ls -la"}, "total 8")
    assert "no verifying run yet" in ambient.ledger_suffix(force=True)


def test_revert_clears_edit_facts(on):
    ambient.annotate("edit", {"path": "a.py"}, "[edited a.py]")
    ambient.annotate("bash", {"command": "git checkout ."}, "[no output]")
    assert ambient.ledger_suffix(force=True) == ""  # nothing landed = nothing to state


# ---------------------------------------------------------------------------
# bash/read skeleton (E3)
# ---------------------------------------------------------------------------

@pytest.fixture
def srcfile(tmp_path, monkeypatch):
    p = tmp_path / "mod.py"
    p.write_text("def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n\n\n"
                 "class Gamma:\n    def m(self):\n        return 3\n")
    monkeypatch.chdir(tmp_path)  # repomap.service() re-roots on cwd change
    return "mod.py"


def test_skeleton_rides_bash_cat_once_per_session(on, srcfile):
    out = ambient.annotate("bash", {"command": f"cat {srcfile}"}, "def alpha(): ...")
    assert f"[file] {srcfile}:" in out
    assert "alpha() 1-2" in out and "Gamma 9-11" in out
    again = ambient.annotate("bash", {"command": f"cat {srcfile}"}, "def alpha(): ...")
    assert "[file]" not in again  # once per file per session


def test_skeleton_rides_read_but_not_error_results(on, srcfile):
    miss = ambient.annotate("read", {"path": "nope.py"}, "[no such file: nope.py]")
    assert "[file]" not in miss
    out = ambient.annotate("read", {"path": srcfile}, "1  def alpha():")
    assert f"[file] {srcfile}:" in out


def test_skeleton_skips_structureless_and_tiny_files(on, tmp_path, monkeypatch):
    p = tmp_path / "flat.py"
    p.write_text("x = 1\n")
    monkeypatch.chdir(tmp_path)
    out = ambient.annotate("bash", {"command": "cat flat.py"}, "x = 1")
    assert "[file]" not in out


def test_zero_hit_grep_gets_definition_pointer(on, srcfile):
    out = ambient.annotate("bash", {"command": "grep -rn beta src/"}, "[exit 1]")
    assert "`beta` is defined at mod.py:5" in out
    # a non-identifier or found grep gets nothing
    ok = ambient.annotate("bash", {"command": "grep -rn beta ."}, "mod.py:5:def beta")
    assert "[file]" not in ok


# ---------------------------------------------------------------------------
# env manifest (E1)
# ---------------------------------------------------------------------------

def test_manifest_content_and_prompt_block(on, monkeypatch):
    fake = {"python3": "/usr/bin/python3", "gcc": "/usr/bin/gcc",
            "apt-get": "/usr/bin/apt-get", "pip": "/usr/bin/pip"}

    def fake_run(argv, **kw):
        class P:
            stdout = {"python3": "Python 3.11.9", "gcc": "gcc (Ubuntu) 13.3.0",
                      "pip": "pip 24.0 from ..."}.get(argv[0], "")
            stderr = ""
        return P()

    # Scoped: `ambient.subprocess` IS the global module, and build_system_prompt
    # below legitimately uses subprocess.run for the workspace snapshot.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ambient.shutil, "which", lambda t: fake.get(t))
        mp.setattr(ambient.subprocess, "run", fake_run)
        body = _REAL_BUILD_MANIFEST()
    assert "python3 3.11.9" in body and "gcc 13.3.0" in body
    assert "NOT installed: " in body and "cargo" in body and "docker" in body
    assert "package managers: apt-get, pip" in body

    monkeypatch.setattr(ambient, "_build_manifest", lambda: body)
    ambient.reset()
    from chad.prompt import build_system_prompt
    prompt = build_system_prompt()
    assert "# Environment manifest" in prompt
    assert "later installs are NOT reflected" in prompt
    assert "python3 3.11.9" in prompt


def test_manifest_built_once_per_session(on, monkeypatch):
    calls = []
    monkeypatch.setattr(ambient, "_build_manifest",
                        lambda: calls.append(1) or "- present: git 2.43")
    assert ambient.env_manifest() == ambient.env_manifest()
    assert len(calls) == 1


def test_manifest_alias_family_absence(on, monkeypatch):
    """pip3-without-pip must not report 'NOT installed: pip' — absence is claimed
    only when the whole alias family is missing."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ambient.shutil, "which",
                   lambda t: "/usr/bin/x" if t in ("pip3", "python") else None)
        mp.setattr(ambient, "_probe_version", lambda t: "")
        body = _REAL_BUILD_MANIFEST()
    assert "pip3" in body and "python" in body
    assert "NOT installed: gcc" in body
    for family in ("pip,", "pip\n", "python3"):
        assert family not in body.split("NOT installed:")[1].split("\n")[0] + "\n"


def test_manifest_probe_survives_broken_tool(on, monkeypatch):
    def boom(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 1.5)
    monkeypatch.setattr(ambient.subprocess, "run", boom)
    monkeypatch.setattr(ambient.shutil, "which",
                        lambda t: "/usr/bin/x" if t in ("gcc", "apt-get") else None)
    body = _REAL_BUILD_MANIFEST()
    assert "gcc" in body  # present, just unversioned


# ---------------------------------------------------------------------------
# agent integration
# ---------------------------------------------------------------------------

def test_annotate_is_wired_into_the_agent_loop():
    """The hook exists on the main-agent dispatch path and is sub-agent-gated —
    pinned textually (an e2e exercise lives in the feel pack; this keeps the
    wiring from silently vanishing in a refactor)."""
    import inspect

    from chad import agent
    src = inspect.getsource(agent.Agent._run_turn)
    assert "ambient.annotate(name, args, result" in src
    assert "if not self._subagent:" in src.split("ambient.annotate")[0].rsplit(
        "guardrails.elide_duplicate_result", 1)[-1]


def test_ledger_rides_done_bounces():
    import inspect

    from chad import agent
    src = inspect.getsource(agent.Agent._run_turn)
    assert src.count("ambient.ledger_suffix(step=step, force=True)") >= 2
