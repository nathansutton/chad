"""The ambient state channel: env_manifest / session_ledger / bash_read_skeleton /
bash_empty_diagnose / verify_baseline.

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
    """Only the ambient levers ON — isolates them from the other levers."""
    monkeypatch.setenv("CHAD_ENABLE",
                       "env_manifest,session_ledger,bash_read_skeleton,"
                       "bash_empty_diagnose,verify_baseline,rg_replace_flag_note")
    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    ambient.reset()


def test_levers_registered():
    for name in ("env_manifest", "session_ledger", "bash_read_skeleton",
                 "bash_empty_diagnose", "verify_baseline", "rg_replace_flag_note"):
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
    r4 = ambient.annotate("bash", {"command": "rg -n zzz src/"}, "[exit 1]")
    r5 = ambient.annotate("bash", {"command": "pytest -q"}, "[exit 1]\n1 failed")
    assert (r1, r2, r3, r4, r5) == ("[edited a.py]", "body", "1  x = 1",
                                    "[exit 1]", "[exit 1]\n1 failed")
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
# empty-result diagnosis (E5)
# ---------------------------------------------------------------------------

@pytest.fixture
def shortfile(tmp_path, monkeypatch):
    p = tmp_path / "short.py"
    p.write_text("".join(f"x{i} = {i}\n" for i in range(10)))
    monkeypatch.chdir(tmp_path)
    return "short.py"


def test_sed_range_past_eof_reports_the_real_length(on, shortfile):
    """The measured loss this repays: `sed -n '330,420p'` on a 324-line file came back
    `[no output]`, and the only way out was to spend the next call on `wc -l`."""
    out = ambient.annotate("bash", {"command": f"sed -n '80,120p' {shortfile}"},
                           "[no output]")
    assert f"[file] {shortfile} has 10 lines" in out
    assert "the requested 80-120 is past the end" in out


def test_in_range_and_nonempty_reads_say_nothing(on, shortfile):
    """The note is a diagnosis of emptiness, not a footer on every sed."""
    empty_but_in_range = ambient.annotate(
        "bash", {"command": f"sed -n '2,4p' {shortfile}"}, "[no output]")
    assert "past the end" not in empty_but_in_range
    got_output = ambient.annotate("bash", {"command": f"sed -n '900,999p' {shortfile}"},
                                  "x0 = 0")
    assert "[file] short.py has" not in got_output


def test_zero_match_names_the_pattern_and_scope(on, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = ambient.annotate("bash", {"command": "rg -n 'NoSuchThing' src/ test/"},
                           "[exit 1]")
    assert "nothing matched 'NoSuchThing' in src/ test/" in out
    assert "rg found no rows" in out


def test_flag_arguments_are_not_mistaken_for_the_pattern(on, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = ambient.annotate("bash", {"command": "rg -m 5 --glob '*.ts' Needle src/"},
                           "[exit 1]")
    assert "nothing matched 'Needle' in src/" in out


def test_inverted_filter_is_not_described_as_a_failed_match(on, tmp_path, monkeypatch):
    """`rg X src/ | grep -v y` comes back empty because everything matched the -v, not
    because nothing did. Saying "nothing matched" would be exactly backwards, and it is
    the last stage — not the tree — that the model has to fix."""
    monkeypatch.chdir(tmp_path)
    out = ambient.annotate(
        "bash", {"command": "rg -n 'TimeoutOptions' source/ | grep -v utils/timeout.ts"},
        "[exit 1]")
    assert "nothing matched" not in out
    assert "the `grep -v` filter" in out and "let no rows through" in out
    assert "the last stage of the pipeline" in out


def test_nonempty_result_gets_no_empty_diagnosis(on, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = ambient.annotate("bash", {"command": "rg -n 'x' src/"}, "src/a.py:1:x")
    assert "[grep]" not in out


# ---------------------------------------------------------------------------
# rg --replace (E6)
# ---------------------------------------------------------------------------

def test_rg_replace_output_is_labelled_as_rewritten(on):
    """The rows are not the file's text. Measured: a model wrote `rg -rn PATTERN` for
    grep's recursive flag, read `test('n option', …)` back out of its own test file, and
    spent three calls concluding the file had been corrupted."""
    out = ambient.annotate("bash", {"command": 'rg -rn "timeoutMessage" test/main.ts'},
                           "test('n option', async t => {")
    assert "REWRITTEN output, not the file's text" in out
    assert "`-rn` is `--replace` with `n` as the replacement text" in out


def test_rg_replace_note_does_not_fire_on_lookalikes(on):
    """`--regexp` is not `-r`, `grep -r` really is recursive, and an rg without a
    replacement is just a search. A note that fired on those would be teaching the model
    a fact about its command that isn't true of that command."""
    for command in ('rg -n "x" src/', "rg --regexp foo src/", "grep -rn foo src/",
                    "ls -r; rg -n foo ."):
        out = ambient.annotate("bash", {"command": command}, "rows")
        assert "REWRITTEN" not in out, command


# ---------------------------------------------------------------------------
# verify baseline (E4)
# ---------------------------------------------------------------------------

def _seq(*calls):
    for name, args, result in calls:
        out = ambient.annotate(name, args, result)
    return out


def test_pre_edit_run_is_recalled_on_a_failing_post_edit_run(on):
    out = _seq(
        ("bash", {"command": "npx ava test/main.ts"},
         "[exit 1]\n  579 tests passed\n  2 known failures"),
        ("edit", {"path": "source/core/Ky.ts"}, "[edited source/core/Ky.ts]"),
        ("bash", {"command": "npx ava test/retry.ts"}, "[exit 1]\n  2 tests failed"),
    )
    assert "[baseline] before your first edit, `npx ava` → exit 1" in out
    # the runner's own words, verbatim — never a paraphrase of what it meant
    assert '"579 tests passed · 2 known failures"' in out


def test_baseline_is_per_runner_never_borrowed_across_tools(on):
    """A `tsc` baseline recalled against an `ava` failure would be a fabricated
    comparison. Each runner answers only for itself, and an un-baselined one stays
    silent rather than guessing."""
    out = _seq(
        ("bash", {"command": "npx tsc --noEmit"}, "[exit 0]"),
        ("edit", {"path": "a.ts"}, "[edited a.ts]"),
        ("bash", {"command": "npx tsc --noEmit -p tsconfig.json"},
         "[exit 1]\na.ts(1,1): error TS2379: nope"),
    )
    assert "before your first edit, `npx tsc` → exit 0" in out
    quiet = ambient.annotate("bash", {"command": "pytest -q"}, "[exit 1]\n3 failed")
    assert "[baseline]" not in quiet


def test_baseline_needs_a_pre_edit_run_and_a_post_edit_failure(on):
    # a run that only happens AFTER the first edit baselines nothing
    after = _seq(
        ("edit", {"path": "a.py"}, "[edited a.py]"),
        ("bash", {"command": "pytest -q"}, "[exit 1]\n1 failed"),
        ("bash", {"command": "pytest -q"}, "[exit 1]\n1 failed"),
    )
    assert "[baseline]" not in after
    # and a passing run asks no question, so it gets no answer
    ambient.reset()
    passing = _seq(
        ("bash", {"command": "pytest -q"}, "[exit 1]\n1 failed"),
        ("edit", {"path": "a.py"}, "[edited a.py]"),
        ("bash", {"command": "pytest -q"}, "[exit 0]\n5 passed"),
    )
    assert "[baseline]" not in passing


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


def test_manifest_names_the_search_toolbox_and_its_holes(on, monkeypatch):
    """The shell-first arm reads and searches with these binaries and nothing else, so
    whether `rg` is on the host is a fact about that arm's whole read path. Presence is
    reported without a version probe (the version is not what costs a turn), and a
    missing one is named — otherwise the first taught move, `rg -n`, exits 127 and the
    model spends a round trip discovering the host has plain grep."""
    present = {"grep", "sed", "awk", "find"}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ambient.shutil, "which",
                   lambda t: "/usr/bin/x" if t in present else None)
        mp.setattr(ambient, "_probe_version", lambda t: "")
        body = _REAL_BUILD_MANIFEST()
    line = next(l for l in body.splitlines() if l.startswith("- search/text:"))
    assert "grep · sed · awk · find" in line
    assert "NOT present: rg, jq" in line

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ambient.shutil, "which",
                   lambda t: "/usr/bin/x" if t in present | {"rg", "jq"} else None)
        mp.setattr(ambient, "_probe_version", lambda t: "")
        full = _REAL_BUILD_MANIFEST()
    line = next(l for l in full.splitlines() if l.startswith("- search/text:"))
    assert line == "- search/text: rg · grep · sed · awk · find · jq"


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
