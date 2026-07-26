"""Done-audit: bounce a would-be-accepted `done` once with the task's own
requirement lines + stat-level path facts, then accept the next `done` unconditionally.

The extraction tests run against a corpus of task statements written to cover the
shapes real task text actually takes — numbered steps, nested bullets, one-paragraph
prose, unnumbered criteria — because the heuristics are regex-level and garble on
shape, not on subject matter. The e2e tests drive the real run_turn loop via the
scripted engine: the once-per-turn latch, precedence below the verify gate, and the
supersedes-recheck contract are all control-flow properties only the loop can prove.
"""

import pytest

from chad import guardrails
from test_agent_e2e import _agent, _tool_call

# --- Task-statement corpus, by structural shape ------------------------------------
# Each fixture is here for the SHAPE it puts in front of the regexes, not its subject:
# numbered steps with a trailing non-path requirement, nested bullets carrying literal
# `s/` bodies and template fragments, a single over-long paragraph, inline prose with
# version/revision noise, unnumbered criteria, and short prose. Between them they cover
# every extraction hazard the heuristics have to survive.

TASK_NUMBERED_STEPS = """\
You need to build a small job-scheduler daemon that accepts newline-delimited JSON \
commands over a Unix domain socket and tracks a queue of named jobs in memory. A client \
will communicate with your daemon over that socket.

You need to:
1. Install click (8.1.7) and jsonschema (4.21.1) python packages system-wide.
2. Create a file /app/schema.json describing the command envelope, which defines two commands:
  a. enqueue takes an object named EnqueueRequest that includes a name (string) as a parameter and returns an EnqueueReply with a position (int) field
  b. drain takes an object named DrainRequest that includes a count (int) as a parameter and returns a DrainReply with a removed (int) field
3. Generate the validator stubs from the schema (the generator emits two python files: {command name}_req.py and {command name}_reply.py) and place them in the /app directory.
4. Create /app/scheduler.py, in which you will implement the daemon logic in a class called Scheduler. You will bind the socket at /tmp/sched.socket.
5. Run the scheduler.py file and keep it running in the background.
"""

TASK_NESTED_BULLETS = """\
Transform /app/source.tsv (2 million rows) to match /app/target.tsv exactly using \
**byte-efficient** sed programs. Save your driver script as /app/rewrite.sh.

Requirements:
  - Create three distinct, non-empty labelled blocks x, y, z with < 200 total bytes (or you will risk timing out on the test suite).
  - Your script must contain ONLY the following allowed constructs:
      + Block definition via labels (each a single line, no linebreaks):
        :x s/old_pattern/new_pattern/g
      + Block invocation: b x, b y, b z
      + Exit: q or Q
  - Block bodies may use basic sed editing commands and essential addressing for text transformation (e.g., s/ for search/replace) - NO shell escapes, file read/write commands, or scripting languages.
  - Script must run headlessly: sh /app/rewrite.sh /app/source.tsv.

Success criteria:
  - /app/rewrite.sh exists and contains only allowed constructs.
  - sed exits 0 after running the script.
  - After editing, source.tsv matches target.tsv byte-for-byte.
"""

TASK_ONE_PARAGRAPH = """\
Ensure that the migration script migrate.sql applies cleanly against the installed \
Postgres server with no "column does not exist" errors. In doing so, the only edits you \
may make are to rename columns in columns.csv to their permitted aliases listed in \
aliases.txt (each line specifies a family of allowed aliases). Do not edit migrate.sql \
or aliases.txt.
"""

TASK_INLINE_PROSE = """\
Given the query "quarterly revenue" retrieve the document with the 5th highest cosine \
similarity among the texts at /app/corpus.txt, where each line is a separate document, \
using the all-MiniLM-L6-v2 embedding model at revision \
c9745ed1d9f207416be6d2e6f8de32d1f16199bf. Write the resulting line to /app/answer.txt. \
You must use the installed sentence-transformers package (version 3.0.1) to load and \
encode the model.
"""

TASK_NUMBERED_CRITERIA = """\
You are provided with a Turtle file at `/app/catalog_graph.ttl` containing both the \
ontology and the instance data for a knowledge graph describing publishers.

Your task:
- Create a SPARQL query that retrieves all editors meeting all of the following criteria:
    1. They are senior editors.
    2. They work on at least one imprint of a publisher headquartered in a Nordic Council country.
    3. Among all imprints they work on, at least one has more than 10 titles currently scheduled for release in that imprint.

In the knowledge graph, countries are represented using their two-letter ISO 3166-1 alpha-2 codes as string values.
Save your query in `/app/editors.sparql`.
"""

TASK_SHORT_PROSE = """\
I have a sqlite database in /app/archive.db that was corrupted through binary truncation. \
Recover as many of the rows as possible, and create a JSON file in /app/rescued.json. \
The output should have the format [{"label": "itemAB", "count": M}, {"label": \
"itemCD","count": N}, ...]
"""


# --- Layer 1: path extraction across the corpus --------------------------------------

@pytest.mark.parametrize("task_text,expected", [
    (TASK_NUMBERED_STEPS,
     ["/app/schema.json", "/app", "/app/scheduler.py", "/tmp/sched.socket"]),
    (TASK_NESTED_BULLETS,
     ["/app/source.tsv", "/app/target.tsv", "/app/rewrite.sh"]),
    (TASK_ONE_PARAGRAPH, ["migrate.sql", "columns.csv", "aliases.txt"]),
    (TASK_INLINE_PROSE, ["/app/corpus.txt", "/app/answer.txt"]),
    (TASK_NUMBERED_CRITERIA, ["/app/catalog_graph.ttl", "/app/editors.sparql"]),
    (TASK_SHORT_PROSE, ["/app/archive.db", "/app/rescued.json"]),
])
def test_extracts_exactly_the_deliverable_paths(task_text, expected):
    """Every path the task names, nothing else: no `json.load`-style attribute access,
    no version numbers, no s/…/ regex bodies, no {template}_req.py fragments."""
    assert guardrails.audit_extract_paths(task_text) == expected


def test_extraction_rejects_the_known_garbage_shapes():
    # Shapes that look path-ish to a naive regex: index arithmetic, attribute access,
    # URL double-slashes, a run-on sentence period, and an `s/` substitution body.
    text = ("prints outputs argv[1]/2. Use json.load and re.sub on it. "
            "See https://example.com/docs for more. All the videos.Your script "
            "may use :s/ for search/replace. Install grpcio (1.73.0).")
    assert guardrails.audit_extract_paths(text) == []


def test_extraction_dedupes_bare_names_already_inside_absolute_paths():
    # /app/vendor/x.tar.gz also matches the bare regex as "x.tar" — one entry only.
    text = "The snapshot is at /app/vendor/dataset-2024-release.tar.gz; use it."
    assert guardrails.audit_extract_paths(text) == [
        "/app/vendor/dataset-2024-release.tar.gz"]


# --- Layer 1: requirement-line quoting ------------------------------------------------

def test_quotes_a_trailing_requirement_that_names_no_deliverable():
    """The "leave it running" class of skipped requirement: the final step names no
    absolute path, so only the numbered-item cue can pull it into the quote."""
    paths = guardrails.audit_extract_paths(TASK_NUMBERED_STEPS)
    lines = guardrails.audit_requirement_lines(TASK_NUMBERED_STEPS, paths)
    assert "5. Run the scheduler.py file and keep it running in the background." in lines


def test_quotes_numbered_criteria_without_imperative_keywords():
    paths = guardrails.audit_extract_paths(TASK_NUMBERED_CRITERIA)
    lines = guardrails.audit_requirement_lines(TASK_NUMBERED_CRITERIA, paths)
    assert "1. They are senior editors." in lines


def test_single_paragraph_task_is_quoted_whole():
    """A whole task can arrive as one ~350-char line whose acceptance criteria (the
    only-these-edits constraint, the do-not-edit list) sit past char 240 — they must
    survive the per-line cap rather than being truncated away."""
    paths = guardrails.audit_extract_paths(TASK_ONE_PARAGRAPH)
    (line,) = guardrails.audit_requirement_lines(TASK_ONE_PARAGRAPH, paths)
    assert "Do not edit migrate.sql or aliases.txt." in line


def test_requirement_lines_are_capped_and_prefer_path_lines():
    text = "\n".join([f"you must do thing {i}" for i in range(20)]
                     + ["write the result to /app/out.txt"])
    lines = guardrails.audit_requirement_lines(
        text, guardrails.audit_extract_paths(text))
    assert len(lines) == guardrails.AUDIT_MAX_REQ_LINES
    assert "write the result to /app/out.txt" in lines


# --- Layer 1: deterministic path facts ------------------------------------------------

def test_path_facts_are_stat_level_and_neutral(tmp_path):
    """Absence is stated as a fact, never as a failure — a task may legitimately delete
    a path it mentions. Existence carries size + whether it changed this turn."""
    old = tmp_path / "old.txt"
    old.write_text("x")
    import os
    os.utime(old, (1_000_000_000, 1_000_000_000))   # long before turn start
    fresh = tmp_path / "fresh.txt"
    fresh.write_text("hello")                        # now — after turn start
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    turn_start_epoch = 1_500_000_000.0               # between old and fresh
    facts = guardrails.audit_path_facts(
        [str(tmp_path / "missing.txt"), str(old), str(fresh), str(empty),
         str(tmp_path)], turn_start_epoch)
    assert facts[0].endswith("mentioned in the task, currently absent")
    assert "not modified during this turn" in facts[1]
    assert "5 bytes, modified during this turn" in facts[2]
    assert "0 bytes" in facts[3]
    assert facts[4].endswith("exists (directory)")
    for f in facts:   # neutral phrasing: no fact may assert a failure it can't know
        assert "fail" not in f.lower() and "wrong" not in f.lower()


# --- done_audit: runway + no-anchor guards --------------------------------------------

def _state(**kw):
    s = {"turn_start_epoch": 0.0, "wall_s": 0.0, "wall_budget_s": None, "step_walls": []}
    s.update(kw)
    return s


def test_audit_declines_without_min_runway():
    # 100s left of 900 < AUDIT_MIN_RUNWAY_S: bouncing now converts a wrong-done into
    # a wall-death.
    assert guardrails.done_audit(
        TASK_SHORT_PROSE, _state(wall_s=800.0, wall_budget_s=900.0)) is None


def test_audit_declines_when_recent_steps_eat_the_runway():
    # 150s left clears the 120s floor, but recent steps run 100s each — 2x the mean
    # step wall does not fit.
    assert guardrails.done_audit(
        TASK_SHORT_PROSE,
        _state(wall_s=3450.0, wall_budget_s=3600.0, step_walls=[100.0, 100.0])) is None
    # Same clock with fast steps: fires.
    assert guardrails.done_audit(
        TASK_SHORT_PROSE,
        _state(wall_s=3450.0, wall_budget_s=3600.0, step_walls=[10.0])) is not None


def test_audit_fires_with_no_wall_budget_configured():
    assert guardrails.done_audit(TASK_SHORT_PROSE, _state()) is not None


def test_audit_declines_when_the_task_text_gives_it_nothing_to_quote():
    assert guardrails.done_audit("hi there, quick question", _state()) is None


def test_audit_steer_quotes_requirements_and_facts(tmp_path):
    f = tmp_path / "out.txt"
    f.write_text("data")
    steer = guardrails.done_audit(
        f"You must write the answer to {f} exactly as specified.",
        _state(turn_start_epoch=0.0))
    assert f"You must write the answer to {f} exactly as specified." in steer
    assert f"{f}: exists, 4 bytes, modified during this turn" in steer
    assert "call done again; it will be accepted" in steer


# --- e2e through the real run_turn loop -----------------------------------------------

def test_audit_bounces_a_done_once_then_accepts_unconditionally(tmp_path):
    """The anti-spiral contract: exactly one bounce per turn, and the audit supersedes
    the generic done_spec_recheck (both levers are ON here — default state)."""
    f = tmp_path / "out.py"
    script = [
        _tool_call("write", path=str(f), content="print('hi')\n"),
        _tool_call("bash", command=f"python {f}"),      # verify -> clears unverified
        _tool_call("done", summary="first done"),       # bounced by the audit
        _tool_call("done", summary="audited and done"),  # accepted, no further audit
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result == "audited and done"
    assert agent.engine._i == len(script)
    audits = [m for m in agent.messages if "done-audit" in m.get("content", "")]
    assert len(audits) == 1
    # The steer quotes the task's own requirement line and a stat fact for the path.
    assert f"write a script at {f} that prints hi" in audits[0]["content"]
    assert f"{f}: exists," in audits[0]["content"]
    # Supersedes the recheck: its generic steer never appears alongside the audit.
    assert not any("every concrete deliverable" in m.get("content", "")
                   for m in agent.messages)


def test_audit_disabled_restores_the_recheck_path(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_DISABLE", "done_audit")
    f = tmp_path / "out.py"
    script = [
        _tool_call("write", path=str(f), content="print('hi')\n"),
        _tool_call("bash", command=f"python {f}"),
        _tool_call("done", summary="first done"),        # deferred by the recheck
        _tool_call("done", summary="rechecked"),
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result == "rechecked"
    assert not any("done-audit" in m.get("content", "") for m in agent.messages)
    assert any("every concrete deliverable" in m.get("content", "")
               for m in agent.messages)


def test_audit_runs_below_the_verify_rejection(tmp_path):
    """Precedence: an unverified-edit done takes the existing `verify` rejection (with
    its own nudge budget); the audit only ever bounces a done that would otherwise be
    ACCEPTED."""
    f = tmp_path / "out.py"
    script = [
        _tool_call("write", path=str(f), content="print('hi')\n"),
        _tool_call("done", summary="premature"),        # verify rejection, NOT audit
        _tool_call("bash", command=f"python {f}"),      # now verify the edit
        _tool_call("done", summary="verified done"),    # audit bounce
        _tool_call("done", summary="final"),            # accepted
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result == "final"
    tool_notes = [m["content"] for m in agent.messages if m.get("role") == "tool"]
    verify_idx = next(i for i, c in enumerate(tool_notes) if "not done yet" in c)
    audit_idx = next(i for i, c in enumerate(tool_notes) if "done-audit" in c)
    assert verify_idx < audit_idx
    assert sum("done-audit" in c for c in tool_notes) == 1


# --- task-text hygiene before extraction -----------------------------

def test_relaunch_progress_note_is_stripped_from_extraction():
    """cli.py relaunches with `{task}\\n\\n[{progress note}]`; the note's paths and
    `$ command` lines are session noise, not requirements: left in, a bounce quotes
    /bin/sh and /dev/null back at the model as if they were deliverables."""
    note = "\n".join([
        guardrails.PROGRESS_NOTE_HEADER,
        "Files already edited: /tmp/cdb_input.txt",
        "Commands already tried (do not blindly repeat):",
        "  $ echo x | /bin/sh > /dev/null",
    ])
    user_text = f"{TASK_SHORT_PROSE}\n\n[{note}]"
    assert guardrails.audit_task_text(user_text) == TASK_SHORT_PROSE
    steer = guardrails.done_audit(guardrails.audit_task_text(user_text), _state())
    for noise in ("/bin/sh", "/dev/null", "/tmp/cdb_input.txt", "already tried"):
        assert noise not in steer


def test_review_pass_preamble_is_stripped_from_extraction():
    user_text = TASK_SHORT_PROSE + guardrails.REVIEW_PASS_PROMPT
    assert guardrails.audit_task_text(user_text) == TASK_SHORT_PROSE


def test_pristine_task_text_passes_through_untouched():
    assert guardrails.audit_task_text(TASK_NUMBERED_STEPS) == TASK_NUMBERED_STEPS


def test_audit_bounces_a_prose_final_answer_once(tmp_path):
    """A prose final answer on an action task is a `done` in all but name; without
    this, every wrong-done that exits through the FINAL-ANSWER accept bypasses the
    audit entirely. Same latch as the done branch."""
    f = tmp_path / "out.py"
    script = [
        _tool_call("write", path=str(f), content="print('hi')\n"),
        _tool_call("bash", command=f"python {f}"),   # verify -> clears unverified
        "All done — the script prints hi.",           # prose end: bounced by the audit
        "Re-verified every requirement; all good.",   # prose end: accepted (latch)
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result == "Re-verified every requirement; all good."
    assert agent.engine._i == len(script)
    audits = [m for m in agent.messages if "done-audit" in m.get("content", "")]
    assert len(audits) == 1
    assert f"{f}: exists," in audits[0]["content"]


def test_final_answer_after_an_audited_done_is_accepted(tmp_path):
    """One bounce per turn TOTAL across both exits: an audited done followed by a prose
    final answer must not bounce again."""
    f = tmp_path / "out.py"
    script = [
        _tool_call("write", path=str(f), content="print('hi')\n"),
        _tool_call("bash", command=f"python {f}"),
        _tool_call("done", summary="first done"),     # bounced by the audit
        "Checked everything again; finished.",         # prose end: accepted, no re-bounce
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result == "Checked everything again; finished."
    audits = [m for m in agent.messages if "done-audit" in m.get("content", "")]
    assert len(audits) == 1


def test_final_answer_path_unchanged_with_lever_off(tmp_path, monkeypatch):
    """The OFF arm's prose accept is byte-identical to pre-lever behavior: no audit,
    and the recheck never applied to final answers."""
    monkeypatch.setenv("CHAD_DISABLE", "done_audit")
    f = tmp_path / "out.py"
    script = [
        _tool_call("write", path=str(f), content="print('hi')\n"),
        _tool_call("bash", command=f"python {f}"),
        "All done — the script prints hi.",
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result == "All done — the script prints hi."
    assert agent.engine._i == len(script)
    assert not any("done-audit" in m.get("content", "") for m in agent.messages)
    assert not any("every concrete deliverable" in m.get("content", "")
                   for m in agent.messages)


def test_subagent_done_is_never_audited(tmp_path):
    """Sub-agents carry no wall budget, so an audit inside one runs at infinite
    runway and can never decline — hence the audit is scoped to the top-level agent.
    The generic recheck applies instead, identical across A/B arms."""
    f = tmp_path / "out.py"
    script = [
        _tool_call("write", path=str(f), content="print('hi')\n"),
        _tool_call("bash", command=f"python {f}"),
        _tool_call("done", summary="first done"),     # recheck defers, audit must not
        _tool_call("done", summary="sub done"),
    ]
    agent = _agent(script, max_steps=10, subagent=True, subagent_tools="all")

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result == "sub done"
    assert not any("done-audit" in m.get("content", "") for m in agent.messages)
    assert any("every concrete deliverable" in m.get("content", "")
               for m in agent.messages)


# --- absent-path re-bounce -----------------------------------------------------------


def test_audit_rebounce_unit(tmp_path, monkeypatch):
    """audit_rebounce fires only for a still-absent path with runway to spare, and its
    lever gates it entirely."""
    gone = str(tmp_path / "missing.npy")
    there = tmp_path / "made.npy"
    there.write_text("x")
    # Still absent + unlimited runway -> the final-audit steer, naming the path.
    msg = guardrails.audit_rebounce([gone], None)
    assert msg and "final audit" in msg and gone in msg
    # Recovered path -> accept.
    assert guardrails.audit_rebounce([str(there)], None) is None
    # Runway too small -> accept (an audit that pushes into the wall converts a
    # wrong-done into a wall-death).
    assert guardrails.audit_rebounce([gone], 100.0) is None
    assert guardrails.audit_rebounce([gone], 500.0) is not None
    # Nothing recorded absent -> accept.
    assert guardrails.audit_rebounce([], None) is None
    # Lever off -> accept.
    monkeypatch.setenv("CHAD_DISABLE", "audit_absent_rebounce")
    assert guardrails.audit_rebounce([gone], None) is None


def test_audit_rebounces_done_when_task_path_still_absent(tmp_path):
    """The absent-deliverable failure: the first audit stat'ed the task's deliverable
    as ABSENT, promised acceptance anyway, and the file was never written. Now: one
    final bounce naming the missing path, then unconditional acceptance."""
    f = tmp_path / "out.py"   # named by the task, never written by the model
    g = tmp_path / "helper.py"
    script = [
        _tool_call("write", path=str(g), content="print(1)\n"),  # edit lands, wrong file
        _tool_call("bash", command=f"python {g}"),                # verified
        _tool_call("done", summary="first done"),    # audit bounce (path absent)
        _tool_call("done", summary="second done"),   # RE-bounce (still absent)
        _tool_call("done", summary="gave up"),       # accepted (cap 2 honored)
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result == "gave up"
    assert agent.engine._i == len(script)
    tool_notes = [m["content"] for m in agent.messages if m.get("role") == "tool"]
    assert sum("done-audit" in c for c in tool_notes) == 1
    rebounces = [c for c in tool_notes if "final audit" in c]
    assert len(rebounces) == 1 and str(f) in rebounces[0]
    # The first audit's promise stays truthful about the extra check.
    first = next(c for c in tool_notes if "done-audit" in c)
    assert "at most one further check" in first


def test_audit_no_rebounce_once_the_path_is_created(tmp_path):
    """A model that actually fixes the missing deliverable after the first bounce is
    accepted on its next done — no second interruption."""
    f = tmp_path / "out.py"
    g = tmp_path / "helper.py"
    script = [
        _tool_call("write", path=str(g), content="print(1)\n"),
        _tool_call("bash", command=f"python {g}"),
        _tool_call("done", summary="first done"),            # audit bounce (absent)
        _tool_call("write", path=str(f), content="print('hi')\n"),
        _tool_call("bash", command=f"python {f}"),           # verify the fix
        _tool_call("done", summary="fixed and done"),        # accepted, no rebounce
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result == "fixed and done"
    tool_notes = [m["content"] for m in agent.messages if m.get("role") == "tool"]
    assert not any("final audit" in c for c in tool_notes)


def test_audit_rebounce_lever_off_keeps_the_unconditional_promise(tmp_path, monkeypatch):
    """OFF arm: legacy wording and exactly one bounce, absent path or not."""
    monkeypatch.setenv("CHAD_DISABLE", "audit_absent_rebounce")
    f = tmp_path / "out.py"
    g = tmp_path / "helper.py"
    script = [
        _tool_call("write", path=str(g), content="print(1)\n"),
        _tool_call("bash", command=f"python {g}"),
        _tool_call("done", summary="first done"),    # audit bounce
        _tool_call("done", summary="second done"),   # accepted despite absent path
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result == "second done"
    tool_notes = [m["content"] for m in agent.messages if m.get("role") == "tool"]
    first = next(c for c in tool_notes if "done-audit" in c)
    assert "without further audit" in first
    assert not any("final audit" in c for c in tool_notes)


# --- Churn→audit handoff at the no-empty-diff hard stop -------------------------------

def test_handoff_steer_promise_is_conditional():
    """The handoff entry must not promise unconditional acceptance — on that path the
    next done still has to clear the no-empty-diff gate, and a promise the harness
    breaks spends the anti-spiral credibility the audit's contract depends on."""
    steer = guardrails.done_audit(
        "You must write the answer to /app/out.txt exactly.", _state(),
        entry="handoff")
    assert "cannot be accepted yet" in steer
    assert "once a change is landed and verified" in steer
    assert "one-time check before this `done` is accepted" not in steer
    assert "Then call done again; it will be accepted" not in steer
    # Default entry keeps the original accept-path wording byte-for-byte.
    legacy = guardrails.done_audit(
        "You must write the answer to /app/out.txt exactly.", _state())
    assert "one-time check before this `done` is accepted" in legacy


def test_handoff_bounces_an_empty_diff_done_then_hard_stops(tmp_path):
    """The audit-silent churn class: an empty-diff done on a path-naming action task
    gets ONE audit steer in context — with the absent-path fact — and the NEXT
    empty-diff done hard-stops exactly as before (churn capped, not replaced)."""
    f = tmp_path / "out.py"
    script = [
        _tool_call("bash", command="true"),          # real work, no landed edit
        _tool_call("done", summary="all set"),       # handoff bounce, not hard stop
        _tool_call("done", summary="really done"),   # empty-diff hard stop as before
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result.startswith("[stopped:")
    assert agent.budget_note
    audits = [m for m in agent.messages if "done-audit" in m.get("content", "")]
    assert len(audits) == 1
    assert "cannot be accepted yet" in audits[0]["content"]
    assert f"{f}: mentioned in the task, currently absent" in audits[0]["content"]


def test_handoff_then_landed_fix_is_accepted(tmp_path):
    """The conversion the handoff exists for: the steer's requirements land in context
    with the turn's work, the model applies + verifies the fix, and the next done
    passes every gate (latch forecloses a second audit)."""
    f = tmp_path / "out.py"
    script = [
        _tool_call("bash", command="true"),
        _tool_call("done", summary="all set"),       # handoff bounce
        _tool_call("write", path=str(f), content="print('hi')\n"),
        _tool_call("bash", command=f"python {f}"),   # verify -> clears unverified
        _tool_call("done", summary="fixed and verified"),
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result == "fixed and verified"
    assert agent.engine._i == len(script)
    audits = [m for m in agent.messages if "done-audit" in m.get("content", "")]
    assert len(audits) == 1


def test_handoff_bounces_a_prose_churn_ending_once(tmp_path):
    """Prose-ending twin: after the answer nudges spend, the ending that used to be
    the hard stop is bounced once by the audit; the following prose ending stops."""
    f = tmp_path / "out.py"
    script = [
        _tool_call("bash", command="true"),
        "The file is already correct.",   # answer nudge 1
        "It is complete.",                # answer nudge 2
        "Task complete.",                 # handoff bounce (was: hard stop)
        "Confirmed complete.",            # hard stop, as before the lever
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result.startswith("[stopped:")
    assert agent.engine._i == len(script)
    audits = [m for m in agent.messages if "done-audit" in m.get("content", "")]
    assert len(audits) == 1
    assert "cannot be accepted yet" in audits[0]["content"]


def test_handoff_lever_off_restores_the_immediate_hard_stop(tmp_path, monkeypatch):
    """OFF arm (ablation): the first empty-diff done hard-stops with no bounce —
    byte-identical to the pre-lever gate order."""
    monkeypatch.setenv("CHAD_DISABLE", "audit_churn_handoff")
    f = tmp_path / "out.py"
    script = [
        _tool_call("bash", command="true"),
        _tool_call("done", summary="all set"),   # immediate hard stop
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result.startswith("[stopped:")
    assert not any("done-audit" in m.get("content", "") for m in agent.messages)


def test_handoff_latch_is_shared_with_the_accept_path_audit(tmp_path):
    """One audit per turn TOTAL: an accept-path bounce forecloses the handoff (and
    vice versa) — a turn must never pay two audit steers."""
    f = tmp_path / "out.py"
    g = tmp_path / "helper.py"
    script = [
        _tool_call("write", path=str(g), content="print(1)\n"),
        _tool_call("bash", command=f"python {g}"),
        _tool_call("done", summary="first done"),    # accept-path audit bounce
        _tool_call("write", path=str(f), content="print('hi')\n"),
        _tool_call("bash", command=f"python {f}"),   # deliverable landed+verified
        _tool_call("done", summary="second done"),   # latch: accepted, no handoff
    ]
    agent = _agent(script, max_steps=10)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result == "second done"
    tool_notes = [m["content"] for m in agent.messages if m.get("role") == "tool"]
    assert sum("done-audit" in c for c in tool_notes) == 1
