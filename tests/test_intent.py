"""Unit tests for agent.classify_intent — the answer-on-paper / verify nudge gate.

This logic decides whether a turn that ends WITHOUT an applied edit is a stall to
nudge ("you wrote code but didn't apply it") or a legitimate prose answer. It was the
crux of the "help me write test cases → answers on paper then stops" bug: the old
keyword list had no "write"/"test", so the nudge never fired. These cases pin the
classification down without needing to load a model.

Run: `uv run python test_intent.py`
"""

import os
import tempfile

from chad.agent import _has_open_tool_call, expand_mentions
from chad.prompt import _detect_test_command, build_system_prompt, classify_intent

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


# (prompt, want_action, want_read_only)
CASES = [
    # The screenshot bug: "write" + "test" must register as action.
    ("help me write additional test cases for the conform_name function in conform.py",
     True, False),
    ("write a test file test_mathx.py for add", True, False),
    # Classic action verbs still work.
    ("fix the off-by-one in inclusive_range", True, False),
    ("add a factorial function to mathx.py", True, False),
    ("refactor construct_addendum to be robust", True, False),
    ("document every function in util.py", True, False),  # 'document' is action
    # Casual / polite action requests — must still be action (not exempted as questions).
    ("can you fix the bug in greet.py?", True, False),
    ("could you add type hints to calc.py?", True, False),
    # Pure explanatory questions — read-only, nudge must stay silent.
    ("what does the parse_date function in dates.py do?", False, True),
    ("how does the cache eviction work?", False, True),
    ("explain the slugify function", False, True),
    # The negation trap: contains the action word "change" but explicitly forbids it.
    ("what does parse_date do, and what format? don't change any code", True, True),
    # 'edit' appears as a negation; action stays True (coarse keyword) but read_only
    # overrides so the nudge is suppressed anyway.
    ("walk me through auth.py, but do not edit anything", True, True),
    ("just explain how the retry loop works", False, True),
    # File-demand override (TB2 count-dataset-tokens): an explanatory opener that
    # ALSO demands the answer be written to a file is an action task, not read-only —
    # read_only=True here disarmed every no-progress gate and let a garbled final
    # step end the task with an empty diff.
    ('tell me how many tokens are in the dataset. write the integer number of tokens '
     'without spaces or commas (e.g. "1000000") to the file /app/answer.txt.',
     True, False),
    ("tell me the answer and save it to results.txt", True, False),
    # …but a plain explanatory ask (no file demand) stays read-only.
    ("tell me how the compaction trigger works", False, True),
    # …and an explicit negation still wins over the file words.
    ("how does save_session write to the file? don't change anything", True, True),
    # Scoped negation: a change-verb negation aimed at a SPECIFIC file or
    # definite object is a scope constraint on an action task, not a read-only ask —
    # TB2.1 overfull-hbox and query-optimize were classified read_only by these,
    # which disarmed the no-empty-diff gate, the recheck, and the done-audit on two
    # real wrong-done failures.
    ("Ensure that the LaTeX document main.tex compiles with no warnings. "
     "Do not edit main.tex or synonyms.txt.", True, False),
    ("Optimize the query. Do not modify the database file in any way. "
     "Please save your solution in the file /app/sol.sql.", True, False),
    ("clean up utils.py without changing the tests directory", True, False),
    # …while unscoped negations ("any code", "anything") stay global read-only
    # (action stays True — coarse keyword — but read_only suppresses the nudge).
    ("summarize this repo structure, do not modify anything", True, True),
    ("do not write anything to disk, only print the result", True, True),
]


def test_intent():
    for prompt, want_action, want_ro in CASES:
        got = classify_intent(prompt)
        check(f"action[{prompt[:40]}]", got["action"] == want_action,
              f"got action={got['action']} want {want_action}")
        check(f"read_only[{prompt[:40]}]", got["read_only"] == want_ro,
              f"got read_only={got['read_only']} want {want_ro}")
    # The decisive predicate the loop actually uses: fire the edit nudge iff
    # (not read_only) and action. Verify the screenshot case arms it and the
    # negation/question cases suppress it.
    def arms_nudge(p):
        i = classify_intent(p)
        return (not i["read_only"]) and i["action"]
    check("screenshot case arms nudge",
          arms_nudge("help me write additional test cases for conform_name"))
    check("negation suppresses nudge",
          not arms_nudge("what does parse_date do? don't change any code"))
    check("explain suppresses nudge",
          not arms_nudge("explain the slugify function"))


def test_run_intent():
    # Run-task class (plan 107 follow-up): system-state imperatives — completable with
    # zero file edits — must classify `run` so the anti-bail nudges arm, WITHOUT
    # feeding `action` (whose no-empty-diff done gate demands a landed edit).
    qemu = ("Start the /app/alpine.iso image in qemu in such a way that I can connect "
            "to it via `telnet 127.0.0.1 6665`. Start the image in the background and "
            "leave it running. Block until it's ready.")
    got = classify_intent(qemu)
    check("qemu-startup is a run task", got["run"], got)
    check("qemu-startup is NOT an action task (no edit demanded)",
          not got["action"], got)
    check("qemu-startup is not read-only", not got["read_only"], got)
    check("service imperative is a run task",
          classify_intent("install nginx and serve /var/www on port 80")["run"])
    # Word boundary: 'restarting'/'rerun' style inflections must not leak via substring.
    check("'brunch' does not classify as run",
          not classify_intent("describe the brunch menu parser")["run"])
    # Explanatory asks never classify run — 'explain how to start the server' ends in prose.
    check("explain-how-to-start is not a run task",
          not classify_intent("explain how to start the server")["run"])
    # An edit ask that also says 'run the tests' keeps action AND gains run — harmless,
    # the merged nudge predicate is already armed by action.
    both = classify_intent("fix the bug in app.py and run the tests")
    check("edit+run ask keeps action", both["action"] and both["run"], both)


def test_open_tool_call():
    # truncated mid-call -> True (drives the "write it in parts" nudge)
    check("open <tool_call> detected",
          _has_open_tool_call('thinking… <tool_call>{"name": "write", "arguments": {"content": "def f(): ...'))
    check("open <function= detected",
          _has_open_tool_call('<function=write><parameter=content>def f():\n    '))
    # complete or absent -> False
    check("closed tool_call not flagged",
          not _has_open_tool_call('<tool_call>{"name": "read"}</tool_call>'))
    check("plain prose not flagged",
          not _has_open_tool_call("Here's what the function does: it sums two numbers."))


def test_mentions():
    d = tempfile.mkdtemp(prefix="mentions_")
    cwd = os.getcwd()
    os.chdir(d)
    try:
        with open("geo.py", "w") as f:
            f.write("def area(w, h):\n    return w * h\n")
        # resolves a real file
        aug, att = expand_mentions("why is @geo.py slow?")
        check("mention resolves", att == ["geo.py"], f"att={att}")
        check("mention inlines content", "def area" in aug and "Attached" in aug)
        # trailing punctuation trimmed
        _, att2 = expand_mentions("see @geo.py.")
        check("trailing punct trimmed", att2 == ["geo.py"], f"att={att2}")
        # email is NOT a mention (no whitespace before @)
        _, att3 = expand_mentions("ping me at foo@bar.com about it")
        check("email not a mention", att3 == [], f"att={att3}")
        # nonexistent file -> no expansion
        txt = "look at @nope.py"
        aug4, att4 = expand_mentions(txt)
        check("nonexistent ignored", att4 == [] and aug4 == txt)
        # dedup
        _, att5 = expand_mentions("@geo.py and @geo.py again")
        check("dedup", att5 == ["geo.py"], f"att={att5}")
        # directory mention -> listing (not file content)
        os.makedirs("pkg", exist_ok=True)
        open("pkg/a.py", "w").close()
        open("pkg/b.py", "w").close()
        aug6, att6 = expand_mentions("what's in @pkg?")
        check("dir resolves", att6 == ["pkg"], f"att={att6}")
        check("dir lists entries", "a.py" in aug6 and "b.py" in aug6 and "directory listing" in aug6)
    finally:
        os.chdir(cwd)


def test_detect_test_command():
    d = tempfile.mkdtemp(prefix="testcmd_")
    cwd = os.getcwd()
    os.chdir(d)
    try:
        # No recognizable config -> "" (model falls back to generic guidance).
        check("nothing detected -> empty", _detect_test_command() == "")

        # CI yaml is authoritative — the exact failure from the trace: a `run:` step
        # invoking unittest via uv. The `run: ` prefix is excluded; runner kept.
        os.makedirs(".github/workflows", exist_ok=True)
        with open(".github/workflows/ci.yml", "w") as f:
            f.write("jobs:\n  test:\n    steps:\n"
                    "      - run: uv run python -m unittest discover ./tests\n")
        check("CI unittest detected",
              _detect_test_command() == "uv run python -m unittest discover ./tests",
              f"got {_detect_test_command()!r}")

        # CI wins over a pyproject pytest fallback when both are present.
        with open("pyproject.toml", "w") as f:
            f.write("[tool.pytest.ini_options]\n")
        check("CI takes priority over pyproject",
              _detect_test_command().startswith("uv run python -m unittest"))

        # Without CI, pytest config implies pytest; uv.lock selects the `uv run` prefix.
        os.remove(".github/workflows/ci.yml")
        check("pyproject pytest, no uv -> bare", _detect_test_command() == "python -m pytest")
        open("uv.lock", "w").close()
        check("pyproject pytest + uv.lock -> uv run",
              _detect_test_command() == "uv run python -m pytest")

        # A Makefile test target outranks the pyproject fallback.
        with open("Makefile", "w") as f:
            f.write("test:\n\tpytest -q\n")
        check("Makefile test target detected", _detect_test_command() == "make test")
    finally:
        os.chdir(cwd)


def test_non_utf8_project_docs_dont_crash():
    # A project doc / build file that isn't UTF-8 (a latin-1 CLAUDE.md with an é byte,
    # a latin-1 pyproject.toml) must NOT crash build_system_prompt — the once-
    # unguarded open().read() raised UnicodeDecodeError and the agent wouldn't construct.
    d = tempfile.mkdtemp(prefix="latin1_")
    cwd = os.getcwd()
    os.chdir(d)
    try:
        with open("CLAUDE.md", "wb") as f:
            f.write("# Guide\nCaf\xe9 rules - na\xefve bytes here.\n".encode("latin-1"))
        with open("pyproject.toml", "wb") as f:
            f.write("[tool.pytest.ini_options]\n# \xe9\n".encode("latin-1"))
        prompt = build_system_prompt()  # must not raise
        check("prompt built despite non-utf8 docs", isinstance(prompt, str) and len(prompt) > 0)
        check("latin-1 doc surfaced (bytes replaced, not crashed)", "Guide" in prompt)
    finally:
        os.chdir(cwd)


def test_plan_prefix():
    """Plan mode must ANSWER a question in prose, not manufacture a plan file (the
    '84-line plan for a 3-sentence tour' regression); a real change request still gets
    the plan-file mandate. _plan_prefix is the intent->preamble selector."""
    from chad.agent import _PLAN_PREFIX, _PLAN_PREFIX_CONCEPTUAL, _plan_prefix
    for q in ["give me a 3-sentence tour of this project",
              "how does the cache eviction work?",
              "what is the single most important file and why"]:
        pre = _plan_prefix(classify_intent(q))
        check(f"plan mode answers question in prose: {q[:34]!r}",
              pre is _PLAN_PREFIX_CONCEPTUAL, f"got {pre[:48]!r}")
    for a in ["add a --json flag to the cli",
              "refactor the engine cache eviction",
              "fix the off-by-one in inclusive_range"]:
        pre = _plan_prefix(classify_intent(a))
        check(f"plan mode writes a plan for a change: {a[:34]!r}",
              pre is _PLAN_PREFIX, f"got {pre[:48]!r}")


def test_clip_tool_result():
    """No single tool result should blow up the next turn's prefill: oversized output is
    truncated with a note (keeping the head); small output passes through untouched."""
    from chad.agent import _MAX_TOOL_RESULT_CHARS, _clip_tool_result
    small = "x" * 100
    check("small tool result passes through unchanged", _clip_tool_result(small) == small)
    big = "y" * (_MAX_TOOL_RESULT_CHARS + 5000)
    clipped = _clip_tool_result(big)
    check("oversized tool result is bounded + annotated",
          len(clipped) < len(big) and "truncated" in clipped, f"len={len(clipped)}")
    check("clip keeps the head", clipped.startswith("y" * 1000))
    tight = _clip_tool_result(big, cap=2000)
    check("explicit cap clips harder than the default",
          tight.startswith("y" * 2000) and not tight.startswith("y" * 2001)
          and "truncated" in tight)


def test_step_tool_cap():
    """A step's SEVERAL tool results must not stack into one giant prefill: the first
    call may use the whole per-step budget, later calls get only what's left, and every
    call keeps at least a floor-sized head (an edit/bash outcome is never swallowed)."""
    from chad.agent import (
        _MAX_TOOL_RESULT_CHARS,
        _STEP_TOOL_BUDGET_CHARS,
        _STEP_TOOL_FLOOR_CHARS,
        _step_tool_cap,
    )
    check("first call of a step gets the full single-call cap",
          _step_tool_cap(0) == _MAX_TOOL_RESULT_CHARS)
    check("later calls get the remaining step budget",
          _step_tool_cap(10000) == _STEP_TOOL_BUDGET_CHARS - 10000)
    check("a spent budget still leaves the floor",
          _step_tool_cap(_STEP_TOOL_BUDGET_CHARS) == _STEP_TOOL_FLOOR_CHARS)
    check("an overspent budget (floor overruns) still leaves the floor",
          _step_tool_cap(_STEP_TOOL_BUDGET_CHARS + 5000) == _STEP_TOOL_FLOOR_CHARS)


if __name__ == "__main__":
    test_intent()
    test_run_intent()
    test_open_tool_call()
    test_mentions()
    test_detect_test_command()
    test_non_utf8_project_docs_dont_crash()
    test_plan_prefix()
    test_clip_tool_result()
    test_step_tool_cap()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
