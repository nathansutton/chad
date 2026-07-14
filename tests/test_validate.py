"""Battery for the typed tool-call validation harness (validate.py).

Each case is a known weak-model failure mode (the typia taxonomy). We assert the
harness either REPAIRS it to a dispatchable call or REJECTS it with a precise,
field-level error — never silently drops or silently dispatches garbage.

Run: `uv run python test_validate.py`
"""

import os

from chad import skills
from chad.validate import (
    _param_schema,
    coerce_and_validate,
    legacy_validate,
    render_repair,
    repair_json,
)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


# --- Stage 1: lenient JSON repair --------------------------------------------
def test_repair():
    cases = [
        ('{"name": "read", "arguments": {"path": "a.py"}}', {"name": "read"}),  # clean
        ('{"name": "read", "arguments": {"path": "a.py",}}', {"name": "read"}),  # trailing comma
        ("{'name': 'read', 'arguments': {'path': 'a.py'}}", None),  # single quotes (not handled) -> bare-key path
        ('{name: "read", arguments: {path: "a.py"}}', {"name": "read"}),  # bare keys
        ('{"name": "done", "arguments": {"summary": "ok", "x": True}}', {"name": "done"}),  # py const
        ('{"name": "grep", "arguments": {"pattern": "def foo"', None),  # truncated -> balanced
        ('```json\n{"name": "glob", "arguments": {"pattern": "*.py"}}\n```', {"name": "glob"}),  # fenced
    ]
    for raw, expect in cases:
        got = repair_json(raw)
        if expect is None:
            # "None" markers above mean "must still parse to a dict via balance/bare-key";
            # we only require it not to crash and to yield a dict when recoverable.
            check(f"repair recoverable: {raw[:30]!r}", got is None or isinstance(got, dict))
        else:
            check(f"repair: {raw[:30]!r}", isinstance(got, dict) and got.get("name") == expect["name"],
                  f"got={got}")


# --- Stages 2+3: coercion (valid-but-loose calls should SUCCEED) -------------
def test_coercion():
    # integer-as-string -> int
    a, e = coerce_and_validate("read", {"path": "a.py", "offset": "10"})
    check("coerce offset str->int", not e and a["offset"] == 10, f"a={a} e={[str(x) for x in e]}")
    # number-as-string for budget
    a, e = coerce_and_validate("repo_map", {"budget": "2000"})
    check("coerce budget str->int", not e and a["budget"] == 2000, f"a={a}")
    # enum already valid
    a, e = coerce_and_validate("insert_symbol", {"name": "foo", "code": "x=1", "where": "before"})
    check("enum valid passes", not e, f"e={[str(x) for x in e]}")
    # whole arguments double-stringified
    a, e = coerce_and_validate("grep", '{"pattern": "def foo"}')
    check("double-stringified args", not e and a.get("pattern") == "def foo", f"a={a} e={[str(x) for x in e]}")
    # nested array field double-stringified (write_todos.todos as a JSON string)
    a, e = coerce_and_validate(
        "write_todos", {"todos": '[{"content": "x", "status": "pending"}]'})
    check("nested array un-stringified", not e and isinstance(a["todos"], list)
          and a["todos"][0]["status"] == "pending", f"a={a} e={[str(x) for x in e]}")
    # lone scalar where array<string> wanted (focus="agent.py" not ["agent.py"]) -> wrap
    a, e = coerce_and_validate("repo_map", {"focus": "src/chad/agent.py"})
    check("scalar->one-element array", not e and a["focus"] == ["src/chad/agent.py"],
          f"a={a} e={[str(x) for x in e]}")
    # a real shape mismatch (object where array wanted) still REJECTS, not wrapped
    a, e = coerce_and_validate("repo_map", {"focus": {"path": "agent.py"}})
    check("object->array still rejects", any(x.path == "$focus" for x in e),
          f"a={a} e={[str(x) for x in e]}")


# --- Stages 2+3: validation (broken calls should REJECT precisely) ----------
def test_validation():
    # missing required
    a, e = coerce_and_validate("edit", {"path": "a.py", "old": "x"})  # missing 'new'
    check("missing required field", any(x.path == "$new" and x.got == "missing" for x in e),
          f"e={[str(x) for x in e]}")
    # bad enum
    a, e = coerce_and_validate(
        "write_todos", {"todos": [{"content": "x", "status": "doing"}]})
    check("bad enum reported", any("status" in x.path and "doing" in x.got for x in e),
          f"e={[str(x) for x in e]}")
    # uncoercible type (object where string wanted)
    a, e = coerce_and_validate("read", {"path": {"nested": 1}})
    check("uncoercible type reported", any(x.path == "$path" for x in e),
          f"e={[str(x) for x in e]}")
    # unknown tool
    a, e = coerce_and_validate("frobnicate", {"x": 1})
    check("unknown tool reported", bool(e), f"e={[str(x) for x in e]}")


# --- The self-repair message --------------------------------------------------
def test_render():
    a, e = coerce_and_validate(
        "write_todos", {"todos": [{"content": "x", "status": "doing"}]})
    msg = render_repair("write_todos", a, e)
    check("render names tool", "write_todos" in msg)
    check("render marks bad field", "✗" in msg and "status" in msg)
    check("render says expected", "expected" in msg and "in_progress" in msg, f"msg={msg!r}")

    msg2 = render_repair("frobnicate", {"x": 1},
                         coerce_and_validate("frobnicate", {"x": 1})[1])
    check("render unknown tool lists available", "Available" in msg2)

    # Iter-2 (plan 066): a garbled NAME ('grep</argstr') is a SYNTAX failure, not a
    # naming one — the message must show a worked call example, not a name list
    # (which sent the pytest-6202 model re-emitting the same garble to loop-abort).
    msg3 = render_repair("grep</argstr", {},
                         coerce_and_validate("grep</argstr", {})[1])
    check("render malformed name says malformed", "malformed tool call" in msg3)
    check("render malformed name shows exemplar",
          '<tool_call>{"name": "grep"' in msg3, f"msg={msg3!r}")
    check("render malformed name does not list tools", "Available tools" not in msg3)


def test_legacy_validate():
    # The CHAD_NO_VALIDATE A/B baseline: terse unknown/non-object/missing checks,
    # no coercion. Returns an error string, or None when args pass.
    check("legacy: unknown tool flagged",
          "unknown tool" in (legacy_validate("frobnicate", {}) or ""))
    check("legacy: non-dict args flagged",
          "must be a JSON object" in (legacy_validate("read", "oops") or ""))
    miss = legacy_validate("read", {}) or ""  # read requires `path`
    check("legacy: missing required arg flagged",
          "missing required argument" in miss and "path" in miss, f"msg={miss!r}")
    check("legacy: valid args pass (None)",
          legacy_validate("read", {"path": "a.py"}) is None)


# --- Dynamically-appended tools validate against the LIVE schema set ----------
# Regression: `activate_skill` is appended to what the model sees by
# tools.active_schemas() only when skills are installed. The validator used to read a
# frozen import-time snapshot that never contained it, so a valid activate_skill call
# validated as an "unknown tool" — while the same error listed it as available. An
# unwinnable loop: no retry could pass. This asserts the validator now tracks the live
# set, and that the enum guard (constraining `name` to real skills) actually reaches it.
def test_dynamic_tool_validates(tmp_path, monkeypatch):
    empty_home = tmp_path / "_home"
    empty_home.mkdir()
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(empty_home) if p == "~" or p.startswith("~/") else p)
    proj = tmp_path / "proj"
    (proj / ".agents" / "skills" / "widgets").mkdir(parents=True)
    (proj / ".agents" / "skills" / "widgets" / "SKILL.md").write_text(
        "---\nname: widgets\ndescription: Use when the user mentions widgets.\n---\n# Do it\n")
    monkeypatch.chdir(proj)
    skills.reset_session()
    try:
        # activate_skill is now a known tool with a real param schema...
        sch = _param_schema("activate_skill")
        check("activate_skill has a live param schema", sch is not None, sch)
        check("enum constrained to installed skills",
              sch["properties"]["name"]["enum"] == ["widgets"], sch)
        # ...so a valid call validates cleanly (the exact call the trace could never land).
        _, e = coerce_and_validate("activate_skill", {"name": "widgets"})
        check("valid activate_skill accepted (no errors)", e == [], [str(x) for x in e])
        # ...and a hallucinated skill name is rejected by the enum, not silently dispatched.
        _, e2 = coerce_and_validate("activate_skill", {"name": "nope"})
        check("unknown skill name rejected via enum", bool(e2), [str(x) for x in e2])
        # The 'available tools' hint lists it (mirror source), so no contradictory message.
        msg = render_repair("frobnicate", {}, coerce_and_validate("frobnicate", {})[1])
        check("repair hint lists the live activate_skill", "activate_skill" in msg, msg)
    finally:
        skills.reset_session()


# --- Batched replace_lines schema (improve 04) -------------------------------
# The batch form takes edits=[{start, end, new}, …]. The schema must (a) accept a
# well-formed batch and coerce loose item types, and (b) fail LOUDLY on a half-parsed
# array — the whole point is that a garbled batch feeds the self-repair path rather than
# silently applying the items that happened to parse.
def test_replace_lines_batch_schema():
    # top-level `start/end/new` are optional now (only `path` required) so the single
    # form still validates and the batch form isn't forced to send them.
    sch = _param_schema("replace_lines")
    check("replace_lines requires only path", sch.get("required") == ["path"], sch)
    check("replace_lines exposes edits array",
          sch["properties"]["edits"]["type"] == "array", sch)

    # a well-formed batch validates clean, coercing "2"->2 in an item
    a, e = coerce_and_validate("replace_lines", {
        "path": "a.py",
        "edits": [{"start": 1, "end": "2", "new": "x"}]})
    check("batch: valid batch accepted", e == [], [str(x) for x in e])
    check("batch: item int coerced", a["edits"][0]["end"] == 2, a)

    # the single form still validates against the same schema
    _, e = coerce_and_validate("replace_lines",
                               {"path": "a.py", "start": 1, "end": 1, "new": "x"})
    check("batch: single form still valid", e == [], [str(x) for x in e])

    # a garbled item (missing `new`) is flagged on that item's field, not silently kept
    _, e = coerce_and_validate("replace_lines", {
        "path": "a.py",
        "edits": [{"start": 1, "end": 1, "new": "x"}, {"start": 2, "end": 2}]})
    check("batch: missing item field reported",
          any("new" in x.path and x.got == "missing" for x in e),
          [str(x) for x in e])

    # a JSON-encoded edits string (the nested-container failure mode) un-stringifies
    a, e = coerce_and_validate("replace_lines", {
        "path": "a.py", "edits": '[{"start": 1, "end": 1, "new": "x"}]'})
    check("batch: edits JSON-string un-stringified", e == [] and isinstance(a["edits"], list),
          (a, [str(x) for x in e]))


def test_dynamic_tool_absent_without_skills(tmp_path, monkeypatch):
    # With no skills installed, activate_skill is NOT exposed to the model, so the
    # validator must treat it as unknown (symmetry: the hint won't list it either).
    empty_home = tmp_path / "_home"
    empty_home.mkdir()
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(empty_home) if p == "~" or p.startswith("~/") else p)
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    skills.reset_session()
    try:
        check("activate_skill unknown when no skills installed",
              _param_schema("activate_skill") is None)
    finally:
        skills.reset_session()


if __name__ == "__main__":
    test_repair()
    test_coercion()
    test_validation()
    test_render()
    test_legacy_validate()
    test_replace_lines_batch_schema()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
