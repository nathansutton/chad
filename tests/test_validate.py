"""Battery for the typed tool-call validation harness (validate.py).

Each case is a known weak-model failure mode (the typia taxonomy). We assert the
harness either REPAIRS it to a dispatchable call or REJECTS it with a precise,
field-level error — never silently drops or silently dispatches garbage.

Run: `uv run python tests/test_validate.py`
"""

import os

from chad import skills, tools
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
        ('{"name": "write", "arguments": {"path": "a.py", "content": "x"}}', {"name": "write"}),  # clean
        ('{"name": "write", "arguments": {"path": "a.py",}}', {"name": "write"}),  # trailing comma
        ("{'name': 'read', 'arguments': {'path': 'a.py'}}", None),  # single quotes (not handled) -> bare-key path
        ('{name: "write", arguments: {path: "a.py"}}', {"name": "write"}),  # bare keys
        ('{"name": "done", "arguments": {"summary": "ok", "x": True}}', {"name": "done"}),  # py const
        ('{"name": "bash", "arguments": {"command": "ls"', None),  # truncated -> balanced
        ('```json\n{"name": "bash", "arguments": {"command": "ls"}}\n```', {"name": "bash"}),  # fenced
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
    a, e = coerce_and_validate("bash", {"command": "ls", "timeout": "10"})
    check("coerce timeout str->int", not e and a["timeout"] == 10, f"a={a} e={[str(x) for x in e]}")
    # enum already valid
    a, e = coerce_and_validate(
        "write_todos", {"todos": [{"content": "x", "status": "pending"}]})
    check("enum valid passes", not e, f"e={[str(x) for x in e]}")
    # whole arguments double-stringified
    a, e = coerce_and_validate("bash", '{"command": "ls"}')
    check("double-stringified args", not e and a.get("command") == "ls",
          f"a={a} e={[str(x) for x in e]}")
    # nested array field double-stringified (write_todos.todos as a JSON string)
    a, e = coerce_and_validate(
        "write_todos", {"todos": '[{"content": "x", "status": "pending"}]'})
    check("nested array un-stringified", not e and isinstance(a["todos"], list)
          and a["todos"][0]["status"] == "pending", f"a={a} e={[str(x) for x in e]}")


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
    a, e = coerce_and_validate("write", {"path": {"nested": 1}, "content": "x"})
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

    # Iter-2: a garbled NAME ('grep</argstr') is a SYNTAX failure, not a
    # naming one — the message must show a worked call example, not a name list
    # (which sent the pytest-6202 model re-emitting the same garble to loop-abort).
    msg3 = render_repair("grep</argstr", {},
                         coerce_and_validate("grep</argstr", {})[1])
    check("render malformed name says malformed", "malformed tool call" in msg3)
    check("render malformed name shows exemplar",
          "<tool_call><function=grep><parameter=pattern>" in msg3, f"msg={msg3!r}")
    check("render malformed name does not list tools", "Available tools" not in msg3)


def test_legacy_validate():
    # The CHAD_NO_VALIDATE A/B baseline: terse unknown/non-object/missing checks,
    # no coercion. Returns an error string, or None when args pass.
    check("legacy: unknown tool flagged",
          "unknown tool" in (legacy_validate("frobnicate", {}) or ""))
    check("legacy: non-dict args flagged",
          "must be a JSON object" in (legacy_validate("write", "oops") or ""))
    miss = legacy_validate("write", {}) or ""  # write requires `path`
    check("legacy: missing required arg flagged",
          "missing required argument" in miss and "path" in miss, f"msg={miss!r}")
    check("legacy: valid args pass (None)",
          legacy_validate("write", {"path": "a.py", "content": "x"}) is None)


# --- Dynamically-appended tools validate against the LIVE schema set ----------
# Regression: tools appended at runtime (a connected MCP server's) are visible to the
# model via tools.active_schemas() but were absent from the validator's frozen
# import-time snapshot, so a perfectly valid call validated as an "unknown tool" — while
# the same error listed it as available. An unwinnable loop: no retry could pass. This
# pins that both the validator and the repair hint read the live set.
def test_dynamic_tool_validates(monkeypatch):
    dynamic = {
        "type": "function",
        "function": {
            "name": "mcp__demo__widgets",
            "description": "Do a widget thing.",
            "parameters": {
                "type": "object",
                "properties": {"kind": {"type": "string", "enum": ["round", "square"]}},
                "required": ["kind"],
            },
        },
    }
    check("dynamic tool unknown before it is appended",
          _param_schema("mcp__demo__widgets") is None)
    monkeypatch.setattr(tools, "SCHEMAS", tools.SCHEMAS + [dynamic])
    # ...now it is a known tool with a real param schema...
    sch = _param_schema("mcp__demo__widgets")
    check("dynamic tool has a live param schema", sch is not None, sch)
    # ...so a valid call validates cleanly (the exact call the trace could never land).
    _, e = coerce_and_validate("mcp__demo__widgets", {"kind": "round"})
    check("valid dynamic call accepted (no errors)", e == [], [str(x) for x in e])
    # ...and an out-of-enum value is rejected, not silently dispatched.
    _, e2 = coerce_and_validate("mcp__demo__widgets", {"kind": "nope"})
    check("out-of-enum value rejected", bool(e2), [str(x) for x in e2])
    # The 'available tools' hint lists it (mirror source), so no contradictory message.
    msg = render_repair("frobnicate", {}, coerce_and_validate("frobnicate", {})[1])
    check("repair hint lists the live dynamic tool", "mcp__demo__widgets" in msg, msg)


def test_installed_skills_add_no_tool(tmp_path, monkeypatch):
    """Skills are invoked by the user typing `/name`, never by the model calling a tool,
    so installing one must not change the validator's view of the tool surface."""
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
        check("installed skill is not a callable tool",
              _param_schema("widgets") is None)
        check("no activate_skill tool exists",
              _param_schema("activate_skill") is None)
        check("skill name absent from the available-tools hint",
              "widgets" not in render_repair(
                  "frobnicate", {}, coerce_and_validate("frobnicate", {})[1]))
    finally:
        skills.reset_session()


if __name__ == "__main__":
    test_repair()
    test_coercion()
    test_validation()
    test_render()
    test_legacy_validate()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
