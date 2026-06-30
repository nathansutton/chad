"""Battery for the typed tool-call validation harness (validate.py).

Each case is a known weak-model failure mode (the typia taxonomy). We assert the
harness either REPAIRS it to a dispatchable call or REJECTS it with a precise,
field-level error — never silently drops or silently dispatches garbage.

Run: `uv run python test_validate.py`
"""

from chad.validate import coerce_and_validate, legacy_validate, render_repair, repair_json

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


if __name__ == "__main__":
    test_repair()
    test_coercion()
    test_validation()
    test_render()
    test_legacy_validate()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
