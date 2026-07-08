"""Characterization tests for the tool-call parser (agent.py).

`parse_tool_calls` is the boundary between raw model text and tool dispatch — the
single most format-fragile surface in the project. It parses three JSON dialects
(templated <tool_call>, ```json fences, bare top-level JSON) plus an XML
<function=…> dialect, with int-coercion, double-stringified-args recovery, a
`parameters` alias, dedup, and <think> stripping.

These tests PIN the CURRENT behavior (including quirks). Every expected value was
derived by reading the live implementation, not by assuming any spec. They run in
the default VALIDATE=on path (CHAD_NO_VALIDATE unset) — do not set it.

Run: `.venv/bin/python test_toolcall_parse.py`
"""

import json

from chad.toolcall_parse import _parse_xml_calls, parse_tool_calls, strip_think

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


def eq(name, got, expected):
    check(name, got == expected, f"got={got!r} expected={expected!r}")


# --- The 11 characterization cases -------------------------------------------
def test_parse():
    # 1. Clean templated <tool_call> JSON.
    eq(
        "1 templated call",
        parse_tool_calls('<tool_call>{"name": "read", "arguments": {"path": "a.py"}}</tool_call>'),
        [("read", {"path": "a.py"})],
    )

    # 2. ```json fence.
    fenced = '```json\n{"name": "write", "arguments": {"path": "b.py", "content": "x"}}\n```'
    eq("2 fenced json", parse_tool_calls(fenced), [("write", {"path": "b.py", "content": "x"})])

    # 3. Bare top-level JSON object (no tags, no fence) -> _iter_json_objects fallback.
    eq(
        "3 bare json",
        parse_tool_calls('{"name":"glob","arguments":{"pattern":"*.py"}}'),
        [("glob", {"pattern": "*.py"})],
    )

    # 4. XML dialect, with _INT_PARAMS coercion: offset must be int 10, not "10".
    res4 = parse_tool_calls(
        "<function=read><parameter=path>a.py</parameter><parameter=offset>10</parameter></function>"
    )
    eq("4 xml dialect", res4, [("read", {"path": "a.py", "offset": 10})])
    check("4 offset is int", res4[0][1]["offset"] == 10 and isinstance(res4[0][1]["offset"], int),
          f"got {res4[0][1]['offset']!r}")
    check("4 offset not str", not isinstance(res4[0][1]["offset"], str))

    # 5. XML takes priority: text containing BOTH an XML call and a <tool_call> JSON
    #    returns the XML parse only (the JSON glob call is ignored).
    both = (
        "<function=read><parameter=path>a.py</parameter></function>\n"
        '<tool_call>{"name":"glob","arguments":{"pattern":"*.py"}}</tool_call>'
    )
    eq("5 xml priority", parse_tool_calls(both), [("read", {"path": "a.py"})])

    # 6. <think> wrapper stripped, call still parsed (both think forms).
    eq(
        "6a explicit think block",
        parse_tool_calls('<think>let me read</think><tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>'),
        [("read", {"path": "a.py"})],
    )
    eq(
        "6b leading-close think",
        parse_tool_calls('reasoning here</think><tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>'),
        [("read", {"path": "a.py"})],
    )
    eq("6c strip_think explicit", strip_think("<think>x</think>y"), "y")
    eq("6d strip_think leading-close", strip_think("reasoning</think>answer"), "answer")

    # 7. Double-stringified `arguments` value recovered to a dict.
    inner = json.dumps({"path": "a", "old": "x", "new": "y"})
    outer = json.dumps({"name": "edit", "arguments": inner})
    eq("7 double-stringified args", parse_tool_calls(outer),
       [("edit", {"path": "a", "old": "x", "new": "y"})])

    # 8. `parameters` alias instead of `arguments`.
    eq(
        "8 parameters alias",
        parse_tool_calls('<tool_call>{"name":"grep","parameters":{"pattern":"foo"}}</tool_call>'),
        [("grep", {"pattern": "foo"})],
    )

    # 9. Two independent calls, in order.
    two = (
        '<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>\n'
        '<tool_call>{"name":"glob","arguments":{"pattern":"*.py"}}</tool_call>'
    )
    eq("9 two calls in order", parse_tool_calls(two),
       [("read", {"path": "a.py"}), ("glob", {"pattern": "*.py"})])

    # 10. Dedup: the exact same <tool_call> JSON twice -> one tuple (the `seen` set).
    dup = '<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>'
    eq("10 dedup identical", parse_tool_calls(dup + dup), [("read", {"path": "a.py"})])

    # 11. Garbage / prose with no JSON and no XML -> [].
    eq("11 no call", parse_tool_calls("I think you should read the file first."), [])

    # Bonus: _parse_xml_calls directly (the XML helper, with int coercion).
    eq(
        "x _parse_xml_calls direct",
        _parse_xml_calls("<function=glob><parameter=pattern>*.py</parameter><parameter=limit>5</parameter></function>"),
        [("glob", {"pattern": "*.py", "limit": 5})],
    )


def test_salvage_garbled_tool_name():
    """Iter-2 (plan 066): quantized greedy decode garbles the CLOSE of a call
    (observed: `grep</argstr`, `grep"`). The leading identifier is salvaged when it
    names a known tool, so the call dispatches instead of bouncing off 'unknown
    tool' until the loop guard kills the turn (the pytest-6202 death spiral)."""
    from chad.toolcall_parse import salvage_tool_name

    # Pure helper.
    eq("s1 xml garble", salvage_tool_name("grep</argstr"), "grep")
    eq("s2 trailing quote", salvage_tool_name('edit"'), "edit")
    eq("s3 known name untouched", salvage_tool_name("replace_symbol"), "replace_symbol")
    eq("s4 unknown garbage untouched", salvage_tool_name("zzz</argstr"), "zzz</argstr")

    # Through the XML dialect (the observed pytest-6202 shape).
    eq(
        "s5 xml call salvaged",
        _parse_xml_calls("<function=grep</argstr><parameter=pattern>foo</parameter></function>"),
        [("grep", {"pattern": "foo"})],
    )

    # Through the JSON dialect.
    eq(
        "s6 json call salvaged",
        parse_tool_calls('<tool_call>{"name": "grep\\"", "arguments": {"pattern": "x"}}</tool_call>'),
        [("grep", {"pattern": "x"})],
    )


if __name__ == "__main__":
    test_parse()
    test_salvage_garbled_tool_name()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
