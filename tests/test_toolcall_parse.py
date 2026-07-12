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


def test_hybrid_name_parameter_dialect():
    """Iter-3: quantized Ornith at temp 1.0 emits a HYBRID dialect — a JSON `{"name":"X"}`
    opener (often with no closing brace) followed by XML `<parameter=…>` blocks. It matches
    neither the `<function=…>` XML path nor the brace-matched JSON path, so 30 fully-formed
    calls were dropped silently in one TB2 run. Recover them (see _parse_hybrid_calls)."""
    from chad.toolcall_parse import _parse_hybrid_calls

    # 1. The exact observed shape: unclosed `{"name": "bash"` + one <parameter>, wrapped in
    #    <tool_call>/</function>/</tool_call> tag soup.
    h1 = ('<tool_call>{"name": "bash" <parameter=command> ls / </parameter> '
          '</function> </tool_call>')
    eq("h1 unclosed-brace bash", parse_tool_calls(h1), [("bash", {"command": "ls /"})])

    # 2. Closed brace `{"name": "write"}` + two params.
    h2 = ('<tool_call>{"name": "write"} <parameter=path> /app/f.py </parameter>'
          '<parameter=content> print(1) </parameter> </function> </tool_call>')
    eq("h2 write two params", parse_tool_calls(h2),
       [("write", {"path": "/app/f.py", "content": "print(1)"})])

    # 3. Int coercion still applies on the hybrid path.
    h3 = '{"name": "read" <parameter=path> a.py </parameter><parameter=offset> 40 </parameter>'
    res3 = parse_tool_calls(h3)
    eq("h3 read offset", res3, [("read", {"path": "a.py", "offset": 40})])
    check("h3 offset is int", isinstance(res3[0][1]["offset"], int))

    # 4. A bare `{"name":"X"}` with NO <parameter> block must NOT be taken by the hybrid
    #    path — it flows through the JSON path to (glob, {...}), unchanged behavior.
    eq("4 bare name no params -> json path",
       parse_tool_calls('{"name":"glob","arguments":{"pattern":"*.py"}}'),
       [("glob", {"pattern": "*.py"})])

    # 5. Two hybrid calls in one message, each owning its own params (scoped by the next
    #    name-opener).
    h5 = ('{"name": "bash" <parameter=command> pwd </parameter> '
          '{"name": "bash" <parameter=command> ls </parameter>')
    eq("h5 two hybrid calls scoped", _parse_hybrid_calls(h5),
       [("bash", {"command": "pwd"}), ("bash", {"command": "ls"})])

    # 6. Garbled tool name on the hybrid path is salvaged too.
    h6 = '{"name": "grep\\"" <parameter=pattern> foo </parameter>'
    eq("h6 salvaged name", parse_tool_calls(h6), [("grep", {"pattern": "foo"})])


def test_salvage_closed_block_unclosed_json():
    """TB2 count-dataset-tokens (2026-07-12 canary): the model re-emitted a complete
    bash call whose JSON args never closed, finished with XML-dialect closers, and
    properly closed the <tool_call> tag. _TAG_RE requires a trailing `}` so the block
    never reached repair_json; the call was dropped and the garble was accepted as a
    final answer with budget left. The salvage path strips the XML cruft and repairs."""
    # 1. The observed shape: unclosed JSON string + </parameter></function> tail.
    g1 = ('<tool_call>{"name": "bash", "arguments": {"command": "python3 -c \\"\\n'
          'print(1)\\n\\"\\n</parameter>\n</function>\n</tool_call>')
    res1 = parse_tool_calls(g1)
    check("g1 salvaged one call", len(res1) == 1 and res1[0][0] == "bash")
    check("g1 command survived", "print(1)" in res1[0][1].get("command", ""))

    # 2. Unclosed JSON but clean tag close, no XML cruft — still salvaged.
    g2 = '<tool_call>{"name": "read", "arguments": {"path": "a.py"</tool_call>'
    eq("g2 unclosed args salvaged", parse_tool_calls(g2),
       [("read", {"path": "a.py"})])

    # 3. An UNCLOSED <tool_call> must NOT be salvaged (mid-content truncation —
    #    running half a call is worse than the malformed-call nudge).
    g3 = '<tool_call>{"name": "write", "arguments": {"content": "def f(): ...'
    eq("g3 unclosed tag not salvaged", parse_tool_calls(g3), [])

    # 4. Well-formed calls are untouched by the salvage path (strict path wins).
    g4 = '<tool_call>{"name": "glob", "arguments": {"pattern": "*.py"}}</tool_call>'
    eq("g4 strict path unchanged", parse_tool_calls(g4),
       [("glob", {"pattern": "*.py"})])


if __name__ == "__main__":
    test_parse()
    test_salvage_garbled_tool_name()
    test_hybrid_name_parameter_dialect()
    test_salvage_closed_block_unclosed_json()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
