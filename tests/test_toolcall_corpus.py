"""The tool-call corpus: raw assistant turns a 27B actually emits (or plausibly could),
each asserted to parse to the intended call, or to a rejection naming the exact fault.

The syntax gate looked fine in unit tests and was wrong 18 times out of 19 in the field
until a corpus of real files found the gaps (PR #121). This is the same treatment for
the other format-fragile surface: `tests/fixtures/toolcall_corpus/<shape>.turn.txt` is
the turn, byte for byte (CRLF files included), and `<shape>.expect.json` is what the
agent's own pipeline — parse, name and shape aliases, coerce + validate, repair message — must
make of it. Nine turns are verbatim from the 2026-10-10 slip census.

`known-gaps/` holds shapes whose expectation is what SHOULD happen; each carries a
`reason` it does not yet, and the test asserts the gap is still open, so a parser change
that closes one has to move the file out (and a change that opens one fails here).

expect.json: {"note", "calls": [{"name", "arguments"?, "argument_keys"?,
"argument_lengths"?, "errors": [paths], "unknown_keys"?}], "today"?, "reason"?}.
An empty `calls` means nothing may run (truncated or prose).
"""
import json
import os
import re
import time

from chad.toolcall_parse import parse_tool_calls
from chad.tools import alias_by_shape, alias_to_bash, alias_write_edit
from chad.validate import coerce_and_validate, render_repair

_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "toolcall_corpus")
_UNKNOWN_KEY_RE = re.compile(r"✗ (\w+) → not a parameter of")


def pipeline(turn: str) -> list:
    """What the agent does with one assistant turn, minus dispatch."""
    out = []
    for name, args in parse_tool_calls(turn):
        name, args = alias_to_bash(name, args)
        renamed, args = alias_write_edit(name, args)
        if renamed == name:
            renamed, args = alias_by_shape(name, args)
        name = renamed
        coerced, errs = coerce_and_validate(name, args)
        msg = render_repair(name, coerced, errs) if errs else ""
        out.append({"name": name, "arguments": coerced,
                    "errors": sorted(e.path for e in errs),
                    "unknown_keys": _UNKNOWN_KEY_RE.findall(msg)})
    return out


def mismatches(expected: list, got: list) -> list:
    """Human-readable differences between an expectation and the pipeline's output."""
    out = []
    if len(expected) != len(got):
        return [f"{len(expected)} call(s) expected, {len(got)} parsed: {[g['name'] for g in got]}"]
    for i, (e, g) in enumerate(zip(expected, got)):
        tag = f"call {i} ({e['name']})"
        if e["name"] != g["name"]:
            out.append(f"{tag}: name {g['name']!r}")
        if "arguments" in e:
            for k, v in e["arguments"].items():
                if g["arguments"].get(k) != v:
                    out.append(f"{tag}: {k}={g['arguments'].get(k)!r}, expected {v!r}")
            if "argument_keys" not in e and "argument_lengths" not in e \
                    and sorted(g["arguments"]) != sorted(e["arguments"]):
                out.append(f"{tag}: keys {sorted(g['arguments'])}, expected {sorted(e['arguments'])}")
        if "argument_keys" in e and sorted(g["arguments"]) != sorted(e["argument_keys"]):
            out.append(f"{tag}: keys {sorted(g['arguments'])}, expected {sorted(e['argument_keys'])}")
        for k, n in e.get("argument_lengths", {}).items():
            if len(g["arguments"].get(k, "")) != n:
                out.append(f"{tag}: len({k})={len(g['arguments'].get(k, ''))}, expected {n}")
        if sorted(e.get("errors", [])) != g["errors"]:
            out.append(f"{tag}: errors {g['errors']}, expected {sorted(e.get('errors', []))}")
        if "unknown_keys" in e and sorted(e["unknown_keys"]) != sorted(g["unknown_keys"]):
            out.append(f"{tag}: unknown keys {g['unknown_keys']}, expected {e['unknown_keys']}")
    return out


def _walk():
    for root, _, files in sorted(os.walk(_CORPUS)):
        gap = os.path.basename(root) == "known-gaps"
        for stored in sorted(files):
            if not stored.endswith(".turn.txt"):
                continue
            shape = stored.removesuffix(".turn.txt")
            with open(os.path.join(root, stored), "rb") as f:
                turn = f.read().decode("utf-8")
            with open(os.path.join(root, shape + ".expect.json"), encoding="utf-8") as f:
                expect = json.load(f)
            yield shape, gap, turn, expect


def test_every_turn_in_the_corpus_parses_to_the_intended_calls():
    t0 = time.perf_counter()
    failures = []
    seen = gaps = 0
    for shape, gap, turn, expect in _walk():
        seen += 1
        diffs = mismatches(expect["calls"], pipeline(turn))
        if gap:
            gaps += 1
            assert expect.get("reason"), f"known-gaps/{shape} has no reason"
            if not diffs:
                failures.append(f"known-gaps/{shape}: the gap is closed — move it out of known-gaps/")
        elif diffs:
            failures.append(f"{shape}: " + "; ".join(diffs))
    elapsed = time.perf_counter() - t0
    assert not failures, "\n".join(failures)
    assert seen >= 60 and gaps >= 1, (seen, gaps)
    assert elapsed < 2.0, f"corpus took {elapsed:.2f}s"


def test_the_pipeline_matches_the_agents_own_order():
    """The corpus is only evidence if `pipeline` does what run_turn does: read alias,
    then write/edit name alias, then shape alias (only when the name alias did nothing),
    then coerce + validate. A `write_file` with `file_text` exercises both the alias and
    the key rename before validation sees it; a `write` with only `todos` the shape."""
    got = pipeline("<function=write_file><parameter=file_path>a</parameter>"
                   "<parameter=file_text>x</parameter></function>")
    assert got == [{"name": "write", "arguments": {"path": "a", "content": "x"},
                    "errors": [], "unknown_keys": []}]
    got = pipeline("<function=write><parameter=todos>[]</parameter></function>")
    assert got[0]["name"] == "write_todos" and got[0]["errors"] == []
