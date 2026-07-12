"""Pure tool-call parser for chad (extracted from agent.py).

`parse_tool_calls` is the boundary between raw model text and tool dispatch — the
single most format-fragile surface in the project. Local models are inconsistent
about tool-call formatting (lesson from prior art like opencode): some emit JSON
inside <tool_call> tags, some emit ```json fences, some emit a bare JSON object,
and some (Qwen3 / Ornith / GLM thinking models) emit an XML function-call dialect:
<function=name><parameter=key>value</parameter></function>. Parse them all,
de-duplicated.

This module is pure (no model, no I/O) so it can be unit-tested directly — see
test_toolcall_parse.py. `agent` re-exports these names so existing importers keep
working unchanged.
"""

import json
import re

from .validate import (  # VALIDATE: single source of truth in validate.py
    VALIDATE,
    _known_tools,
    repair_json,
)

_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json|tool_call)?\s*(\{.*?\})\s*```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_XML_FUNC_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)
_INT_PARAMS = {"offset", "limit", "timeout"}
# Hybrid dialect: a JSON-style `{"name": "bash"` opener followed by XML-style
# `<parameter=…>…</parameter>` blocks — NOT the `<function=name>` form. Quantized Ornith
# emits this constantly under temp-1.0 sampling (30 occurrences in one TB2 run), most
# often as `<tool_call>{"name": "bash" <parameter=command>…</parameter></function>`.
# Neither the JSON path (the `{…}` never closes, so brace-matching yields nothing) nor the
# XML path (no `<function=`) matches it, so a fully-specified call — command and all — was
# being dropped silently, wasting the step and a re-prefill. The opener tolerates a missing
# closing brace and single or double quotes.
_HYBRID_NAME_RE = re.compile(r"""\{\s*["']name["']\s*:\s*["']([^"']+)["']""")
# Lenient variant of _TAG_RE for salvage: a CLOSED <tool_call> block whose interior
# starts as JSON but doesn't end with `}` — the observed TB2 garble is JSON args that
# never close, followed by XML-dialect closers (`…\n</parameter>\n</function>\n
# </tool_call>`). _TAG_RE requires a trailing `}` so these never even reach
# repair_json; count-dataset-tokens (2026-07-12 canary) died exactly here: a complete,
# runnable bash command dropped, then the turn accepted as a final answer. Closed
# blocks only — an UNCLOSED block usually means generation truncated mid-content, and
# running a half-emitted call is worse than the existing malformed-call nudge.
_TAG_SALVAGE_RE = re.compile(r"<tool_call>\s*(\{.*?)\s*</tool_call>", re.DOTALL)
# Trailing XML-ish cruft to strip from a salvaged interior before JSON repair.
_XML_CRUFT_RE = re.compile(r"(?:\s*</?(?:parameter|function|tool_call)[^>]*>)+\s*$")


def strip_think(text: str) -> str:
    """Remove reasoning. Handles both an explicit <think>...</think> block and the
    template-opened case, where generation starts inside <think> so the text has a
    leading </think> with no opening tag."""
    o = text.find("<think>")
    c = text.find("</think>")
    if c != -1 and (o == -1 or c < o):  # leading close => everything up to it is reasoning
        text = text[c + len("</think>"):]
    return _THINK_RE.sub("", text)


_IDENT_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def salvage_tool_name(name: str) -> str:
    """Recover a tool name from trailing garbage. Quantized local models garble the
    CLOSE of a call — observed on the llama backend as `grep</argstr` and `grep"` —
    and the raw string then validates as an 'unknown tool', which tells a model with
    a *syntax* problem that it has a *naming* problem (it retried the same garbage
    to the loop-abort; pytest-6202 died exactly this way). If the name as given is
    not a known tool but its leading identifier run is, use that. Names that are
    already valid (including MCP names with dots/dashes) pass through untouched."""
    known = _known_tools()
    if name in known:
        return name
    m = _IDENT_RE.match(name)
    if m and m.group(0) in known:
        return m.group(0)
    return name


def _parse_params(body: str) -> dict:
    """Parse `<parameter=key>value</parameter>` blocks out of `body` into an args dict,
    with the same int-coercion as the XML dialect. Shared by the `<function=…>` and the
    hybrid `{"name":…}`+`<parameter>` parsers."""
    args = {}
    for pm in _XML_PARAM_RE.finditer(body):
        key = pm.group(1).strip()
        val = pm.group(2).strip()
        if key in _INT_PARAMS and val.lstrip("-").isdigit():
            val = int(val)
        args[key] = val
    return args


def _parse_xml_calls(text: str):
    calls = []
    for fm in _XML_FUNC_RE.finditer(text):
        name = salvage_tool_name(fm.group(1).strip())
        if name:
            calls.append((name, _parse_params(fm.group(2))))
    return calls


def _parse_hybrid_calls(text: str):
    """Parse the hybrid `{"name": "X"}` + `<parameter=…>` dialect (see _HYBRID_NAME_RE).
    Each name-opener owns the `<parameter>` blocks between it and the next opener (the same
    scoping the `<function=…>` block gives its params). Only emits a call when the scope
    actually contains a `<parameter>` block — a bare `{"name":"X"}` with no params is left
    to the JSON path, so this never double-counts a plain JSON call."""
    opens = list(_HYBRID_NAME_RE.finditer(text))
    if not opens:
        return []
    calls = []
    for i, m in enumerate(opens):
        scope_end = opens[i + 1].start() if i + 1 < len(opens) else len(text)
        body = text[m.end():scope_end]
        if "<parameter=" not in body:
            continue
        name = salvage_tool_name(m.group(1).strip())
        if name:
            calls.append((name, _parse_params(body)))
    return calls


def _iter_json_objects(text: str):
    """Yield top-level {...} JSON spans via brace matching (handles strings)."""
    depth = 0; start = None; in_str = False; esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start : i + 1]


def parse_tool_calls(text: str):
    text = strip_think(text)
    # XML function-call dialect takes priority when present (thinking models).
    xml_calls = _parse_xml_calls(text)
    if xml_calls:
        return xml_calls
    # Hybrid `{"name":…}`+`<parameter>` dialect: only when real <parameter> blocks exist,
    # so a plain JSON call still flows through the JSON path below unchanged. Sits above
    # the JSON fallback because that path would extract just the name (empty args) from the
    # unclosed `{"name":"bash"` opener and drop every parameter.
    if "<parameter=" in text:
        hybrid = _parse_hybrid_calls(text)
        if hybrid:
            return hybrid
    calls = []
    seen = set()
    candidates = [m.group(1) for m in _TAG_RE.finditer(text)]
    candidates += [m.group(1) for m in _FENCE_RE.finditer(text)]
    if not candidates:
        # Salvage: closed <tool_call> blocks whose JSON never closed (garbled tail,
        # e.g. `…\n</parameter>\n</function>\n</tool_call>`). Strip the XML cruft and
        # let repair_json below reconstruct the call.
        candidates = [_XML_CRUFT_RE.sub("", m.group(1))
                      for m in _TAG_SALVAGE_RE.finditer(text)]
    if not candidates:  # last resort: any bare top-level JSON object
        candidates = list(_iter_json_objects(text))
    for raw in candidates:
        raw = raw.strip()
        if raw in seen:
            continue
        seen.add(raw)
        # Lenient repair (typia stage 1): recover trailing commas, bare keys,
        # Python literals, truncated/unbalanced output — so a malformed call is
        # repaired and surfaced for validation, not silently dropped (which left
        # the model with no tool result and no idea what went wrong).
        if VALIDATE:
            obj = repair_json(raw)
        else:
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                obj = None
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not name:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):  # whole arguments value double-stringified
            args = repair_json(args) or args
        if not isinstance(args, dict):
            args = {}
        calls.append((salvage_tool_name(str(name)), args))
    return calls
