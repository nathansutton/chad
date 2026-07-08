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


def _parse_xml_calls(text: str):
    calls = []
    for fm in _XML_FUNC_RE.finditer(text):
        name = salvage_tool_name(fm.group(1).strip())
        args = {}
        for pm in _XML_PARAM_RE.finditer(fm.group(2)):
            key = pm.group(1).strip()
            val = pm.group(2).strip()
            if key in _INT_PARAMS and val.lstrip("-").isdigit():
                val = int(val)
            args[key] = val
        if name:
            calls.append((name, args))
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
    calls = []
    seen = set()
    candidates = [m.group(1) for m in _TAG_RE.finditer(text)]
    candidates += [m.group(1) for m in _FENCE_RE.finditer(text)]
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
