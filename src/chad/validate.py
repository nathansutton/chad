"""Typed tool-call validation + self-repair (typia / autobe-inspired).

Weak local models fail tool calls in a small set of PREDICTABLE, RECOVERABLE
ways: a trailing comma, an integer sent as a string, a missing required field,
a wrong enum value, or a whole `arguments` object double-encoded as a JSON
string. The cloud fix (typia, https://typia.io) is a three-stage harness —

    lenient PARSE  ->  type COERCE  ->  typed VALIDATE

— and on failure it feeds the model back its OWN arguments annotated with
exactly which fields are wrong and what was expected, so it repairs only the
marked fields instead of regenerating blindly. typia reports this took Qwen
6.75% -> 100% on a hard nested-union schema.

This is the Python port over chad's EXISTING JSON-Schema tool definitions
(`tools.SCHEMAS`) — no schema duplication. The same dict that is sent to the
model as the tool contract is the validation contract, so the two can never
drift. Three entry points:

    repair_json(raw)            -> dict | None   (stage 1: lenient parse)
    coerce_and_validate(name, a)-> (args, errs)  (stages 2+3: coerce + validate)
    render_repair(name, a, errs)-> str           (the self-repair feedback)
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Tuple

from . import config
from .tools import active_schemas

# A/B knob (mirrors CHAD_NO_SYMBOLS), the single source of truth for both the
# typed-validate path here and the lenient tool-call parse in toolcall_parse.py.
# With CHAD_NO_VALIDATE set, callers bypass the typia-style lenient-parse +
# typed-validate + self-repair loop and fall back to strict json.loads + the terse
# missing-required check in `legacy_validate` below. Lets the eval harness measure
# exactly what the validation harness buys, per model.
VALIDATE = not config.flag("CHAD_NO_VALIDATE")

def _param_schema(name):
    """The `parameters` JSON-Schema for a tool as it is exposed to the model RIGHT NOW —
    chad's builtins plus every dynamically-appended tool (`activate_skill` when skills are
    installed, `task`, and any connected MCP server's tool). None if the name is not
    currently callable.

    This reads the LIVE `active_schemas()` set rather than a frozen import-time snapshot.
    The snapshot was the bug: dynamic tools like `activate_skill` are appended by
    `active_schemas()` (so the model sees them) but were absent from the frozen table, so a
    perfectly valid `activate_skill` call validated as an "unknown tool" — while the same
    error listed it as available (`_known_tools` reads the dispatch table, which *does*
    contain it). Sourcing both from `active_schemas()` guarantees the validation contract
    can never drift from what the model is shown."""
    for s in active_schemas():
        if s["function"]["name"] == name:
            return s["function"].get("parameters", {"type": "object"})
    from . import mcp
    return mcp.param_schema(name)


def _known_tools():
    """All currently-callable tool names, for the 'available tools' hint in unknown-tool
    repair messages. Sourced from the same live `active_schemas()` set as `_param_schema`
    so the hint can never advertise a name the validator would then reject (the exact
    inconsistency that produced 'unknown tool X ... Available: ...X...')."""
    return [s["function"]["name"] for s in active_schemas()]


def legacy_validate(name, args):
    """The terse pre-`coerce_and_validate` check, kept ONLY as the CHAD_NO_VALIDATE
    A/B baseline: unknown-tool / non-object-args / missing-required, no coercion or
    self-repair. Returns an error string, or None when the args pass."""
    if _param_schema(name) is None:
        return f"[unknown tool '{name}'. Available: {', '.join(_known_tools())}]"
    if not isinstance(args, dict):
        return f"[arguments for '{name}' must be a JSON object]"
    required = _param_schema(name).get("required", [])
    missing = [p for p in required if p not in args]
    if missing:
        return f"[tool '{name}' missing required argument(s): {', '.join(missing)}. Retry with them.]"
    return None


# ---------------------------------------------------------------------------
# Stage 1 — lenient JSON repair.
# Recovers the malformed-but-obvious JSON weak models emit. Each transform is
# applied additively and we re-attempt json.loads after the cheap ones, so a
# call that only needs one fix isn't risked by a later, more aggressive one.
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json|tool_call)?\s*|\s*```\s*$", re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_BARE_KEY = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*):")
_PY_CONST = re.compile(r"\b(True|False|None)\b")
_PY_MAP = {"True": "true", "False": "false", "None": "null"}


def _balance(s: str) -> str:
    """Close unterminated strings/brackets at end-of-string (truncated output)."""
    in_str = False
    esc = False
    stack: List[str] = []
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    if in_str:
        s += '"'
    return s + "".join(reversed(stack))


def repair_json(raw: Any) -> Optional[dict]:
    """Best-effort parse of a possibly-malformed JSON object. Returns the dict,
    or None if even the repaired text won't parse."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    # Plain parse first — the overwhelmingly common case, never mutated.
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    # Progressive repairs, cheapest first; re-try after each.
    s = _FENCE.sub("", s).strip()
    for transform in (
        lambda x: x,
        lambda x: _TRAILING_COMMA.sub(r"\1", x),
        lambda x: _PY_CONST.sub(lambda m: _PY_MAP[m.group(1)], x),
        lambda x: _BARE_KEY.sub(r'\1"\2"\3:', x),
        _balance,
    ):
        s = transform(s)
        try:
            v = json.loads(s)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Stages 2+3 — type coercion + typed validation, in one schema walk.
# ---------------------------------------------------------------------------


class Err:
    """One field-level validation failure: where, what was expected, what came."""

    __slots__ = ("path", "expected", "got")

    def __init__(self, path: str, expected: str, got: str):
        self.path = path or "$"
        self.expected = expected
        self.got = got

    def __str__(self) -> str:
        return f"{self.path}: expected {self.expected}, got {self.got}"


def _tname(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    if v is None:
        return "null"
    return type(v).__name__


def _expected(schema: dict) -> str:
    enum = schema.get("enum")
    if enum:
        return " | ".join(json.dumps(e) for e in enum)
    typ = schema.get("type", "value")
    if typ == "array":
        item = schema.get("items", {})
        return f"array<{item.get('type', 'value')}>"
    return typ


_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0"}


def _coerce_scalar(value: Any, typ: Optional[str]) -> Tuple[Any, bool]:
    """Return (coerced_value, ok). ok=False means it cannot be made to fit."""
    if typ == "integer":
        if isinstance(value, bool):
            return value, False
        if isinstance(value, int):
            return value, True
        if isinstance(value, float) and value.is_integer():
            return int(value), True
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            return int(value.strip()), True
        return value, False
    if typ == "number":
        if isinstance(value, bool):
            return value, False
        if isinstance(value, (int, float)):
            return value, True
        if isinstance(value, str):
            try:
                return float(value.strip()), True
            except ValueError:
                return value, False
        return value, False
    if typ == "boolean":
        if isinstance(value, bool):
            return value, True
        if isinstance(value, str):
            low = value.strip().lower()
            if low in _TRUE:
                return True, True
            if low in _FALSE:
                return False, True
        return value, False
    if typ == "string":
        if isinstance(value, str):
            return value, True
        # A scalar where a string was wanted (e.g. a path typed as a number) is
        # safe to stringify; containers are not.
        if isinstance(value, (int, float, bool)):
            return json.dumps(value) if isinstance(value, bool) else str(value), True
        return value, False
    # Unknown/unconstrained type: accept as-is.
    return value, True


def _walk(value: Any, schema: dict, path: str) -> Tuple[Any, List[Err]]:
    typ = schema.get("type")
    # Un-double-stringify: a container field whose value arrived as a JSON string
    # (Qwen3/Ornith do this on nested fields — typia's signature failure mode).
    if typ in ("object", "array") and isinstance(value, str):
        un = repair_json(value) if typ == "object" else _load_json(value)
        if un is not None:
            value = un

    if typ == "object":
        if not isinstance(value, dict):
            return value, [Err(path, "object", _tname(value))]
        props = schema.get("properties", {})
        required = schema.get("required", [])
        out: dict = {}
        errs: List[Err] = []
        for k, v in value.items():
            sub = props.get(k)
            if sub is None:  # unknown key — keep it, stay lenient
                out[k] = v
                continue
            cv, e = _walk(v, sub, f"{path}.{k}" if path else f"${k}")
            out[k] = cv
            errs += e
        for r in required:
            if r not in out:
                errs.append(Err(f"{path}.{r}" if path else f"${r}",
                                _expected(props.get(r, {})), "missing"))
        return out, errs

    if typ == "array":
        # A lone scalar where an array was wanted (focus="agent.py" instead of
        # ["agent.py"]) is the most common weak-model array mistake — wrap it into a
        # one-element list and let the item walk below validate/coerce it, rather than
        # burning a self-repair round-trip. A JSON-encoded array string was already
        # un-stringified above; a dict/None stays an error (genuine shape mismatch).
        if isinstance(value, (str, int, float, bool)):
            value = [value]
        if not isinstance(value, list):
            return value, [Err(path, "array", _tname(value))]
        item_schema = schema.get("items", {})
        out_l: list = []
        errs = []
        for i, item in enumerate(value):
            cv, e = _walk(item, item_schema, f"{path}[{i}]")
            out_l.append(cv)
            errs += e
        return out_l, errs

    # Scalar leaf.
    coerced, ok = _coerce_scalar(value, typ)
    if not ok:
        return value, [Err(path, _expected(schema), _tname(value))]
    enum = schema.get("enum")
    if enum is not None and coerced not in enum:
        return coerced, [Err(path, _expected(schema), json.dumps(coerced))]
    return coerced, []


def _load_json(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def coerce_and_validate(name: str, args: Any) -> Tuple[Any, List[Err]]:
    """Coerce loosely-typed args toward the schema and return (coerced, errors).
    An empty error list means `coerced` is safe to dispatch."""
    schema = _param_schema(name)
    if schema is None:
        return args, [Err("$", "a known tool", f"unknown tool {name!r}")]
    # Whole arguments value double-encoded as a string.
    if isinstance(args, str):
        un = repair_json(args)
        if un is not None:
            args = un
    if not isinstance(args, dict):
        return args, [Err("$", "object", _tname(args))]
    return _walk(args, schema, "")


# ---------------------------------------------------------------------------
# The self-repair feedback message.
# typia's lever: show the model its OWN arguments annotated with exactly which
# fields are wrong and what was expected, and tell it to fix ONLY those.
# ---------------------------------------------------------------------------


def render_repair(name: str, args: Any, errors: List[Err]) -> str:
    if _param_schema(name) is None:
        # A name with non-identifier characters ('grep</argstr') is not a naming
        # mistake — the model's call SYNTAX was garbled. Saying "unknown tool, pick
        # from this list" misdiagnoses it (the model re-emits the same garble; the
        # pytest-6202 death spiral). Show a worked example of the one format we
        # most reliably parse instead.
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*", str(name)):
            return (f"[malformed tool call — the tool name parsed as {name!r}, which "
                    "means your call syntax was garbled. Do not retry the same text. "
                    "Re-emit the call as ONE json object in this exact shape: "
                    '<tool_call>{"name": "grep", "arguments": {"pattern": "..."}}'
                    "</tool_call> — substituting your intended tool and arguments.]")
        return (f"[unknown tool {name!r}. Available tools: {', '.join(_known_tools())}. "
                "Re-emit the call using one of these names.]")
    lines = [f"[invalid arguments for `{name}` — fix ONLY the fields marked ✗ below, "
             "keep everything else, and re-emit the tool call]"]
    for e in errors:
        if e.got == "missing":
            lines.append(f"  ✗ {e.path}  → required, but missing (expected {e.expected})")
        else:
            lines.append(f"  ✗ {e.path} = {e.got}  → expected {e.expected}")
    try:
        echo = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
    except (TypeError, ValueError):
        echo = str({"name": name, "arguments": args})
    if len(echo) <= 600:
        lines.append("your call was:")
        lines.append(f"<tool_call>{echo}</tool_call>")
    return "\n".join(lines)
