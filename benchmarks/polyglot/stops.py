"""What ended a trial, and how its wall was spent: the `stop` object of a trial row.

    {"kind": "wall-think", "step": 6, "where": {"decode_s": 1050.1, "prefill_s": 22.4, "tool_s": 127.5}}

A row's `capped` flag says the wall cap ended the trial; it does not say what the agent
was doing when it did. A think that ran 20 minutes, a `go test` hung on a deadlocked
channel program and a llama-server that stopped answering all read `capped: true`, and
a paired comparison with a six-flip threshold cannot afford an infrastructure failure
counted as a model failure. `kind` is one of:

    done           the agent ended the turn itself (a `done` call, or a final answer)
    wall-think     the cap fell while the agent was decoding inside an open <think>
    wall-act       ... while decoding the action after its reasoning closed
    wall-tool      ... while a tool it had dispatched was still running
    wall-prefill   ... after a step finished and before the next produced a token
    step-cap, budget, loop, repetition, done-rejected, no-change
                   the agent's own stops, as `Agent.stop_kind` names them
    error          a CLI arm exited non-zero on its own
    infra          the engine or server failed: not the model's trial, and redone on resume
    unknown        a foreign harness the wall cap ended, whose trajectory cannot say where

`step` is the agent step that was running or had just finished at the end. `where`
splits the trial's seconds into decode, prefill and the rest (tools, and the runner
between steps), from the per-step metrics the ATIF trajectory carries; a step still in
flight at the kill has no measured decode time, so its share is the remainder. A
foreign harness's trajectory carries the server's counts but not its seconds, so its
`where` is the wall alone. Stdlib only.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catalog import JsonValue, is_array, is_number, is_object, is_text  # noqa: E402

KINDS = ("done", "wall-think", "wall-act", "wall-tool", "wall-prefill", "step-cap", "budget",
         "loop", "repetition", "done-rejected", "no-change", "error", "infra", "unknown")
INFRA = "infra"
_TOOL_KILLED = "[interrupted by user"      # what chad's bash tool returns when stopped


def _obj(value: JsonValue) -> Mapping[str, JsonValue]:
    return value if is_object(value) else {}


def _items(value: JsonValue) -> Sequence[JsonValue]:
    return value if is_array(value) else ()


def _num(value: JsonValue) -> float:
    return float(value) if is_number(value) else 0.0


def _text(value: JsonValue) -> str:
    return value if is_text(value) else ""


def agent_steps(doc: JsonValue) -> list[Mapping[str, JsonValue]]:
    return [_obj(s) for s in _items(_obj(doc).get("steps")) if _obj(s).get("source") == "agent"]


def is_chad(doc: JsonValue) -> bool:
    return _obj(_obj(doc).get("agent")).get("name") == "chad"


def kill_kind(last: Mapping[str, JsonValue]) -> str:
    """Where chad's last recorded step was when the wall ended the trial."""
    extra = _obj(last.get("extra"))
    calls = _items(last.get("tool_calls"))
    acting = bool(calls) or bool(_text(last.get("message")).strip())
    if extra.get("in_flight") is True:
        return "wall-act" if acting else "wall-think"
    finish = _obj(_obj(last.get("metrics")).get("extra")).get("finish")
    if finish == "interrupted":
        return "wall-act" if acting else "wall-think"
    results = _items(_obj(last.get("observation")).get("results"))
    if calls and any(_text(_obj(r).get("content")).startswith(_TOOL_KILLED) for r in results):
        return "wall-tool"
    return "wall-prefill"


def where(steps: Sequence[Mapping[str, JsonValue]], wall_s: float, kind: str) -> dict[str, float]:
    decode = prefill = 0.0
    in_flight = False
    for step in steps:
        extra = _obj(_obj(step.get("metrics")).get("extra"))
        decode += _num(extra.get("gen_s"))
        prefill += _num(extra.get("prefill_s"))
        in_flight = _obj(step.get("extra")).get("in_flight") is True
    rest = max(0.0, wall_s - decode - prefill)
    if in_flight and kind in ("wall-think", "wall-act"):
        decode, rest = decode + rest, 0.0
    return {"decode_s": round(decode, 1), "prefill_s": round(prefill, 1), "tool_s": round(rest, 1)}


def classify(doc: JsonValue, capped: bool, wall_s: float,
             agent_stop: str | None = None) -> dict[str, JsonValue]:
    """The `stop` object for one trial. `agent_stop` is the arm's own word for how the
    trial ended when it was not the wall: chad's `Agent.stop_kind`, `infra` for an engine
    failure, `error` for a CLI arm's non-zero exit; None means it ended as intended."""
    steps = agent_steps(doc)
    last = steps[-1] if steps else None
    step = int(_num(last.get("step_id"))) if last else None
    if agent_stop == INFRA:
        kind = INFRA
    elif not capped:
        kind = agent_stop.replace("_", "-") if agent_stop else "done"
    elif not is_chad(doc):
        kind = "unknown"
    elif last is None:
        kind = "wall-prefill"
    else:
        kind = kill_kind(last)
    split = (where(steps, wall_s, kind) if is_chad(doc)
             else {"decode_s": 0.0, "prefill_s": 0.0, "tool_s": round(wall_s, 1)})
    return {"kind": kind, "step": step, "where": split}


def kind_of(row: Mapping[str, JsonValue]) -> str:
    """A row's stop kind, or "" for a row written before rows carried one."""
    kind = _obj(row.get("stop")).get("kind")
    return kind if is_text(kind) else ""
