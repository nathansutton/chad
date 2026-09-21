"""ATIF (Agent Trajectory Interchange Format) v1.7 trajectory emitter.

The Terminal-Bench leaderboard requires an ATIF trajectory for **every passing trial**
(the Apr-2026 "Leaderboard Integrity Update"). Harbor's built-in agents get this for free
by subclassing its agent base; chad is driven as a CLI, so it has to emit the document
itself or a submission is rejected.

Set `CHAD_TRAJECTORY_JSON=/path/trajectory.json` to turn this on. It is a pure observer:
nothing here changes agent behavior, and any failure is swallowed (a benchmark harness
must never lose a trial because telemetry broke).

**Why derive from `agent.messages` rather than instrument the tool dispatch.**
`run_turn` appends `{"role": "tool", ...}` from ~9 different sites (validation rejects,
edit nudges, done-gate rejections, the loop-break paths), several behind their own `continue`.
Hooking each one would rot the moment a tenth appears. The message list is the single
place every one of them converges, so we rebuild the trajectory from it after each step.

**Segments.** A trial can span several `Agent` objects: the governor relaunches a fresh turn
(new context, reset KV) after a budget stop, and each relaunch is a new `Agent` with an
empty `messages`. Each Agent claims a segment; `dump()` concatenates them and renumbers
`step_id` globally, because ATIF requires step ids sequential from 1 across the document.

Schema (harbor/models/trajectories/): `Trajectory{schema_version, session_id, agent,
steps[], final_metrics}`; `Step{step_id, timestamp, source: system|user|agent, message,
reasoning_content, tool_calls[], observation, metrics, model_name}`. Every model sets
`extra: "forbid"`, and a validator requires each `observation.results[].source_call_id`
to name a `tool_call_id` **in the same step** — hence one step carries both a turn's tool
calls and their results.

**What a step cost.** The schema's three token counts cannot tell a step that reasoned for
twenty minutes from one that decoded a long file, so each agent step's `metrics.extra`
carries the rest of what the engine measured (`StepStat`): the reasoning slice, prefill and
decode seconds, speculative yield, and why the generation ended.

**The step in flight.** A rebuild from `messages` only ever shows finished steps, and the
step worth seeing is the one that never finishes — a runaway `<think>` a harness kills at
its timeout leaves no message behind. So the agent hands over the partial text every
`PENDING_EVERY` tokens and the document ends in one step marked `extra.in_flight`, replaced
by the real step once its message lands.
"""

import json
import logging
import os
import re
import tempfile
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from . import config
from .toolcall_parse import parse_tool_calls, strip_think
from .tools import JsonValue, is_json_object

log = logging.getLogger("chad")

SCHEMA_VERSION = "ATIF-v1.7"
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.S)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_think(text: str) -> tuple[str, str]:
    """(reasoning, visible). Mirrors `strip_think`'s two cases: an explicit
    `<think>…</think>` block, and the template-opened case where generation starts *inside*
    the block so the text carries a leading `</think>` with no opening tag."""
    close = text.find("</think>")
    open_ = text.find("<think>")
    if close != -1 and (open_ == -1 or close < open_):
        return text[:close], text[close + len("</think>"):]
    reasoning = "\n".join(m.group(1) for m in _THINK_BLOCK.finditer(text))
    return reasoning, strip_think(text)


PENDING_EVERY = 256     # tokens between rewrites of the in-flight step (~10 s of decode)


@dataclass(frozen=True)
class StepStat:
    """What one generation cost, as the engine measured it. `finish` names why it ended:
    `eos`, `max_tokens`, `think_cap`, `repeat_stop` or `interrupted`."""
    prompt_tokens: int = 0          # newly prefilled — excludes the cached prefix
    cached_tokens: int = 0
    generated_tokens: int = 0
    think_tokens: int = 0
    prefill_s: float = 0.0
    gen_s: float = 0.0
    forwards: int = 0
    draft_proposed: int = 0
    draft_accepted: int = 0
    finish: str = "eos"
    salvaged: bool = False
    compacted: bool = False         # compaction shrank the render, so this step's prefill
                                    # is a re-prefill of kept context, not new content
    # Drafted rounds by proposed width k: entry i counts the rounds that accepted exactly
    # i of the k. Split by phase, because reasoning and a tool call accept differently.
    draft_hist_think: Mapping[int, Sequence[int]] = field(default_factory=dict)
    draft_hist_act: Mapping[int, Sequence[int]] = field(default_factory=dict)


def _draft_phase(hist: Mapping[int, Sequence[int]]) -> dict[str, JsonValue]:
    """One phase's drafting: totals, the per-verify yield, and the histogram itself.
    JSON keys are strings, so the width is spelled out."""
    rounds = sum(sum(row) for row in hist.values())
    proposed = sum(k * sum(row) for k, row in hist.items())
    accepted = sum(i * n for row in hist.values() for i, n in enumerate(row))
    return {"rounds": rounds, "proposed": proposed, "accepted": accepted,
            "accepted_per_round": round(accepted / rounds, 2) if rounds else 0.0,
            "first_token_miss": round(sum(row[0] for row in hist.values()) / rounds, 3)
            if rounds else 0.0,
            "by_width": {str(k): list(row) for k, row in sorted(hist.items())}}


def _metrics(stat: StepStat) -> dict[str, JsonValue]:
    """chad's GenStats counts `prompt_tokens` as tokens *actually prefilled*, excluding the
    prefix served from cache. ATIF's `prompt_tokens` is documented as including cached
    tokens, so add them back — otherwise a warm KV cache reads as a shrinking prompt."""
    extra: dict[str, JsonValue] = {
        "think_tokens": stat.think_tokens, "prefill_s": round(stat.prefill_s, 2),
        "gen_s": round(stat.gen_s, 2),
        "tok_per_s": round(stat.generated_tokens / stat.gen_s, 1) if stat.gen_s > 0 else 0.0,
        "forwards": stat.forwards,
        "draft_proposed": stat.draft_proposed, "draft_accepted": stat.draft_accepted,
        "finish": stat.finish}
    if stat.salvaged:
        extra["salvaged"] = True
    if stat.compacted:
        extra["compacted"] = True
    if stat.draft_hist_think or stat.draft_hist_act:
        extra["draft"] = {"think": _draft_phase(stat.draft_hist_think),
                          "act": _draft_phase(stat.draft_hist_act)}
    return {"prompt_tokens": stat.prompt_tokens + stat.cached_tokens,
            "completion_tokens": stat.generated_tokens,
            "cached_tokens": stat.cached_tokens, "extra": extra}


# First-seen timestamps, keyed by the identity of the message each step describes.
# The segment is rebuilt from scratch after every step, so a step's timestamp has to be
# recovered from somewhere or the whole trajectory reads as written at the final flush
# (which looks synthetic under a leaderboard integrity review). Recovering it by POSITION
# was only sound while the transcript was append-only, and it is not: compaction deletes
# messages out of the middle, after which every surviving step inherited some earlier
# message's stamp. So the stamp follows the message object itself.
#
# The entry keeps a strong reference to that dict, which is what makes `id()` safe as a
# key — the address cannot be recycled for a different message while the entry lives.
# Entries for messages that have left the transcript are dropped on the next rebuild, so
# the table tracks the live transcript and nothing more. Observing, never mutating: the
# message dicts are the agent's own and are rendered, saved and resumed.
_STAMPS: dict[int, tuple[dict, str]] = {}


def _stamp(m: dict, clock: Callable[[], str] = now) -> str:
    """The time `m` was first seen in a rebuild, minted on first sight."""
    hit = _STAMPS.get(id(m))
    if hit is not None and hit[0] is m:
        return hit[1]
    ts = clock()
    _STAMPS[id(m)] = (m, ts)
    return ts


def pending_step(text: str, thinking: bool, model_name: Optional[str], generated: int,
                 started: str) -> dict[str, JsonValue]:
    """The step still being generated, from the text decoded so far.

    On a thinking step the template opened `<think>`, so text with no close tag yet is all
    reasoning — `split_think` alone would file it as the visible message."""
    if thinking and "</think>" not in text:
        reasoning, visible = text, ""
    else:
        reasoning, visible = split_think(text)
    step: dict[str, JsonValue] = {"source": "agent", "message": visible, "timestamp": started,
                                  "extra": {"in_flight": True, "generated_tokens": generated}}
    if model_name:
        step["model_name"] = model_name
    if reasoning.strip():
        step["reasoning_content"] = reasoning
    return step


def steps_from_messages(messages: list, model_name: Optional[str],
                        stats: list[StepStat], clock: Callable[[], str] = now) -> list[dict]:
    """Convert one Agent's `messages` into ATIF steps (without global `step_id`s).

    A `role: "tool"` message is not a step — it is an *observation* attached to the
    assistant step that produced the call. Results are paired with tool calls positionally;
    surplus results (an edit nudge, a validation reject with no matching call) attach with
    `source_call_id: None`, which ATIF permits.

    Each step carries the timestamp its message was FIRST seen with (see `_STAMPS`), so
    rebuilding after a step re-stamps only what is genuinely new.
    """
    live = {id(m) for m in messages}
    for stale in [k for k in _STAMPS if k not in live]:
        del _STAMPS[stale]
    steps: list[dict] = []
    i, agent_seen = 0, 0
    while i < len(messages):
        m = messages[i]
        role, content = m.get("role"), m.get("content") or ""
        if role == "system":
            steps.append({"source": "system", "message": content,
                          "timestamp": _stamp(m, clock)})
            i += 1
        elif role == "user":
            steps.append({"source": "user", "message": content,
                          "timestamp": _stamp(m, clock)})
            i += 1
        elif role == "assistant":
            reasoning, visible = split_think(content)
            calls = parse_tool_calls(content)
            j = i + 1
            results = []
            while j < len(messages) and messages[j].get("role") == "tool":
                results.append(messages[j])
                j += 1

            step: dict[str, JsonValue] = {"source": "agent", "message": visible,
                                          "timestamp": _stamp(m, clock)}
            if model_name:
                step["model_name"] = model_name
            if reasoning.strip():
                step["reasoning_content"] = reasoning
            call_ids = []
            if calls:
                tcs = []
                for k, (name, args) in enumerate(calls):
                    cid = f"call_{len(steps) + 1}_{k}"
                    call_ids.append(cid)
                    tcs.append({"tool_call_id": cid, "function_name": name,
                                "arguments": args if is_json_object(args) else {}})
                step["tool_calls"] = tcs
            if results:
                step["observation"] = {"results": [
                    {"source_call_id": call_ids[k] if k < len(call_ids) else None,
                     "content": r.get("content") or ""}
                    for k, r in enumerate(results)]}
            if agent_seen < len(stats):
                step["metrics"] = _metrics(stats[agent_seen])
            agent_seen += 1
            steps.append(step)
            i = j
        else:                       # a stray `tool` with no assistant before it
            i += 1
    return steps


def _totals(mets: list[dict]) -> dict[str, JsonValue]:
    """The trial in one row: what a score table needs without re-reading every step."""
    prompt = sum(m["prompt_tokens"] for m in mets)
    gen_s = sum(m["extra"]["gen_s"] for m in mets)
    out: dict[str, JsonValue] = {
        "total_think_tokens": sum(m["extra"]["think_tokens"] for m in mets),
        "total_gen_s": round(gen_s, 2),
        "total_prefill_s": round(sum(m["extra"]["prefill_s"] for m in mets), 2),
        "tok_per_s": round(sum(m["completion_tokens"] for m in mets) / gen_s, 1)
        if gen_s > 0 else 0.0,
        "cache_hit_rate": round(sum(m["cached_tokens"] for m in mets) / prompt, 3)
        if prompt else 0.0,
        "compactions": sum(1 for m in mets if m["extra"].get("compacted")),
        "finishes": {f: sum(1 for m in mets if m["extra"]["finish"] == f)
                     for f in sorted({m["extra"]["finish"] for m in mets})}}
    for phase in ("think", "act"):
        blocks = [m["extra"]["draft"][phase] for m in mets if "draft" in m["extra"]]
        rounds = sum(b["rounds"] for b in blocks)
        if rounds:
            proposed = sum(b["proposed"] for b in blocks)
            accepted = sum(b["accepted"] for b in blocks)
            out[f"draft_{phase}"] = {
                "rounds": rounds, "accept_rate": round(accepted / proposed, 3),
                "accepted_per_round": round(accepted / rounds, 2)}
    return out


def _pkg_version() -> str:
    """The installed chad-code version, for the trajectory's agent.version field.
    Was a hardcoded literal, which silently rots on every release — a leaderboard
    submission would then claim the wrong agent version. Falls back for a source
    tree that was never pip/uv-installed."""
    try:
        from importlib.metadata import version
        return version("chad-code")
    except Exception:  # noqa: BLE001 — uninstalled source tree
        return "0.0.0"


class TrajectoryRecorder:
    """Accumulates per-Agent segments and writes the whole ATIF document on every dump."""

    def __init__(self, path: str, agent_name: str = "chad",
                 agent_version: Optional[str] = None, extra: Optional[dict] = None):
        self.path = path
        self.agent_name = agent_name
        self.agent_version = agent_version or _pkg_version()
        self.extra = extra
        self.session_id = str(uuid.uuid4())
        self.model_name: Optional[str] = None
        self._segments: list[list[dict]] = []
        self._pending: dict[int, dict[str, JsonValue]] = {}
        self._lock = threading.Lock()

    def new_segment(self) -> int:
        with self._lock:
            self._segments.append([])
            return len(self._segments) - 1

    def set_segment(self, idx: int, steps: list[dict]) -> None:
        """Replace a segment's steps.

        The steps arrive already stamped with the time each one's message was first
        seen (`steps_from_messages`), so the rebuild after every step carries the old
        timestamps forward on its own and this is a plain replacement."""
        with self._lock:
            if 0 <= idx < len(self._segments):
                self._segments[idx] = steps

    def set_pending(self, idx: int, step: Optional[dict[str, JsonValue]]) -> None:
        """Set or clear (None) the in-flight step that trails segment `idx`."""
        with self._lock:
            if step is None:
                self._pending.pop(idx, None)
            else:
                self._pending[idx] = step

    def to_dict(self) -> dict:
        with self._lock:
            steps = []
            for idx, seg in enumerate(self._segments):
                steps.extend(seg)
                if idx in self._pending:
                    steps.append(dict(self._pending[idx]))
        for n, s in enumerate(steps, 1):     # ATIF: sequential from 1, document-wide
            s["step_id"] = n
        agent: dict[str, JsonValue] = {"name": self.agent_name, "version": self.agent_version}
        if self.model_name:
            agent["model_name"] = self.model_name
        if self.extra:
            agent["extra"] = self.extra
        doc: dict[str, JsonValue] = {"schema_version": SCHEMA_VERSION,
                                     "session_id": self.session_id,
                                     "agent": agent, "steps": steps}
        mets = [s["metrics"] for s in steps if "metrics" in s]
        if mets:
            doc["final_metrics"] = {
                "total_prompt_tokens": sum(m["prompt_tokens"] for m in mets),
                "total_completion_tokens": sum(m["completion_tokens"] for m in mets),
                "total_cached_tokens": sum(m["cached_tokens"] for m in mets),
                "total_steps": len(steps),
                "extra": _totals(mets)}
        return doc

    def dump(self) -> None:
        """Atomically rewrite the trajectory. Called after every step, not just at exit:
        a benchmark harness kills the agent at its timeout, and a half-written or missing
        trajectory would make an otherwise-passing trial unsubmittable."""
        doc = self.to_dict()
        if not doc["steps"]:
            return                     # ATIF requires steps: min_length=1
        try:
            d = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(doc, f)
            os.replace(tmp, self.path)
        except Exception as e:         # telemetry must never break a run
            log.warning("atif: dump to %s failed: %s", self.path, e)


_RECORDER: Optional[TrajectoryRecorder] = None
_INIT = False


def recorder() -> Optional[TrajectoryRecorder]:
    """The process-wide recorder, or None when `CHAD_TRAJECTORY_JSON` is unset. One per
    process so governor relaunches (each a fresh Agent) land in one trajectory document."""
    global _RECORDER, _INIT
    if not _INIT:
        _INIT = True
        path = config.env_str("CHAD_TRAJECTORY_JSON")
        if path:
            _RECORDER = TrajectoryRecorder(path)
    return _RECORDER


def start(path: str) -> TrajectoryRecorder:
    """Install a fresh process recorder writing to `path`, for a driver that runs many
    trials in one process and wants a document per trial. Agents built afterwards record
    into it; one built earlier keeps the recorder it was constructed with."""
    global _RECORDER, _INIT
    _RECORDER, _INIT = TrajectoryRecorder(path), True
    return _RECORDER


def _reset_for_tests() -> None:
    global _RECORDER, _INIT
    _RECORDER, _INIT = None, False
    _STAMPS.clear()
