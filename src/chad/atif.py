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
edit nudges, sub-agent returns, the loop-break paths), several behind their own `continue`.
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
"""

import json
import logging
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from . import config
from .toolcall_parse import parse_tool_calls, strip_think

log = logging.getLogger("chad")

SCHEMA_VERSION = "ATIF-v1.7"
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.S)


def _now() -> str:
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


def _metrics(stat: dict) -> dict:
    """chad's GenStats counts `prompt_tokens` as tokens *actually prefilled*, excluding the
    prefix served from cache. ATIF's `prompt_tokens` is documented as including cached
    tokens, so add them back — otherwise a warm KV cache reads as a shrinking prompt."""
    prompt = stat.get("prompt_tokens", 0) + stat.get("cached_tokens", 0)
    return {"prompt_tokens": prompt,
            "completion_tokens": stat.get("generated_tokens", 0),
            "cached_tokens": stat.get("cached_tokens", 0)}


def steps_from_messages(messages: list, model_name: Optional[str],
                        stats: list) -> list[dict]:
    """Convert one Agent's `messages` into ATIF steps (without global `step_id`s).

    A `role: "tool"` message is not a step — it is an *observation* attached to the
    assistant step that produced the call. Results are paired with tool calls positionally;
    surplus results (an edit nudge, a validation reject with no matching call) attach with
    `source_call_id: None`, which ATIF permits.
    """
    steps: list[dict] = []
    i, agent_seen = 0, 0
    while i < len(messages):
        m = messages[i]
        role, content = m.get("role"), m.get("content") or ""
        if role == "system":
            steps.append({"source": "system", "message": content, "timestamp": _now()})
            i += 1
        elif role == "user":
            steps.append({"source": "user", "message": content, "timestamp": _now()})
            i += 1
        elif role == "assistant":
            reasoning, visible = split_think(content)
            calls = parse_tool_calls(content)
            j = i + 1
            results = []
            while j < len(messages) and messages[j].get("role") == "tool":
                results.append(messages[j])
                j += 1

            step: dict[str, Any] = {"source": "agent", "message": visible,
                                    "timestamp": _now()}
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
                                "arguments": args if isinstance(args, dict) else {}})
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
        self._lock = threading.Lock()

    def new_segment(self) -> int:
        with self._lock:
            self._segments.append([])
            return len(self._segments) - 1

    def set_segment(self, idx: int, steps: list[dict]) -> None:
        """Replace a segment's steps, KEEPING each existing step's first-seen timestamp.

        `steps_from_messages` rebuilds the whole segment after every step and stamps
        `_now()` on every step it emits — so without this merge, each rewrite dragged
        every prior step's timestamp forward to the dump time, and a submitted
        trajectory showed ~all steps at the moment of the final flush (which reads as
        synthetic under a leaderboard integrity review). The transcript is append-only
        within a turn (compaction rewrites content in place, never reorders), so
        matching by position is stable: step i keeps the timestamp it was first dumped
        with, and only genuinely new steps get fresh ones."""
        with self._lock:
            if 0 <= idx < len(self._segments):
                prev = self._segments[idx]
                for i, s in enumerate(steps[:len(prev)]):
                    if "timestamp" in prev[i]:
                        s["timestamp"] = prev[i]["timestamp"]
                self._segments[idx] = steps

    def to_dict(self) -> dict:
        with self._lock:
            steps = [s for seg in self._segments for s in seg]
        for n, s in enumerate(steps, 1):     # ATIF: sequential from 1, document-wide
            s["step_id"] = n
        agent: dict[str, Any] = {"name": self.agent_name, "version": self.agent_version}
        if self.model_name:
            agent["model_name"] = self.model_name
        if self.extra:
            agent["extra"] = self.extra
        doc: dict[str, Any] = {"schema_version": SCHEMA_VERSION,
                               "session_id": self.session_id,
                               "agent": agent, "steps": steps}
        mets = [s["metrics"] for s in steps if "metrics" in s]
        if mets:
            doc["final_metrics"] = {
                "total_prompt_tokens": sum(m["prompt_tokens"] for m in mets),
                "total_completion_tokens": sum(m["completion_tokens"] for m in mets),
                "total_cached_tokens": sum(m["cached_tokens"] for m in mets),
                "total_steps": len(steps)}
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


def _reset_for_tests() -> None:
    global _RECORDER, _INIT
    _RECORDER, _INIT = None, False
