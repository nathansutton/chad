"""Agentic loop + REPL for chad.

Renders the conversation through the model's chat template (with tool schemas),
streams the assistant turn, parses Qwen/Hermes-style <tool_call> blocks, runs the
tools, feeds results back, and repeats until the model stops calling tools.

Prefix caching lives in Engine: re-rendering the whole transcript each step is
cheap because only the newly appended tokens get prefilled.
"""

import json
import os
import re
import sys
import time

from . import atif, compaction, config, guardrails, levers, session, syntaxgate
from .base_engine import BackendError, BaseEngine
from .diag import args_preview, log, redact, result_preview
from .prompt import build_subagent_prompt, build_system_prompt, classify_intent
from .render import (
    C_DIM,
    C_RED,
    C_RST,
    C_YEL,
    _default_emit,
    _disp_path,
    _StreamView,
    banner,
    confirm_preview,
    render_tool_result,
    render_tool_start,
)
from .toolcall_parse import parse_tool_calls, strip_think
from .tools import IGNORE_DIRS, TERMINAL, _under_plans, active_schemas, dispatch_for, is_mutating

# Sub-agent / Task tool (plan 041). A spawned sub-agent runs on the SAME engine (after a
# cache push) but a fresh transcript, with a tight step/context budget and — by default —
# a read-only toolset, so it can spelunk without mutating anything or bloating the main
# context. The read-only set is exploration + planning + the terminal tools; bash, write,
# edit and the symbol editors are excluded (they land only under tools="all").
# Spelunking is step-hungry (grep → read → grep → read) and its whole value is bringing
# back a condensed answer, so starve it and it returns nothing. These are ceilings, not
# targets: a sub-agent that finds its answer calls `done` on step 3. ctx_limit is further
# clamped to the parent's by min() at the spawn site.
SUBAGENT_MAX_STEPS = 24
SUBAGENT_CTX_LIMIT = 32000
SUBAGENT_READ_ONLY = {
    "read", "grep", "glob", "repo_map", "overview", "view_symbol",
    "find_symbol", "find_refs", "done", "finish", "stop",
}
# `write_todos` is deliberately absent: a sub-agent that plans its own work would mutate
# the process-global `_TODOS` and clobber the parent's pinned todo panel (and `_sub_emit`
# has no panel route for the "todos" kind anyway). A sub-agent's plan is not the parent's
# (plan 052).


def subagent_tools_for(parent_mode: str, requested: str) -> str:
    """The toolset a sub-agent may run with. A sub-agent auto-approves its own tool
    calls (mode='auto', no confirm callback), so it must never hold more autonomy
    than its parent: only an 'auto' parent (--yolo / headless) may delegate 'all';
    a 'normal' parent's human-approval promise and plan mode's read-only promise
    both clamp the sub-agent to read-only."""
    if parent_mode == "auto" and requested == "all":
        return "all"
    return "read-only"


from .validate import VALIDATE, coerce_and_validate, legacy_validate, render_repair

# Validation (VALIDATE knob, legacy_validate baseline) lives in validate.py, the
# single source of truth shared with toolcall_parse.py — imported above.

# Plan 035 (measurement spike): env-gated per-step prefill telemetry. Resolved ONCE at
# import (like diag._DISABLED) so the hot loop pays only a truthiness check when unset —
# zero string/dict/IO work, per plan 020's "instrumentation must not tax the product"
# rule. Set CHAD_PREFILL_TRACE=path/to/trace.jsonl to capture one JSON row per engine
# generate() call (== one prefill event) for scripts/prefill_tax.py to analyze offline.
_PREFILL_TRACE = config.env_str("CHAD_PREFILL_TRACE")


def _trace_prefill(row: dict) -> None:
    """Append one JSON line to the prefill trace and flush (sessions get killed mid-run,
    so we durably land every row). Best-effort: a trace IO error must never break a turn,
    so swallow it. Only ever called when _PREFILL_TRACE is set."""
    assert _PREFILL_TRACE is not None
    try:
        with open(_PREFILL_TRACE, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
    except OSError:
        pass

# @file mentions (Claude-Code parity): a message like "why is @geo.py slow?" pulls the
# named file into context directly, no `read` round-trip. `@` must be at start-of-string
# or after whitespace so an email (foo@bar.com) or decorator never matches; trailing
# sentence punctuation is trimmed off the path.
_MENTION_RE = re.compile(r"(?:^|\s)@([A-Za-z0-9_.~/-]+)")

# Chat-template special-token literals (<|im_end|>, <|mask_end|>, …) leaked into
# generated TEXT. They are never legitimate content; see the scrub site in run_turn.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[A-Za-z0-9_]+\|>")


def expand_mentions(text: str):
    """Expand each @path in `text` into attached context: a real FILE becomes a bounded
    snapshot (reusing the read tool's skeleton/char-cap policy, so a big file can't blow
    up prefill); a DIRECTORY becomes a short listing of its entries. Returns
    (augmented_text, [resolved_paths]); text unchanged and list empty when nothing
    resolves."""
    resolved = []  # (kind, path)
    seen = set()
    for m in _MENTION_RE.finditer(text):
        raw = m.group(1).rstrip(".,;:!?)")  # trailing punctuation isn't part of the path
        path = os.path.expanduser(raw)
        if path in seen:
            continue
        if os.path.isfile(path):
            resolved.append(("file", path)); seen.add(path)
        elif os.path.isdir(path):
            resolved.append(("dir", path)); seen.add(path)
    if not resolved:
        return text, []
    from .tools import tool_read
    blocks = []
    for kind, p in resolved:
        if kind == "file":
            blocks.append(f"@{p}:\n{tool_read(p)}")
        else:
            entries = sorted(e + ("/" if os.path.isdir(os.path.join(p, e)) else "")
                             for e in os.listdir(p) if e not in IGNORE_DIRS)
            listing = "\n".join(entries[:200]) or "(empty)"
            if len(entries) > 200:
                listing += f"\n… (+{len(entries) - 200} more)"
            blocks.append(f"@{p}/ (directory listing):\n{listing}")
    augmented = text + "\n\n[Attached for the @ reference(s) above:]\n" + "\n\n".join(blocks)
    return augmented, [p for _, p in resolved]


def _has_open_tool_call(text: str) -> bool:
    """True if `text` opened a tool-call block it never closed — the signature of a
    generation truncated mid-call (most often a `write` whose content blew the token
    cap). Used to give targeted "write it in parts" guidance instead of generic advice."""
    return (text.count("<tool_call>") > text.count("</tool_call>")
            or text.count("<function=") > text.count("</function>"))


def reject_escalation(name: str) -> str:
    """Extra guidance appended when the SAME tool call has been rejected identically
    back-to-back. The plain repair message clearly is not landing — the model can't see
    the fix (or the call genuinely can't succeed as written), so re-emitting it verbatim
    just burns turns. Break the loop: stop repeating, and — critically — do NOT fabricate
    the result the tool would have returned. A silently-failed `activate_skill` that the
    model papers over by reciting a skill from memory is the worst outcome: confident
    output that never loaded the real instructions."""
    extra = ("\n[you have now emitted this exact call twice and it was rejected both "
             "times — re-emitting it unchanged will not work. Either change the flagged "
             "field(s), or use a DIFFERENT tool. Do NOT invent or guess the output this "
             "tool would have returned.]")
    if name == "activate_skill":
        extra += ("\n[the skill was NOT loaded. Do not proceed from memory or fabricate "
                  "its steps. Check the exact skill `name` against the '# Skills' list, or "
                  "continue the task with the normal read/grep/bash tools instead.]")
    return extra


def close_unclosed_think(text: str, thinking: bool) -> str:
    """Close a dangling `<think>` block so the stored assistant turn re-tokenizes into
    a prefix of the live KV cache.

    On a thinking model the chat template auto-opens `<think>\\n` in the generation
    prompt, so the model's output *continues inside* the think block and a normal turn
    emits its own `</think>`. A turn truncated at the token cap can stop BEFORE that
    close. The template then renders the stored (unclosed) content as an EMPTY think
    block + trailing content — which diverges from the cache at the very first content
    token. On Ornith's non-trimmable cache that single divergence forces a FULL
    re-prefill of the whole transcript next step (measured: tens of thousands of tokens
    at large context). Appending the missing `</think>` keeps the cached tokens a strict
    prefix of the re-render, so only a couple of tokens prefill instead. No-op when
    thinking is off or the block is already closed."""
    if thinking and "<think>" not in text and "</think>" not in text and text:
        return text + "\n</think>"
    return text


# Permission modes (Claude Code parity, cycled with shift-tab in the TUI):
#   normal — confirm each mutating tool (bash/write/edit)
#   auto   — auto-approve mutating tools (yolo)
#   plan   — read-only: changes are blocked; the model researches and proposes a plan
MODES = ("normal", "auto", "plan")
MODE_LABEL = {"normal": "normal", "auto": "auto-accept edits", "plan": "plan mode"}
_PLAN_PREFIX = (
    "[PLAN MODE. Research first (read/grep/glob/repo_map/overview) — do NOT edit "
    "project files or run commands. Then write ONE self-contained plan to "
    "./plans/NNN-kebab-title.md (continue the existing number sequence) with the "
    "`write` tool. The plan must inline everything an executor needs WITHOUT this "
    "chat: a Context section (why), exact file paths with current-state code "
    "excerpts, numbered step-by-step changes, the verify commands to run, repo "
    "conventions, and explicit out-of-scope. After writing the file, call done. "
    "Writing the plan file is your only allowed change.]\n\n"
)
# Plan mode on a QUESTION (no change requested) should answer it, not manufacture a
# plan file — a "give me a 3-sentence tour" must not become an 84-line plans/ doc.
# Still fully read-only: research, then reply in prose.
_PLAN_PREFIX_CONCEPTUAL = (
    "[PLAN MODE. This is a question, not a change request — do NOT edit project files, "
    "run commands, or write a plan file. Research as needed "
    "(read/grep/glob/repo_map/overview), then ANSWER the question directly in prose and "
    "call done.]\n\n"
)


def _plan_prefix(intent: dict) -> str:
    """The plan-mode preamble for a turn, chosen by intent: a real change request gets
    the plan-FILE mandate; a bare question gets answered in prose (no plans/ doc — the
    "84-line plan for a 3-sentence tour" regression). Split out to be unit-testable."""
    return _PLAN_PREFIX if intent["action"] else _PLAN_PREFIX_CONCEPTUAL


# Backstop cap on any single tool result appended to the transcript. Each big char is
# prefill on a bandwidth-bound local model, and dogfooding showed a single tool call can
# stall the next turn for tens of seconds: a whole-file read (fixed by READ_MAX_CHARS), a
# wide grep (GREP_MAX_CHARS), and — the case those per-tool caps miss — the symbolic map
# tools (overview/repo_map/view_symbol), whose output scales with a file's symbol count
# (a 9,864-token `overview` = a 32s stall). This bounds EVERY tool uniformly so no one
# call blows up prefill; it sits just above the per-tool caps so it only bites the
# otherwise-uncapped tools. ~14k chars ≈ 4k tokens ≈ a ~14s worst-case prefill on the 9B.
# Transient backend faults (llama.cpp 5xx / mid-stream error chunks) are re-rolled rather
# than allowed to kill an unattended turn. Bounded so a server that is genuinely down
# surfaces as an error instead of burning the task's whole budget on retries.
_MAX_BACKEND_RETRIES = 3

_MAX_TOOL_RESULT_CHARS = 14000
# Per-STEP budget on the total tool-output chars appended before the next prefill. The
# per-result cap above bounds each call, but a step that emits SEVERAL calls stacks
# them into one prefill (two 10k greps = ~5.7k tokens = a ~20s stall the per-result cap
# never sees). Budget == the single-call cap, so a step is never worse than its worst
# single call: the first result can use the whole budget, later ones get what's left,
# and every result keeps at least a FLOOR-sized head (an edit/bash outcome must never
# be swallowed whole). Worst case ≈ 14k + 1k per extra call.
_STEP_TOOL_BUDGET_CHARS = _MAX_TOOL_RESULT_CHARS
_STEP_TOOL_FLOOR_CHARS = 1000


def _step_tool_cap(spent: int) -> int:
    """Char cap for the next tool result of a step whose earlier results already
    appended `spent` chars. Pure and testable."""
    return max(_STEP_TOOL_FLOOR_CHARS,
               min(_MAX_TOOL_RESULT_CHARS, _STEP_TOOL_BUDGET_CHARS - spent))


def _clip_tool_result(result: str, cap: int = _MAX_TOOL_RESULT_CHARS) -> str:
    """Bound a tool result's size so no single call blows up the next turn's prefill.
    Keeps the head (usually the most relevant) and notes how to fetch the rest."""
    if not isinstance(result, str) or len(result) <= cap:
        return result
    omitted = len(result) - cap
    return (result[:cap]
            + f"\n[… {omitted} chars truncated to keep the turn responsive — narrow the "
            f"query (grep a pattern, view_symbol(name), or read(path, offset=N)), or "
            f"re-run this call by itself, to pull just what you need.]")

# Canned task behind the /init slash command (Claude-Code parity): scaffold a CLAUDE.md
# the way `claude /init` does. Runs through the normal agentic loop (repo_map → read the
# config files → write), so it benefits from every reliability fix above.
INIT_PROMPT = (
    "Create a CLAUDE.md file at the root of this project to help an AI coding assistant "
    "work here effectively. First orient yourself: call repo_map, then read the key "
    "config/entry files that exist (README, pyproject.toml/package.json/go.mod/Cargo.toml, "
    "Makefile). Then write CLAUDE.md with the `write` tool containing, concisely (aim for "
    "under 60 lines): a one-paragraph overview of what the project does; the main "
    "components and how they fit together; the ACTUAL commands to build, run, and test it "
    "(copy them from the config you read, don't invent); and any conventions worth noting. "
    "If a CLAUDE.md already exists, read it first and improve it rather than clobbering it. "
    "After writing, call done."
)


class Agent:
    def __init__(self, engine: BaseEngine, yolo: bool = False, max_steps: int = 40,
                 ctx_limit: int = 24000, mode: str = None, emit=None,
                 confirm=None, should_stop=None, drain_steering=None,
                 thinking: bool = True,
                 max_gen_tokens: int = 8192, resume: list = None, persist: bool = False,
                 think_budget: int = None, turn_budget_tokens: int = None,
                 turn_budget_s: float = None, subagent: bool = False,
                 subagent_tools: str = "read-only", session_id: str = None,
                 ctx_limit_fn=None):
        self.engine = engine
        # A spawned sub-agent (plan 041) SHARES the parent's session — the same engine,
        # the same live skills/MCP connections — so it must NOT tear those down. Only a
        # top-level Agent resets them (a fresh session clears stale activation state and
        # reaps prior MCP processes; matches engine._reset_cache on /reset).
        self._subagent = subagent
        # Which tools a sub-agent may see: "read-only" (default) or "all". Ignored for a
        # top-level agent (it always gets the full toolset minus the CHAD_NO_* gates).
        self._subagent_tools = subagent_tools
        if not subagent:
            from . import skills
            skills.reset_session()
            from . import mcp
            mcp.reset_session()
        self.mode = mode or ("auto" if yolo else "normal")
        self.thinking = thinking  # Ornith is a reasoning model; toggles <think> blocks
        # Steps per WINDOW, not a hard kill: a window that landed+verified a change
        # earns an extension (see guardrails.extend_step_cap; absolute ceiling 4x).
        self.max_steps = max_steps
        # Session persistence (cli.py --continue): when `persist`, the conversation is
        # saved to disk (keyed by cwd) after each turn; `resume` seeds it from a prior
        # save. A fresh system prompt is always rebuilt (cwd/workspace may have changed);
        # only the non-system turns are restored.
        self.persist = persist
        # This turn-thread's session id (plan 043), minted here so every resume forks: a
        # resumed conversation is loaded into a FRESH Agent, which mints a NEW id and
        # saves to a NEW file — the original session file is never rewritten. `save()`
        # writes only to this id's file.
        self.session_id = session_id or session.new_session_id()
        # Per-step generation cap. The old 2048 default truncated legitimate work —
        # a reasoning turn that thinks, then emits a `write` of a whole test file can
        # exceed 2048 tokens, and the cut-off was being misread as a final answer
        # (the "answers on paper then stops" bug). 8192 leaves room for think + a
        # full-file write; truncation past it is now detected and nudged, not accepted.
        self.max_gen_tokens = max_gen_tokens
        # Soft think-cap base (plan 039). None => the mechanism is OFF and generation is
        # byte-identical to before. Falls back to the CHAD_THINK_BUDGET env knob so the
        # eval harness can arm an arm without a code change (matches the CHAD_* family).
        # When set, run_turn stops each step's <think> run once it exceeds this many
        # tokens and force-closes the block (prefix-safe); the cap escalates with the
        # turn's stuck-signals (see guardrails.think_budget).
        if think_budget is None:
            think_budget = config.env_int("CHAD_THINK_BUDGET")
        self.think_budget = think_budget
        # Degenerate-repetition stop (see guardrails.degenerate_tail): default ON —
        # it only fires on output that is already garbage (a literal decode loop), so
        # unlike the think-budget there is no capability trade. A/B off via env.
        self._no_repeat_guard = config.flag("CHAD_NO_REPEAT_GUARD")
        self.ctx_limit = ctx_limit  # prompt-token budget before compaction kicks in
        # Cross-call compaction state (plan 075 WS1.5): holds the over-limit latch so
        # a transcript whose protected floor exceeds ctx_limit isn't re-compacted —
        # and its warm prefix cache destroyed — on every single step.
        self._compact_state: dict = {}
        # Live ctx-limit recheck (plan 075 WS1.4): the startup limit was computed on
        # an idle box; Docker/harbor spinning up mid-session changes what's safe.
        # Called at the top of each turn; only a >10% move is applied (hysteresis).
        self._ctx_limit_fn = ctx_limit_fn
        # Runaway-turn governor (plan 040): a per-turn budget on cumulative prefill tokens
        # (and optional wall-clock) that ends a turn which has burned through a checkpoint
        # WITHOUT landing+verifying a change — banking a deterministic progress note so the
        # caller can relaunch fresh (shedding both the ramble and the huge prefill a stuck
        # weak model drags around). Off entirely under CHAD_NO_GOVERNOR=1 (A/B family). The
        # token budget defaults to 3× the context limit — far above a normal task's use
        # (passing eval tasks: 14–35k prefill tokens; the pathological timeout tail:
        # 130–187k). Wall budget is off unless set: interactively the human is the wall
        # clock; evals/one-shot set it via --turn-budget-s / CHAD_TURN_BUDGET_S.
        self._no_governor = config.flag("CHAD_NO_GOVERNOR")
        if turn_budget_tokens is None:
            turn_budget_tokens = config.env_int("CHAD_TURN_BUDGET_TOKENS", max(0, 3 * self.ctx_limit))
        self._turn_budget_tokens = turn_budget_tokens
        if turn_budget_s is None:
            turn_budget_s = config.env_float("CHAD_TURN_BUDGET_S")
        self._turn_budget_s = turn_budget_s
        # Set when a turn hard-stops on budget (like last_plan_path): holds the progress
        # note so the caller (TUI / one-shot / evals) can relaunch a fresh turn seeded
        # with it. Reset at the start of every run_turn.
        self.budget_note: str | None = None
        # The profile is keyed off the served model id, so an `--backend openai` run
        # against a non-Ornith endpoint drops the Ornith accommodations automatically
        # instead of silently carrying them into a cross-model comparison.
        _mid = getattr(engine, "model_id", None)
        self.messages = [{"role": "system",
                          "content": build_subagent_prompt(_mid) if subagent
                          else build_system_prompt(_mid)}]
        if resume:
            self.messages += [m for m in resume if m.get("role") != "system"]
        self._emit = emit or _default_emit
        self._confirm_cb = confirm  # callable(name, args)->bool; None => input() prompt
        self._should_stop = should_stop or (lambda: False)
        # Mid-run steering (improve 01): callable() -> list[str] of user redirections
        # typed while the turn runs, drained between steps and injected into the live
        # transcript (a pure append — the warm KV prefix stays valid) instead of
        # forcing interrupt + re-prefill. None (headless/bench/sub-agent) = off.
        self._drain_steering = drain_steering
        self._backend_retries = 0   # reset per turn; see _MAX_BACKEND_RETRIES
        # ATIF trajectory capture (plan 068), off unless CHAD_TRAJECTORY_JSON is set. A
        # sub-agent runs on the parent's engine inside one of the parent's steps; recording
        # it as a sibling segment would interleave two transcripts into one step sequence.
        self._atif = None if subagent else atif.recorder()
        self._atif_seg = self._atif.new_segment() if self._atif else None
        self._atif_stats: list = []   # one entry per successful generate, in step order
        if self._atif and self._atif.model_name is None:
            self._atif.model_name = getattr(engine, "model_id", None)
        self.interrupted = False
        # Absolute path of the plan file written during a plan-mode turn (consumed by
        # the TUI to offer the steer/accept handoff); reset each time it's read.
        self.last_plan_path = None
        # rolling throughput accounting (read by evals / status line)
        self.gen_tokens = 0
        self.gen_time = 0.0
        self.forwards = 0
        self.draft_proposed = 0
        self.draft_accepted = 0
        self.think_tokens = 0   # tokens spent inside <think> blocks (reasoning overhead)
        self.think_capped = 0   # times the soft think-cap force-closed a step (plan 039)
        # prefill accounting: the master cost for a local model is how many *new*
        # tokens it has to prefill across a turn (context bloat -> big prefills).
        # This is the metric symbolic/repo-map retrieval is meant to shrink.
        self.prefill_tokens = 0   # sum of newly-prefilled (uncached) tokens, all steps
        self.peak_ctx = 0         # largest prompt the turn ever rendered (tokens)
        # Plan 035: total cache length (cached+prefilled+generated) after the prior
        # traced step, so the trace can name this step's sync_kind by comparing this
        # step's `cached_tokens` against it. Persists across run_turn calls (file order
        # is chronological). Only updated when CHAD_PREFILL_TRACE is set.
        self._prefill_trace_prev = 0
        self._prefill_trace_seq = 0
        # Tool executions since the last traced row: (name, wall_s) pairs. The tools a
        # step runs produce the *next* step's appended prompt, so they're recorded on
        # that next row (field `prev_tools`). Only populated when tracing is on.
        self._trace_tools_pending: list = []

    @property
    def yolo(self) -> bool:
        return self.mode == "auto"

    @property
    def tok_per_s(self) -> float:
        return self.gen_tokens / self.gen_time if self.gen_time else 0.0

    @property
    def accept_rate(self) -> float:
        return self.draft_accepted / self.draft_proposed if self.draft_proposed else 0.0

    def cycle_mode(self) -> str:
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        return self.mode

    # Compaction logic now lives in compaction.py (operates on a messages list +
    # render/emit callbacks). These aliases keep `self._COLLAPSED`/`self._headtail`
    # working for compact_now below, byte-identical to the old in-class versions.
    _COLLAPSED = compaction._COLLAPSED
    _headtail = staticmethod(compaction._headtail)

    def _compact_if_needed(self, prompt_ids):
        return compaction.compact_if_needed(
            self.messages, self._render, self._emit, self.ctx_limit, prompt_ids,
            state=self._compact_state)

    def save(self):
        """Persist the conversation for the current dir (no-op unless `persist`)."""
        if self.persist:
            session.save_session(os.getcwd(), self.messages,
                                 {"mode": self.mode, "thinking": self.thinking},
                                 session_id=self.session_id)

    def compact_now(self):
        """Manual context reclaim (the /compact command). Runs only the SAFE, lossless-
        ish passes — strip <think> reasoning from older assistant turns and head/tail-
        truncate older tool outputs — and never drops a message, so it can't break the
        conversation. The next turn re-syncs the prefix cache normally. Returns
        (before_tokens, after_tokens)."""
        before = len(self._render())
        for i in [i for i, m in enumerate(self.messages)
                  if m.get("role") == "assistant" and "</think>" in m["content"]][:-2]:
            c = self.messages[i]["content"]
            self.messages[i]["content"] = c.split("</think>", 1)[1].lstrip("\n")
        # A compaction notice is guidance, not output: head/tail-clipping one would leave
        # the model a mangled instruction, and it must not consume one of the four spared
        # slots either. Same exemption the skill messages get in compact_if_needed.
        idxs = [i for i, m in enumerate(self.messages)
                if m.get("role") == "tool" and self._COLLAPSED not in m["content"]
                and compaction._NOTICE_TAG not in m["content"]]
        for i in idxs[:max(0, len(idxs) - 4)]:  # keep the last 4 tool outputs verbatim
            if len(self.messages[i]["content"]) > 400:
                self.messages[i]["content"] = self._headtail(self.messages[i]["content"])
        return before, len(self._render())

    def _active_schemas(self):
        """The tool schemas to expose THIS agent, on top of the module-level gates
        (CHAD_NO_SYMBOLS / CHAD_NO_TASK / skills / MCP). A sub-agent (plan 041) never
        sees `task` — reentrancy guard, subagents can't spawn subagents — and, unless it
        was granted tools="all", is restricted to the read-only exploration set so it
        can't mutate the repo it's spelunking through."""
        schemas = active_schemas()
        if not self._subagent:
            return schemas
        allow = None if self._subagent_tools == "all" else SUBAGENT_READ_ONLY
        out = []
        for s in schemas:
            n = s["function"]["name"]
            if n == "task":
                continue
            if allow is not None and n not in allow:
                continue
            out.append(s)
        return out

    @staticmethod
    def _template_ids(rendered):
        """Coerce `apply_chat_template`'s return to a plain list of token ids. Some
        HF tokenizers return a dict-like `BatchEncoding` (not an int list); `list()`
        of that yields its string keys (`['input_ids', 'attention_mask']`), which the
        `--backend llama` path would then ship verbatim as the prompt (garbage). The
        MLX path returns a plain list already — untouched by this guard."""
        return rendered["input_ids"] if hasattr(rendered, "input_ids") else rendered

    def _render(self):
        ids = self._template_ids(self.engine.tok.apply_chat_template(
            self.messages, tools=self._active_schemas(), add_generation_prompt=True,
            enable_thinking=self.thinking,
        ))
        # Debug hook (env-gated, off by default): dump the first decoded render so a
        # rendered-prompt difference across environments can be diffed. Best-effort.
        dump = os.environ.get("CHAD_DUMP_RENDER")
        if dump and not getattr(self, "_dumped_render", False):
            try:
                with open(dump, "w") as f:
                    f.write(self.engine.tok.decode(list(ids)))
                self._dumped_render = True
            except Exception:  # noqa: BLE001 — a debug dump must never break a turn
                pass
        return ids

    def _stable_prefix_ids(self):
        """The byte-identical head of every session: system prompt + tool schemas,
        up to (but not including) the first user turn. Computed by diffing two
        first-turn renders with different user text — their common token prefix is
        exactly the system+tools block. This is what warm_prefix checkpoints to disk
        so a cold start skips re-prefilling it (the template can't render a system
        message alone — it raises 'No user query found' — hence the diff trick)."""
        sysm = self.messages[0]
        def render1(u):
            return self._template_ids(self.engine.tok.apply_chat_template(
                [sysm, {"role": "user", "content": u}], tools=self._active_schemas(),
                add_generation_prompt=True, enable_thinking=self.thinking))
        a, b = render1("a"), render1("the quick brown fox jumps")
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return list(a[:n])

    def _confirm(self, name, args) -> bool:
        # Destructive-bash seatbelt: a catastrophic shell command (rm -rf ~, mkfs,
        # curl|sh, …) is screened even in --yolo/auto mode, because the model acts on
        # untrusted repo contents and auto mode has no human in the loop. If a confirm
        # channel exists (TTY or callback) we force the prompt; headless with no channel
        # we BLOCK rather than execute on injection. CHAD_NO_DESTRUCTIVE_GUARD=1 opts out.
        dangerous = (name == "bash" and isinstance(args, dict)
                     and not config.flag("CHAD_NO_DESTRUCTIVE_GUARD")
                     and guardrails.is_destructive_bash(str(args.get("command", ""))))
        if self.mode == "auto" or not is_mutating(name):
            if not dangerous:
                return True
            if self._confirm_cb is None and not sys.stdin.isatty():
                self._emit("info", f"  [blocked destructive command in auto mode: "
                                   f"{args.get('command', '')!r}; set CHAD_NO_DESTRUCTIVE_GUARD=1 to allow]")
                return False
        if self._confirm_cb is not None:
            return self._confirm_cb(name, args)
        preview = confirm_preview(name, args)
        warn = f"{C_RED}  ⚠ looks destructive — review carefully\n{C_RST}" if dangerous else ""
        ans = input(f"{C_YEL}  allow {name}:\n{preview}\n{warn}  approve? [y/N] {C_RST}").strip().lower()
        return ans in ("y", "yes")

    def _sub_emit(self, kind: str, text: str):
        """Emit callback handed to a spawned sub-agent so its activity renders DIMMED and
        subordinate in the main transcript. Live status gauges (spinner verb, ctx/gen/
        prefill counters) pass through so the UI still reflects the sub-agent's work; its
        streamed prose and reasoning are suppressed (only its final return matters); its
        tool activity and notices are downgraded to the dim 'muted' channel."""
        if kind in ("status", "ctx", "gen", "prefill"):
            self._emit(kind, text)
        elif kind in ("stream", "think"):
            return  # sub-agent prose/reasoning isn't the deliverable — keep it out
        elif kind == "tool":
            self._emit("muted", "   ⌊ " + text)
        else:  # muted / info / add / del / error
            self._emit("muted", "     " + str(text).lstrip())

    def _run_subagent(self, description: str, prompt: str, tools: str = "read-only") -> str:
        """Run a scoped sub-agent (plan 041) on a QUARANTINED cache and return its final
        text as the `task` tool result. push_cache stashes the main session's warm cache
        aside; the sub-agent runs on a fresh one with a tight step/context budget and a
        (default) read-only toolset; pop_cache restores the main cache bit-identically —
        even on error/interrupt (the finally), so a stuck sub-agent never corrupts the
        parent. Its grep/read churn never enters the main transcript; only this return
        does. Depth 1 only: a sub-agent can't itself call `task` (its schema omits it)."""
        # Never grant a sub-agent more autonomy than its parent: the sub-agent runs
        # mode='auto' with no confirm callback, so an 'all' toolset is honored only when
        # the parent itself is auto-approved (--yolo/headless); 'normal' (human confirms
        # every mutation) and 'plan' (read-only) parents clamp it to read-only.
        requested, tools = tools, subagent_tools_for(self.mode, tools)
        if requested == "all" and tools == "read-only":
            self._emit("muted", "   ⌊ sub-agent clamped to read-only (parent mode is not auto)")
        self._emit("muted", f"   ⌊ delegating to sub-agent: {description}")
        log.info("TASK start | desc=%r | tools=%s | prompt=%r",
                 description, tools, redact(prompt[:300]))
        sub = None
        _t0 = time.perf_counter()
        self.engine.push_cache()
        try:
            sub = Agent(
                self.engine,
                mode="auto",                       # auto-approve within its restricted toolset
                max_steps=SUBAGENT_MAX_STEPS,
                ctx_limit=min(self.ctx_limit, SUBAGENT_CTX_LIMIT),
                thinking=self.thinking,
                max_gen_tokens=self.max_gen_tokens,
                emit=self._sub_emit,
                should_stop=self._should_stop,
                think_budget=self.think_budget,
                turn_budget_tokens=0,              # the sub-agent's own max_steps bounds it
                turn_budget_s=0.0,
                subagent=True,
                subagent_tools=tools,
            )
            result = sub.run_turn(prompt)
        except Exception as e:  # noqa: BLE001 — a sub-agent crash must not kill the parent turn
            result = f"[task failed: {type(e).__name__}: {e}]"
        finally:
            self.engine.pop_cache()
        # Roll the sub-agent's cost into the parent's rolling accounting so total turn
        # spend stays visible (its prefill also feeds the governor budget); peak_ctx is
        # left as the MAIN transcript's — the whole point is that it does NOT grow.
        if sub is not None:
            self.gen_tokens += sub.gen_tokens
            self.gen_time += sub.gen_time
            self.prefill_tokens += sub.prefill_tokens
            self.think_tokens += sub.think_tokens
            self.forwards += sub.forwards
            # Fail-safe: a sub-agent that ends early — step cap, crash, interrupt — or
            # that returns nothing at all must still hand back where it got to. Never
            # surface a bare sentinel: the parent then restarts the localization from
            # zero (the django-14007/sphinx-9230 failure), and the anti-respawn guard
            # above refuses the retry, so the turn dies with the findings still in the
            # dead sub-agent's transcript. progress_note is deterministic and model-free,
            # so it works even from a crashed turn: it re-reads the sub-agent's own tool
            # calls for the files it examined, the commands it ran, and the last
            # hypothesis it stated.
            def _salvage(res: str) -> str:
                if not levers.enabled("subagent_budget_note"):
                    return res   # pre-iter-2: the capped sub-agent's findings are discarded
                note = sub.budget_note or guardrails.progress_note(sub.messages)
                if not note:
                    return res
                return (res.rstrip() + "\n[sub-agent progress before it "
                        f"stopped: {note}]").strip()
            if sub.interrupted:
                result = _salvage("[task interrupted]")
            elif (not result or result.startswith("[stopped:")
                    or result.startswith("[task failed:")):
                result = _salvage(result or "[task returned nothing]")
        # tool_calls (not `forwards`, a speculative-decoding counter that is 0 on the
        # normal path) is the number that diagnoses a sub-agent returning nothing: it
        # separates "never got to search" from "searched and lost its findings".
        log.info("TASK end | desc=%r | %.1fs | tool_calls=%d gen=%d prefill=%d | -> %s",
                 description, time.perf_counter() - _t0,
                 sum(1 for m in sub.messages if m.get("role") == "tool") if sub else 0,
                 sub.gen_tokens if sub else 0,
                 sub.prefill_tokens if sub else 0, result_preview(result or ""))
        return result or "[task returned nothing]"

    def _atif_sync(self) -> None:
        """Rebuild this Agent's ATIF segment from `messages` and rewrite the document.
        Cheap (the transcript is small) and idempotent; a no-op unless capture is on."""
        if self._atif is None or self._atif_seg is None:
            return
        try:
            self._atif.set_segment(
                self._atif_seg,
                atif.steps_from_messages(self.messages, self._atif.model_name,
                                         self._atif_stats))
            self._atif.dump()
        except Exception as e:      # never let telemetry break a turn
            log.warning("atif: sync failed: %s", e)

    def run_turn(self, user_text: str, stream=True):
        """Thin wrapper so the trajectory is flushed on EVERY exit from the turn —
        normal return, interrupt, or an exception escaping the loop. Harbor SIGKILLs chad
        at the task timeout, so the per-step dump inside `_run_turn` is what actually saves
        a long trial; this `finally` covers the clean paths."""
        try:
            return self._run_turn(user_text, stream)
        finally:
            self._atif_sync()

    def _run_turn(self, user_text: str, stream=True):
        self.interrupted = False
        self._backend_retries = 0   # per-turn allowance; see _MAX_BACKEND_RETRIES
        # Live ctx-limit recheck (plan 075 WS1.4): re-derive the compaction trigger
        # from current memory conditions. Hysteresis: apply only a >10% move, so the
        # limit doesn't jitter with ordinary turn-to-turn allocator noise.
        if self._ctx_limit_fn is not None:
            try:
                fresh = self._ctx_limit_fn()
                if fresh and abs(fresh - self.ctx_limit) > 0.10 * self.ctx_limit:
                    log.info("GOVERNOR ctx_limit %d -> %d (live memory recheck)",
                             self.ctx_limit, fresh)
                    self.ctx_limit = fresh
            except Exception:  # noqa: BLE001 — a pressure probe must never kill a turn
                pass
        # ds4-style warm start: on a cold cache, load the system+tools KV from disk
        # (or prefill+persist it once) so the first turn doesn't re-prefill the
        # ~3.2k-token stable prefix every session. Cheap no-op on a warm cache.
        if self.engine.cache_dir and not self.engine._cached_ids:
            try:
                status, n = self.engine.warm_prefix(self._stable_prefix_ids(),
                                                     should_stop=self._should_stop)
                log.info("CACHE warm-start %s: %d prefix tokens (disk KV cache)", status, n)
                if status == "hit":
                    self._emit("info", f"  [warm start: {n:,} prefix tokens from disk cache]")
            except Exception as e:  # never let cache warming break a turn
                log.warning("warm_prefix failed: %s", e)
        # @file mentions: inline any referenced files so the model has them without a
        # read round-trip (intent classification below still uses the original text).
        expanded, attached = expand_mentions(user_text)
        for p in attached:
            self._emit("info", f"  [attached @{_disp_path(p)}]")
        # Intent (see classify_intent) drives both the plan-mode prefix here and the
        # answer-on-paper nudge below; compute once on the ORIGINAL text. In plan mode a
        # non-action ask (a question) gets the conceptual prefix so it's answered in
        # prose instead of forced into a plan file; only a real change request keeps the
        # plan-file mandate.
        _intent = classify_intent(user_text)
        prompt = (_plan_prefix(_intent) + expanded) if self.mode == "plan" else expanded
        self.messages.append({"role": "user", "content": prompt})
        log.info("TURN start | cwd=%s | mode=%s | attached=%s | query=%r",
                 os.getcwd(), self.mode, attached, redact(user_text[:200]))
        recent_sigs = []  # forge-style loop guard: detect repeated identical tool calls
        self._loop_nudges = 0
        self._validate_fails = 0  # typia self-repair rounds this turn (telemetry)
        self._last_reject_sig = None  # (name,args) of the last validation-rejected call
        self._reject_repeats = 0      # consecutive identical rejections (loop breaker)
        unverified_edit = False  # files changed but not run/tested since
        verify_nudges = 0
        did_work = False  # a substantive tool (not plan/done) ran this turn
        empty_done_nudges = 0
        done_recheck_done = False  # the one-shot deliverable recheck fired this turn
        recheck_fix_edits = 0  # landed edits AFTER the recheck fired (plan 070 spiral cap)
        made_edit = False  # an edit/write/replace/insert actually landed this turn
        answer_nudges = 0
        noop_edit_streak = 0  # consecutive edits that failed to land (plan 047 loop-break)
        last_edit_fail_kind = None  # 'noop'/'nomatch' of the latest dead edit (nudge choice)
        break_nudges = 0      # times we escalated a stuck edit this turn
        readonly_streak = 0   # consecutive steps with substantive tools but no landed edit
        gate_nudges = 0       # times the investigation->edit gate fired this turn
        subagent_sigs = set() # (description, prompt) of sub-agents already spawned this turn
        truncation_nudges = 0  # times we pushed past a token-cap truncation this turn
        landing_nudges = 0  # one-shot "you're out of steps, land the edit" near the cap
        consecutive_failed_bash = 0  # back-to-back errored bash with no edit (thrash)
        thrash_nudges = 0
        plan_reviews = 0    # one-shot "re-read the plan you just wrote" (levers.plan_review)
        # Files edited this turn -> mtime at last syntax check. Bash can mutate files
        # too (sed -i, python rewrites) but bypasses the write/edit syntax gate; watch
        # edited files and re-check them after any bash that touched them (iter-2:
        # sphinx-7440's file survived 9 blind `sed -i` "fixes" unparseable, unflagged).
        edited_syntax_watch: dict = {}
        think_cap_hits = 0  # soft think-cap firings this turn (plan 039; drives escalation)
        repeat_stops = 0  # degenerate-repetition cut-offs this turn (3rd aborts the turn)
        # Runaway-turn governor state (plan 040). turn_start drives the optional wall
        # budget; gov_band is the highest budget checkpoint already evaluated; gov_progress
        # tracks whether a change landed+verified within the CURRENT band (resets each time
        # a checkpoint is crossed, so every band must re-earn progress); gov_soft_fired
        # bounds the soft nudge to one per turn.
        self.budget_note = None
        turn_start = time.monotonic()
        gov_band = 0
        gov_progress = False
        gov_soft_fired = False
        # Action vs Q&A intent (see classify_intent): a task that asks to change code
        # should END in an applied edit, not a prose explanation. Telemetry caught the
        # model navigating to the right function then "answering on paper" without
        # applying the fix; action_task arms the nudge that pushes it to actually edit,
        # and read_only_intent exempts genuine explain-only asks. (_intent computed above,
        # where it also selects the plan-mode prefix.)
        action_task = self.mode != "plan" and _intent["action"]
        read_only_intent = _intent["read_only"]
        # Progress-aware step cap (see guardrails.extend_step_cap): max_steps is the
        # WINDOW size, not a hard kill. A window that landed+verified a change earns
        # another window (warm cache — no re-prefill, unlike a governor rollover); a
        # window that didn't ends the turn with a progress note banked. The absolute
        # ceiling bounds a turn that keeps "progressing" without ever finishing.
        step_cap = self.max_steps
        hard_ceiling = self.max_steps * guardrails.STEP_CAP_CEILING
        landed_in_window = False  # a change landed AND verified since this window began
        for step in range(hard_ceiling):
            # Flush at the TOP of each step: the previous step's tool results are now in
            # `messages`, and a harness SIGKILL at the task timeout lands mid-step. A
            # trajectory written only at exit would be lost on exactly the long trials.
            self._atif_sync()
            if step >= step_cap:
                new_cap = guardrails.extend_step_cap(
                    step_cap, self.max_steps, landed_in_window, hard_ceiling)
                if new_cap is None:
                    break
                log.info("STEP-CAP extended at step %d: %d -> %d (landed+verified "
                         "progress in the last window)", step, step_cap, new_cap)
                self._emit("info", "  [still landing verified changes — extending the "
                                   f"turn to {new_cap} steps]")
                step_cap = new_cap
                landed_in_window = False
                landing_nudges = 0  # the new window gets its own forced-landing nudge
            # Runaway-turn governor (plan 040): watch the GLOBAL trajectory — cumulative
            # prefill tokens (+ optional wall clock) vs progress made. It fires only when a
            # budget checkpoint is CROSSED, evaluating the just-completed band's progress:
            # a change landed+verified in that band resets the checkpoint (slow-but-working
            # turns are never interrupted); no progress -> one soft nudge at ~50%, then bank
            # a deterministic progress note and end the turn at ~80% so the caller can
            # relaunch fresh. Off under CHAD_NO_GOVERNOR / when no budget is configured.
            if not self._no_governor:
                frac = guardrails.budget_fraction(
                    self.prefill_tokens, self._turn_budget_tokens,
                    time.monotonic() - turn_start, self._turn_budget_s)
                new_band = guardrails.budget_band(frac)
                gov, gov_band, gov_progress = guardrails.advance_governor(
                    gov_band, new_band, gov_progress, gov_soft_fired)
                if gov == "hard":
                    self.budget_note = guardrails.progress_note(self.messages)
                    log.info("GOVERNOR hard-stop at step %d: %d/%s prefill tokens, %.0fs — "
                             "banking progress note, ending turn", step, self.prefill_tokens,
                             self._turn_budget_tokens, time.monotonic() - turn_start)
                    self._emit("info", "  [turn hit its budget with no landed+verified "
                                       "change — stopping and banking a progress note]")
                    return f"{guardrails.BUDGET_SENTINEL} {self.budget_note}"
                if gov == "soft":
                    gov_soft_fired = True
                    log.info("GOVERNOR soft-nudge at step %d: %d/%s prefill tokens", step,
                             self.prefill_tokens, self._turn_budget_tokens)
                    self.messages.append({"role": "tool", "name": "edit",
                                          "content": guardrails.GOVERNOR_SOFT_NUDGE})
            # Forced landing (A): inside the last few steps with nothing cleanly applied,
            # tell the model to stop exploring and commit its edit before the hard cap —
            # otherwise the loop just dies at max_steps with the task untouched.
            land = guardrails.landing_nudge(
                step, step_cap, made_edit, unverified_edit, landing_nudges)
            if land:
                landing_nudges += 1
                log.info("LANDING nudge at step %d (remaining=%d, made_edit=%s, "
                         "unverified_edit=%s)", step, step_cap - step,
                         made_edit, unverified_edit)
                self.messages.append({"role": "tool", "name": "edit", "content": land})
            # Mid-run steering (improve 01): user text typed while the turn ran is
            # injected HERE, between steps — after the previous step's tool results,
            # before the next generation (this point sits at the top of the iteration
            # so the retry/nudge `continue` paths above also pass through it). It rides
            # the same synthetic tool-role path as the guardrail nudges, a template
            # shape proven not to confuse Ornith; and it is a pure append, so the warm
            # KV prefix stays valid and the steer prefills only itself — vs an
            # interrupt's lost work + big re-prefill on the non-trimmable cache.
            if self._drain_steering is not None:
                for steer in self._drain_steering():
                    log.info("STEER injected at step %d: %r", step, redact(steer[:200]))
                    self.messages.append({
                        "role": "tool", "name": "steer",
                        "content": "[user steering — this overrides prior instructions "
                                   "for the rest of the turn]\n" + steer})
                    self._emit("info", "  [steering injected]")
            # The engine diffs this full render against its live KV cache and prefills
            # only the appended tokens, so a plain re-render IS the cache-extension path
            # (truncated-turn divergence is handled inside engine._sync_to, not here).
            _t0 = time.perf_counter()
            prompt_ids = self._render()
            _render_s = time.perf_counter() - _t0
            _pre_compact_len = len(prompt_ids)
            _t0 = time.perf_counter()
            prompt_ids = self._compact_if_needed(prompt_ids)
            _compact_s = time.perf_counter() - _t0
            # Did compaction materially shrink the render this step? Derived from the
            # length delta at the call site (not compaction internals — see plan 034),
            # so a big next-prefill can be named as a re-prefill rather than a mystery.
            compacted = len(prompt_ids) < _pre_compact_len
            # Live context gauge + honest activity verb (overrides any stale tool
            # verb like "Searching" left over from the previous step).
            self._emit("ctx", str(len(prompt_ids)))
            self._emit("status", "Thinking")

            # Plan 039 soft think-cap: when armed (self.think_budget set) and thinking is
            # on, stop this step's generation once its <think> run exceeds a budget so
            # reasoning can't balloon. The budget escalates with the turn's stuck-signals
            # (prior caps this turn + loop/thrash/verify nudges) so a genuinely hard step
            # gets more room instead of being chunked repeatedly. None => generation is
            # byte-identical to before (the mechanism is off by default).
            think_stop = None
            think_cap = None
            if self.think_budget and self.thinking:
                stuck = (think_cap_hits + self._loop_nudges + thrash_nudges
                         + verify_nudges)
                think_cap = guardrails.think_budget(stuck, base=self.think_budget)

                def think_stop(text_so_far, n, _cap=think_cap):
                    # Still inside <think> (no close emitted) and over budget -> stop; the
                    # turn is then force-closed by close_unclosed_think, a prefix-safe append.
                    return n >= _cap and "</think>" not in text_so_far

            # Degenerate-repetition stop (default ON): greedy decode can lock into
            # repeating one short string until the 8192-token cap — ~4 minutes of dead
            # generation per occurrence at 9B decode speed. Checked every 16 tokens on
            # the generation's tail; a hit stops the step (rep_fired tells this stopper
            # apart from the think-cap, which shares stats.stop_condition_fired) and the
            # branch below nudges the model out of the loop instead of grinding on.
            rep_fired = [False]
            stop_condition = None
            if think_stop is not None or not self._no_repeat_guard:
                def stop_condition(text_so_far, n):
                    if think_stop is not None and think_stop(text_so_far, n):
                        return True
                    if (not self._no_repeat_guard and n % 16 == 0
                            and guardrails.degenerate_tail(text_so_far)):
                        rep_fired[0] = True
                        return True
                    return False

            view = _StreamView(self._emit, started_in_think=self.thinking) if stream else None
            gen_count = [0]  # decoded chunks this step (~tokens); fed to the live ↓ counter

            def on_token(t):
                if view:
                    view.feed(t)
                gen_count[0] += 1
                # Throttle: a status emit per ~16 tokens keeps the queue cheap while the
                # bottom-line ↓ counter still climbs visibly (the refresher renders ~20 Hz).
                if gen_count[0] % 16 == 0:
                    self._emit("gen", str(gen_count[0]))

            def on_prefill(new, cached):
                # A large prefill (e.g. a full recompute after compaction on a
                # non-trimmable cache) blocks for a while — name the cause and let the
                # progress hook below advance a %, so the wait is legible and ctrl-c'able
                # rather than a frozen spinner.
                if new > 2000:
                    self._emit("status", "Re-prefilling after compaction"
                               if compacted else "Prefilling context")

            last_pct = [-1]

            def on_prefill_progress(done, total):
                # Forward an advancing prefill % to the status line, throttled to whole-
                # percent changes (≤101 emits per prefill). Only for big prefills — small
                # appends stay silent (verb only), matching on_prefill's >2000 gate.
                if total <= 2000:
                    return
                pct = int(100 * done / total)
                if pct != last_pct[0]:
                    last_pct[0] = pct
                    self._emit("prefill", f"{done}/{total}")

            try:
                text, stats = self.engine.generate(
                    prompt_ids, max_tokens=self.max_gen_tokens, on_token=on_token,
                    should_stop=self._should_stop, on_prefill=on_prefill,
                    on_prefill_progress=on_prefill_progress, stop_condition=stop_condition)
            except BackendError as e:
                # A transient backend fault (5xx / mid-stream error chunk) used to escape
                # run_turn and kill the process from cli.main — forfeiting the rest of an
                # unattended task's budget (TB2 make-mips-interpreter died at 721s of a
                # 1770s budget on a single llama.cpp 500). Re-issue the step instead: the
                # prompt is rebuilt from `messages` each iteration and the failed
                # generation was never appended, so a retry is a clean re-roll — and at
                # temp>0 a resample usually clears a parser-rejected completion.
                if view:            # the retry skips the normal close() below
                    view.close()
                if (not levers.enabled("backend_retry") or not e.transient
                        or self._backend_retries >= _MAX_BACKEND_RETRIES):
                    raise
                self._backend_retries += 1
                log.warning("backend error (retry %d/%d): %s",
                            self._backend_retries, _MAX_BACKEND_RETRIES, e)
                self._emit("status", f"Backend error; retrying "
                                     f"({self._backend_retries}/{_MAX_BACKEND_RETRIES})")
                time.sleep(min(2 ** self._backend_retries, 8))
                continue
            if gen_count[0] % 16:  # flush the final (un-throttled) count for the ↓ readout
                self._emit("gen", str(gen_count[0]))
            # The generation ran to the token cap: the turn was cut off, not finished.
            # A truncated assistant turn is NOT a final answer (and on Ornith's
            # non-trimmable cache its decoded text won't re-tokenize identically, so it
            # also forces the next-step re-prefill). Tracked so the no-call branch can
            # tell "truncated mid-thought" apart from "deliberately answered."
            if self._atif:   # one entry per SUCCESSFUL generate — a retried step appends
                             # nothing, keeping this list aligned with assistant messages
                self._atif_stats.append(
                    {"prompt_tokens": stats.prompt_tokens,
                     "cached_tokens": stats.cached_tokens,
                     "generated_tokens": stats.generated_tokens})
            hit_cap = stats.generated_tokens >= self.max_gen_tokens
            self.gen_tokens += stats.generated_tokens
            self.gen_time += stats.gen_s
            self.prefill_tokens += stats.prompt_tokens
            self.peak_ctx = max(self.peak_ctx, len(prompt_ids))
            self.forwards += stats.forwards
            self.draft_proposed += stats.draft_proposed
            self.draft_accepted += stats.draft_accepted
            if view:
                view.close()
                if view.saw_prose:
                    self._emit("stream", "\n")
            # strip any trailing special tokens the template will re-add — and any
            # LEAKED special-token literal anywhere in the text. A quantized model
            # can emit a stray marker like <|mask_end|> mid-turn (NIGHT-7
            # django-14404 r3: one leaked at step 12 and the turn read as a clean
            # final answer, ending the run rc=0 with an unverified edit); scrubbed
            # here so it can neither pollute the transcript nor masquerade as
            # content.
            text = _SPECIAL_TOKEN_RE.sub("", text).rstrip()

            # Interrupted (often mid-prefill, so text is empty): stop cleanly without
            # appending an empty assistant turn.
            if self._should_stop():
                self.interrupted = True
                if text:
                    self.messages.append({"role": "assistant",
                                          "content": close_unclosed_think(text, self.thinking)})
                self._emit("info", "  [interrupted]")
                return "[interrupted]"

            # Close a think block left dangling by a token-cap/interrupt truncation so the
            # stored turn stays a prefix of the live KV cache (else: full re-prefill next
            # step on the non-trimmable cache). See close_unclosed_think.
            self.messages.append({"role": "assistant",
                                  "content": close_unclosed_think(text, self.thinking)})

            # Estimate reasoning overhead: the generation opens inside <think> (the
            # template emits the opening tag), so everything up to </think> is thinking.
            # A soft-cap stop (plan 039) fires only while still inside <think>, so ALL of
            # this step's tokens are reasoning — count them so think-token telemetry (the
            # metric the budget is measured against) doesn't under-report the capped runs.
            # (A repetition stop can also land inside think — same accounting; one that
            # fired after </think> falls to the fraction path below like any other turn.)
            if stats.stop_condition_fired and "</think>" not in text:
                self.think_tokens += stats.generated_tokens
            elif "</think>" in text and len(text):
                frac = len(text.split("</think>", 1)[0]) / len(text)
                self.think_tokens += int(stats.generated_tokens * frac)

            log.info("step %d: %d tok @ %.1f tok/s | prefill %d new + %d cached | "
                     "accept %.2f", step, stats.generated_tokens, stats.tok_per_s,
                     stats.prompt_tokens, stats.cached_tokens, self.accept_rate)
            # Plan 035: env-gated per-step prefill telemetry (one row per prefill event).
            # Guarded so an unset trace pays only this truthiness check — no string/dict/IO.
            # sync_kind names *why* this step prefilled what it did, derived purely from
            # cached_tokens vs the prior step's total cache length + the `compacted` flag
            # (no engine internals): a step whose cached prefix shrank re-prefilled lost
            # content; tagging that by cause is what feeds the A/B/C attribution.
            if _PREFILL_TRACE:
                common = stats.cached_tokens
                if common >= self._prefill_trace_prev:
                    sync_kind = "append"          # prefix fully retained; only new tail
                elif compacted:
                    sync_kind = "reprefill-compaction"  # head rewrite shrank the render
                elif common > 0:
                    sync_kind = "warm-reload"     # diverged but salvaged a prefix (cheap)
                else:
                    sync_kind = "reprefill-other" # cold reset (think-mismatch / mid-edit)
                self._prefill_trace_seq += 1
                _trace_prefill({
                    "seq": self._prefill_trace_seq, "step": step,
                    "prompt_tokens": stats.prompt_tokens,
                    "cached_tokens": stats.cached_tokens,
                    "prefill_s": round(stats.prefill_s, 4),
                    "gen_tokens": stats.generated_tokens,
                    "gen_s": round(stats.gen_s, 4),
                    "compacted": compacted, "sync_kind": sync_kind,
                    "peak_ctx": len(prompt_ids),
                    # Loop overhead outside the engine: chat-template re-tokenization
                    # and compaction wall time this step, plus the tools the *previous*
                    # step ran (they produced the transcript this step prefilled).
                    "render_s": round(_render_s, 4),
                    "compact_s": round(_compact_s, 4),
                    "prev_tools": self._trace_tools_pending,
                })
                self._trace_tools_pending = []
                self._prefill_trace_prev = (
                    common + stats.prompt_tokens + stats.generated_tokens)
            # Cache divergence: a mid-session step that re-prefills everything (0 cached)
            # means the re-rendered conversation diverged from the cache — on a
            # non-trimmable cache that's a full, costly rebuild (usually triggered by a
            # max_tokens-truncated turn whose decoded text doesn't re-tokenize identically).
            if step > 0 and stats.cached_tokens == 0 and stats.prompt_tokens > 2000:
                log.warning("CACHE DIVERGENCE at step %d: full re-prefill of %d tokens "
                            "(non-trimmable cache reset — see prior turn for a truncated "
                            "generation)", step, stats.prompt_tokens)

            # Degenerate-repetition stop fired: the step's output locked into repeating
            # one short string (see guardrails.degenerate_tail) and was cut off early —
            # a truncation of garbage, never a final answer, so don't parse it. The
            # stored turn is prefix-safe (generated tokens are already in the KV cache;
            # close_unclosed_think handled a mid-think cut). Nudge the model out of the
            # loop and continue; if the nudge can't break the decode loop, abort rather
            # than crawl to max_steps one stall at a time.
            if stats.stop_condition_fired and rep_fired[0]:
                repeat_stops += 1
                self._emit("info", "  [output degenerated into repetition — cut off]")
                log.info("REPEAT-STOP at step %d after %d tok (stops=%d this turn)",
                         step, stats.generated_tokens, repeat_stops)
                if guardrails.repeat_stop_abort(repeat_stops):
                    log.info("END step %d: REPEAT ABORT (nudges not breaking the loop)",
                             step)
                    return ("[stopped: the model keeps degenerating into repetitive "
                            "output. Try rephrasing the request or breaking it into "
                            "smaller steps (docs/troubleshooting.md maps symptom → "
                            "fix).]")
                self.messages.append({"role": "tool", "name": "edit",
                                      "content": guardrails.REPEAT_STOP_NUDGE})
                continue

            # Plan 039: the soft think-cap fired — generation was stopped while still
            # inside <think> (the reasoning run exceeded this step's budget). The assistant
            # turn was force-closed by close_unclosed_think above (it ends with </think>,
            # no content after), so it re-tokenizes as a strict prefix of the live KV cache
            # — next step is a ~2-token append, not a rebuild (the close_unclosed_think
            # contract). Count it (escalates the budget for the next step so we don't chunk
            # forever) and continue; the model resumes reasoning / acts next step.
            if stats.stop_condition_fired:
                think_cap_hits += 1
                self.think_capped += 1
                # Display-only signal (plan 057): tell the surface a step was trimmed this
                # turn, carrying the running per-turn count so the status line can show it.
                # Only ever emitted when the cap is armed (self.think_budget set), so the
                # default path is byte-identical — the surface renders nothing when off.
                self._emit("thinkcap", str(think_cap_hits))
                log.info("THINK-CAP at step %d: closed <think> after %d tok "
                         "(cap=%d, hits=%d)", step, stats.generated_tokens,
                         think_cap, think_cap_hits)
                continue

            calls = parse_tool_calls(text)
            if not calls:
                # No tool call this step. Decide whether this is a genuine final answer
                # or a stall to push past. Three stalls telemetry caught, in priority
                # order: (1) TRUNCATED — generation hit the token cap mid-thought, so it
                # isn't an answer at all; (2) ANSWERED ON PAPER — produced code / described
                # an edit but never applied it (the demonstrated "write test cases" bug:
                # the old keyword gate missed "write" entirely); (3) UNVERIFIED EDIT — it
                # edited but never ran the check. Each nudge is bounded so real answers and
                # genuinely-stuck cases still escape.
                has_code = "```" in text
                # A CLOSED tool-call block that still parsed to nothing is a garbled
                # call, not an answer (parse salvage handles most; this is the
                # backstop for what it can't reconstruct). Markers checked on the
                # think-stripped text so a call drafted inside <think> doesn't count.
                _stripped_for_markers = strip_think(text)
                garbled = ("<tool_call>" in _stripped_for_markers
                           or "</function>" in _stripped_for_markers
                           or "<function=" in _stripped_for_markers)
                kind, nudge = guardrails.nudge_for_no_calls(
                    text, hit_cap, made_edit, unverified_edit, read_only_intent,
                    action_task, truncation_nudges, answer_nudges, verify_nudges,
                    _has_open_tool_call(text), garbled_call=garbled)
                if kind == "truncated":
                    truncation_nudges += 1
                elif kind == "no-edit":
                    answer_nudges += 1
                elif kind == "unverified-edit":
                    verify_nudges += 1
                if nudge:
                    log.info("END-ANSWER rejected step %d: %s (hit_cap=%s, has_code=%s, "
                             "action_task=%s)", step, kind, hit_cap, has_code, action_task)
                    self.messages.append({"role": "tool", "name": "edit", "content": nudge})
                    continue
                # Iter-3 did-nothing gate: in auto/headless mode (every benchmark run), a
                # turn that ends having executed ZERO real tools is never a legitimate
                # completion — the keyword intent classifier misses tasks like "extract the
                # secret and save it" (no action verb), so action_task is False and the
                # gate below wouldn't fire (TB2 vulnerable-secret shipped a 28-token garble
                # as its answer). read_only/explain asks are exempt. Banks a note so
                # auto-continue relaunches fresh instead of shipping nothing.
                did_nothing = self.mode == "auto" and not read_only_intent and not did_work
                if (action_task and not read_only_intent
                        and (not made_edit or unverified_edit)) or did_nothing:
                    # Iter-2 no-empty-diff gate: an action task may not END on a prose
                    # "final answer" while no change landed (or the change is
                    # unverified) — the demonstrated failures (django-14007,
                    # sphinx-9230): 49–97s bails accepted as final answers with an
                    # empty diff and 97% of the budget unused. Bank a progress note
                    # and end as a hard stop, so --auto-continue (headless) or the
                    # user's 'continue' (TUI) relaunches a fresh attempt with the
                    # note instead of silently shipping nothing.
                    self.budget_note = guardrails.progress_note(self.messages)
                    log.info("END step %d: FINAL ANSWER blocked by no-empty-diff gate "
                             "(made_edit=%s, unverified_edit=%s) — progress note banked",
                             step, made_edit, unverified_edit)
                    self._emit("info", "  [turn ended without a landed+verified change "
                                       "— progress note banked; say 'continue' to retry]")
                    return ("[stopped: the turn ended without applying a verified "
                            "change — say 'continue' to resume]")
                log.info("END step %d: model produced a FINAL ANSWER, no tool calls "
                         "(did_work=%s, made_edit=%s, unverified_edit=%s)",
                         step, did_work, made_edit, unverified_edit)
                return strip_think(text).strip()
            log.info("step %d: model emitted %d tool call(s): %s",
                     step, len(calls), ", ".join(n for n, _ in calls))

            # Terminal tool -> end the turn cleanly, but enforce verify-before-done
            # (forge's prerequisite idea): if files were changed and nothing has been
            # run since, send the model back to actually test its work.
            terminal = next((a for n, a in calls if n in TERMINAL), None)
            if terminal is not None:
                # Don't accept `done` if the model only narrated/planned without running
                # any real tool (the markdown-code-fence failure mode).
                log.info("step %d: model says DONE (summary=%r) | did_work=%s "
                         "unverified_edit=%s", step, terminal.get("summary"),
                         did_work, unverified_edit)
                rejection = guardrails.done_rejection(
                    did_work, unverified_edit, empty_done_nudges, verify_nudges)
                if self.mode == "plan" and rejection == "verify":
                    # Writing the plan file marks unverified_edit, but in plan mode the
                    # plan IS the deliverable — there is nothing to run/verify. Accept.
                    rejection = None
                if rejection == "empty":
                    empty_done_nudges += 1
                    log.info("DONE rejected: no real work yet -> nudge #%d", empty_done_nudges)
                    self.messages.append({
                        "role": "tool", "name": "done",
                        "content": "[you have not actually done anything yet — no file was "
                                   "read or changed. Markdown code fences are not executed. "
                                   "Use real <tool_call> blocks: grep/read the file, edit it, "
                                   "run the check with bash. Then call done.]",
                    })
                    continue
                if rejection == "verify":
                    verify_nudges += 1
                    log.info("DONE rejected: edits not verified -> nudge #%d", verify_nudges)
                    self.messages.append({
                        "role": "tool", "name": "done",
                        "content": "[not done yet: you changed files but have not run anything "
                                   "to verify them. Run the project's tests (or the code) with "
                                   "bash, check the output is correct, then call done. If a test "
                                   "fails, fix the code first.]",
                    })
                    continue
                if action_task and not read_only_intent and self.mode != "plan" \
                        and (not made_edit or unverified_edit):
                    # Same no-empty-diff gate as the prose-final-answer path: `done`
                    # with nothing landed (or landed-unverified after the verify
                    # nudges ran out) becomes a resumable hard stop, not a success
                    # (matplotlib-25332 r3: done accepted at 84s with edits in tree
                    # and zero successful post-edit commands).
                    self.budget_note = guardrails.progress_note(self.messages)
                    log.info("END step %d: DONE blocked by no-empty-diff gate "
                             "(made_edit=%s, unverified_edit=%s) — progress note banked",
                             step, made_edit, unverified_edit)
                    self._emit("info", "  [done rejected: no landed+verified change — "
                                       "progress note banked; say 'continue' to retry]")
                    return ("[stopped: `done` was called without a landed+verified "
                            "change — say 'continue' to resume]")
                # Iter-3 deliverable recheck (levers.done_spec_recheck): one last self-check
                # that the required outputs actually exist at the right path/format before
                # we accept done — the hidden container-end-state verifier gives no second
                # chance. Fires at most once per turn; a task that re-checks and is genuinely
                # complete just calls done again next step.
                if self.mode != "plan" and levers.enabled("done_spec_recheck") \
                        and guardrails.done_spec_recheck(
                            did_work, unverified_edit, done_recheck_done, read_only_intent):
                    done_recheck_done = True
                    log.info("DONE deferred for deliverable recheck (step %d)", step)
                    self.messages.append({"role": "tool", "name": "done",
                                          "content": guardrails.DONE_SPEC_RECHECK})
                    continue
                log.info("END step %d: DONE accepted | summary=%r", step,
                         terminal.get("summary"))
                return terminal.get("summary") or text or "Done."

            # Loop guard: count identical tool-call sets across the whole turn (not a
            # sliding window) so alternating cycles like read A / read B / read A are
            # caught too. 3rd identical occurrence -> nudge; if nudges don't help, abort.
            sig = guardrails.loop_signature(calls)
            seen_before = recent_sigs.count(sig)
            recent_sigs.append(sig)
            if guardrails.is_repeat_loop(seen_before):
                self._loop_nudges += 1
                log.info("LOOP detected at step %d (identical call set seen %dx) -> nudge #%d",
                         step, seen_before + 1, self._loop_nudges)
                if guardrails.loop_should_abort(self._loop_nudges):
                    log.info("END step %d: LOOP ABORT (nudges exhausted)", step)
                    return ("[stopped: the model is stuck in a loop, repeating the same "
                            "tool calls without making progress. Tip: a smaller, scoped "
                            "ask recovers this — name the exact file you want changed "
                            "(docs/troubleshooting.md maps symptom → fix).]")
                self.messages.append({
                    "role": "tool", "name": calls[0][0],
                    "content": "[loop detected: you have made this exact tool call 3 times "
                               "with no new progress. Do NOT repeat it. Either run the test "
                               "with bash to verify, take a different action, or — if the task "
                               "is already done — stop and summarize the result.]",
                })
                continue

            # Governor progress watermark (plan 040): capture the edit flags before this
            # step's tools run so we can tell afterward whether a change actually LANDED
            # or a pending edit got VERIFIED this step — the forward motion that resets the
            # budget checkpoint (vs mere reads/greps, which don't count as progress).
            _gov_prev_made = made_edit
            _gov_prev_unverified = unverified_edit
            step_tool_chars = 0  # tool-output chars appended this step (see _step_tool_cap)
            for name, args in calls:
                render_tool_start(self._emit, name, args)
                # typia stages 2+3: coerce loosely-typed args toward the schema
                # ("10"->10, '["x"]'->[...]) and validate. On failure, feed the
                # model its own args annotated with exactly which fields are
                # wrong (self-repair loop) instead of dispatching garbage.
                if VALIDATE:
                    coerced, verrs = coerce_and_validate(name, args)
                    reject = render_repair(name, coerced, verrs) if verrs else None
                    detail = " | ".join(str(e) for e in verrs)
                else:  # legacy: terse missing-required check, no coercion
                    coerced, reject, detail = args, legacy_validate(name, args), ""
                fn = dispatch_for(name)
                if reject is not None:
                    result = reject
                    self._validate_fails += 1
                    # Loop breaker (plan): the same call rejected identically back-to-back
                    # means the repair message isn't landing (the model can't see the fix,
                    # or the call can't succeed as written). Escalate — tell it to stop
                    # re-emitting and to NOT fabricate the tool's result — instead of
                    # letting it flail. Signature is over (name, args) so a *different*
                    # attempt at the same tool resets the counter.
                    rej_sig = guardrails.loop_signature([(name, args)])
                    if rej_sig == self._last_reject_sig:
                        self._reject_repeats += 1
                    else:
                        self._last_reject_sig, self._reject_repeats = rej_sig, 0
                    if self._reject_repeats >= 1:  # 2nd identical rejection
                        result += reject_escalation(name)
                        log.info("VALIDATE %s rejected identically %dx -> escalate",
                                 name, self._reject_repeats + 1)
                    log.info("VALIDATE %s rejected (%d): %s", name, self._validate_fails, detail)
                    render_tool_result(self._emit, name, args, result)
                    self.messages.append({"role": "tool", "name": name, "content": result})
                    continue
                if coerced != args:
                    log.info("VALIDATE %s coerced: %s -> %s", name,
                             args_preview(args), args_preview(coerced))
                args = coerced
                # Subagent/Task tool (plan 041): run a scoped sub-agent on a quarantined
                # cache and feed its condensed return back as this call's result. Handled
                # here (not via DISPATCH) because it needs the engine + a fresh Agent. Only
                # a top-level agent dispatches it; a sub-agent never sees `task` in its
                # schema, so if one somehow emits it we fall through to the unknown-tool
                # repair path rather than nesting.
                if name == "task" and not self._subagent:
                    # Plan 047 — anti-respawn: a sub-agent is expensive and runs on a
                    # quarantined cache, so re-spawning the SAME (description, prompt) is
                    # almost always waste. The demonstrated failure: a localization
                    # sub-agent hit its step cap returning nothing usable, and the model
                    # re-ran the identical one, which capped out again — half the turn's
                    # budget gone. Refuse the duplicate and push the model to act directly.
                    _sub_sig = (str(args.get("description", "")).strip(),
                                str(args.get("prompt", "")).strip())
                    if _sub_sig in subagent_sigs and levers.enabled("subagent_no_respawn"):
                        result = ("[you already ran this exact sub-agent this turn — do NOT "
                                  "re-run it. Use what it returned, or do the work yourself "
                                  "now with grep/read to locate the code and edit to change "
                                  "it. Re-spawning the same task will not produce more.]")
                        render_tool_result(self._emit, name, args, result)
                        self.messages.append({"role": "tool", "name": name, "content": result})
                        continue
                    subagent_sigs.add(_sub_sig)
                    result = self._run_subagent(
                        args.get("description", ""), args.get("prompt", ""),
                        args.get("tools", "read-only"))
                    render_tool_result(self._emit, name, args, result)
                    # This branch `continue`s past the shared TOOL log below, so log here
                    # too — otherwise the whole sub-agent path is invisible in session.log.
                    log.info("TOOL task(%s) -> %s", args_preview(args), result_preview(result))
                    self.messages.append({"role": "tool", "name": name, "content": result})
                    did_work = True  # exploration counts as work (like a read), not an edit
                    if self._should_stop():
                        self.interrupted = True
                        self._emit("info", "  [interrupted]")
                        return "[interrupted]"
                    continue
                # Plan mode is read-only EXCEPT for writing the plan file itself:
                # write/edit are allowed only under ./plans/. Every other mutating
                # tool (bash, symbol edits) and any write outside ./plans/ is blocked.
                plan_write = (self.mode == "plan" and name in ("write", "edit")
                              and _under_plans(args.get("path", "")))
                _tool_s = 0.0  # stays 0 when the tool is blocked/denied (fn never ran)
                if self.mode == "plan" and is_mutating(name) and not plan_write:
                    result = ("[plan mode: only writing the plan file under ./plans/ is "
                              "allowed. Do not edit project files or run commands. "
                              "Investigate with read/grep/glob/repo_map, then write your "
                              "plan to ./plans/NNN-title.md.]")
                elif not plan_write and not self._confirm(name, args):
                    # A plan write is the expected action in plan mode, so it skips the
                    # confirm prompt; everything else still goes through _confirm.
                    result = "[denied by user]"
                else:
                    _t0 = time.perf_counter()
                    try:
                        result = fn(args, self._should_stop)
                        if plan_write and result.startswith("[wrote"):
                            self.last_plan_path = os.path.abspath(args["path"])
                    except Exception as e:  # noqa: BLE001 - surface tool errors to model
                        result = f"[tool error: {type(e).__name__}: {e}]"
                    _tool_s = time.perf_counter() - _t0
                    # Backstop: bound the prefill from any tool, AND from the step as a
                    # whole — several calls in one step stack into one prefill, so later
                    # results only get what's left of the step budget (floor-protected).
                    result = _clip_tool_result(result, cap=_step_tool_cap(step_tool_chars))
                    step_tool_chars += len(result)
                    if _PREFILL_TRACE:
                        self._trace_tools_pending.append([name, round(_tool_s, 4)])
                render_tool_result(self._emit, name, args, result)
                log.info("TOOL %s(%s) -> %s [%.2fs]", name, args_preview(args),
                         result_preview(result), _tool_s)
                self.messages.append({"role": "tool", "name": name, "content": result})
                # Plan-then-review, delivered as two messages instead of one instruction.
                # The plan-mode preamble already demands a Context section, exact paths,
                # numbered steps and verify commands, and the model still ships plans that
                # skip half of it — a standing rule in the preamble washes out by the time
                # the plan is being written. Asking for the review AFTER the artifact
                # exists puts the ask next to the thing it is about, which is the same
                # placement lesson as the keep-reading notice in `read`'s return value.
                if (plan_write and result.startswith("[wrote") and not plan_reviews
                        and levers.enabled("plan_review")):
                    plan_reviews += 1
                    log.info("PLAN REVIEW nudge for %s", self.last_plan_path)
                    self.messages.append({"role": "tool", "name": "read", "content": (
                        f"[plan written to {self.last_plan_path}. Before you call `done`, "
                        f"review it: `read` the file back and check it has (1) a Context "
                        f"section saying why, (2) exact file paths with current-state code "
                        f"excerpts, (3) numbered step-by-step changes, (4) the exact verify "
                        f"commands to run, (5) repo conventions, (6) explicit out-of-scope. "
                        f"An executor will follow this file WITHOUT access to this "
                        f"conversation, so anything only you know must be on the page. If "
                        f"any item is missing or vague, fix the file now with `edit`. Then "
                        f"call `done`.]")})
                # Update the guardrail bookkeeping flags (did_work / made_edit /
                # unverified_edit) for this tool result — see guardrails.update_work_flags.
                did_work, made_edit, unverified_edit = guardrails.update_work_flags(
                    name, args, result, did_work, made_edit, unverified_edit)
                # Plan 047: track consecutive edits that DIDN'T land (no-op / unmatched
                # `old`), so the model gets pushed to read-then-replace instead of looping
                # on variations of a broken edit. A landed edit of any kind resets it.
                # replace_lines/insert_lines were missing from this set until plan 073:
                # their rejects never fed the streak, so the 073 dogfood burned ~40 steps
                # of line-edit churn with only the (much slower) identical-call loop
                # detector escalating.
                if name in ("edit", "write", "replace_lines", "insert_lines",
                            "replace_symbol", "insert_symbol", "rename_symbol"):
                    noop_edit_streak = (noop_edit_streak + 1
                                        if guardrails.edit_failed_to_land(result) else 0)
                    last_edit_fail_kind = guardrails.edit_fail_kind(result)
                    if not guardrails.edit_failed_to_land(result):
                        recent_sigs.clear()  # a landed change resets the identical-call
                                             # loop guard: re-running the same verify
                                             # command after an edit is progress, not a
                                             # loop (false-fired 2x in dbf9dee0/20260713)
                        _wp = str(args.get("path", "") or "")
                        if _wp and os.path.exists(_wp):
                            try:
                                edited_syntax_watch[_wp] = os.path.getmtime(_wp)
                            except OSError:
                                pass
                        # Plan 070 — recheck spiral cap: a landed edit AFTER the
                        # deliverable recheck fired. The recheck is a verify-then-done
                        # pass; a real fix is one or two targeted edits. A run of them
                        # is the model re-editing already-correct output into a thrash
                        # (poly_two_bucket: 3 edits / 16k tok / 621s after the recheck
                        # on an answer that had already passed the verify gate). Stop
                        # and keep the result that was ship-ready when the recheck fired.
                        if done_recheck_done:
                            recheck_fix_edits += 1
                            if guardrails.recheck_spiral(recheck_fix_edits):
                                log.info("END step %d: recheck edit-spiral (%d "
                                         "post-recheck landed edits) — accepting the "
                                         "completed work", step, recheck_fix_edits)
                                self._emit("info", "  [deliverable recheck kept "
                                           "re-editing already-finished work — stopping "
                                           "and keeping the result]")
                                return ("Done (stopped a post-recheck edit spiral; the "
                                        "deliverable was already complete before the "
                                        "extra edits).")
                elif name == "bash":
                    # Re-check syntax of watched files a bash command rewrote — the
                    # sed -i escape hatch around the edit-tool syntax gate.
                    for _wp, _wmt in list(edited_syntax_watch.items()):
                        try:
                            _cur = os.path.getmtime(_wp)
                        except OSError:
                            continue
                        if _cur != _wmt:
                            edited_syntax_watch[_wp] = _cur
                            _warn = syntaxgate.check_syntax(_wp, None)
                            if _warn:
                                self.messages[-1]["content"] += _warn
                # Thrash counter (C): count back-to-back failed bash with no edit between.
                consecutive_failed_bash = guardrails.update_thrash(
                    name, result, consecutive_failed_bash)

            # Governor progress (plan 040): a fresh edit LANDED, or a pending edit got
            # VERIFIED (unverified_edit cleared by a clean bash) this step — real forward
            # motion, so mark the current budget band as having made progress (resets the
            # checkpoint that would otherwise fire the governor).
            if (made_edit and not _gov_prev_made) or (_gov_prev_unverified and not unverified_edit):
                gov_progress = True
                landed_in_window = True  # also earns a step-cap extension (same signal)

            # Plan 047 — edit loop-break: after ~2 consecutive edits that failed to land,
            # stop the model re-trying variations and send it to read-then-replace.
            brk = guardrails.edit_loop_break(noop_edit_streak, break_nudges,
                                             last_edit_fail_kind)
            if brk:
                break_nudges += 1
                noop_edit_streak = 0  # give the escalation a clean slate
                log.info("EDIT-LOOP-BREAK at step %d (streak reset) -> nudge #%d",
                         step, break_nudges)
                self.messages.append({"role": "tool", "name": "edit", "content": brk})

            # Plan 047 — investigation->edit gate: a step that ran substantive tools but
            # landed no edit is investigation; a long run of it on an action task means the
            # model is exploring instead of acting (it named the fix but grep-looped into an
            # empty patch). Steer it to edit before the loop-abort/step-cap kills the turn.
            # read_only/explain asks are exempt (nothing to edit).
            if made_edit and not _gov_prev_made:
                readonly_streak = 0
            elif did_work and not made_edit:
                readonly_streak += 1
            if not read_only_intent:
                gate = guardrails.investigation_gate(readonly_streak, made_edit, gate_nudges)
                if gate:
                    gate_nudges += 1
                    readonly_streak = 0
                    log.info("INVESTIGATION-GATE at step %d (streak, no edit) -> nudge #%d",
                             step, gate_nudges)
                    self.messages.append({"role": "tool", "name": "edit", "content": gate})

            # Break a flailing-probe run (e.g. guessing the test runner, repeated
            # `python -c import` checks) that the exact-call loop guard can't see because
            # each failing command differs by a few characters.
            thrash = guardrails.bash_thrash_nudge(consecutive_failed_bash, thrash_nudges)
            if thrash:
                thrash_nudges += 1
                log.info("THRASH nudge: %d consecutive failed bash -> nudge #%d",
                         consecutive_failed_bash, thrash_nudges)
                self.messages.append({"role": "tool", "name": "bash", "content": thrash})

            if self._should_stop():
                self.interrupted = True
                log.info("END step %d: INTERRUPTED by user", step)
                self._emit("info", "  [interrupted]")
                return "[interrupted]"
        # Step cap reached with no landed+verified change in the final window (or the
        # absolute ceiling hit). Bank a progress note — same contract as the governor
        # hard-stop — so the TUI's "continue" and one-shot --auto-continue can resume
        # instead of silently dropping the half-done task at the prompt.
        self.budget_note = guardrails.progress_note(self.messages)
        log.info("END: hit step cap (%d, ceiling %d) | did_work=%s unverified_edit=%s",
                 step_cap, hard_ceiling, did_work, unverified_edit)
        self._emit("info", "  [turn stopped at its step cap — progress note banked; "
                           "say 'continue' to resume]")
        return ("[stopped: hit the step cap before finishing — say 'continue' to "
                "resume, or re-scope the ask smaller (docs/troubleshooting.md)]")


def repl(engine: BaseEngine, yolo: bool, ctx_limit: int = 24000, resume: list = None,
         thinking: bool = True, ctx_limit_fn=None):
    agent = Agent(engine, yolo=yolo, ctx_limit=ctx_limit, thinking=thinking,
                  resume=resume, persist=True, ctx_limit_fn=ctx_limit_fn)
    label = engine.model_id.split("/")[-1] + (" + draft" if getattr(engine, "draft", None) else "")
    print(banner(label, ctx_limit, mode=agent.mode))
    print(f"{C_DIM}type a task, or /reset, /exit.{C_RST}")
    while True:
        try:
            line = input(f"{C_YEL}» {C_RST}").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        if line in ("/exit", "/quit"):
            break
        if line in ("/reset", "/clear"):
            agent = Agent(engine, yolo=yolo, ctx_limit=ctx_limit, thinking=thinking,
                          persist=True, ctx_limit_fn=ctx_limit_fn)
            engine.reset()
            print(f"{C_DIM}session reset.{C_RST}")
            continue
        if line == "/mode":
            print(f"{C_DIM}mode: {MODE_LABEL[agent.cycle_mode()]}{C_RST}")
            continue
        if line == "/compact":
            b, a = agent.compact_now()
            print(f"{C_DIM}compacted context: {b:,}→{a:,} tokens{C_RST}")
            continue
        if line == "/model":
            print(f"{C_DIM}model {engine.model_id} · context {engine.effective_ctx:,} "
                  f"(compact at {ctx_limit:,}) · mode {agent.mode}{C_RST}")
            continue
        if line == "/skills":
            from . import skills
            for ln in skills.summary_lines():
                print(f"{C_DIM}  {ln}{C_RST}")
            continue
        if line == "/mcp trust":
            from . import mcp
            mcp.trust()
            print(f"{C_DIM}trusted this project — its .mcp.json servers will connect "
                  f"on the next turn{C_RST}")
            continue
        if line.startswith("/mcp login"):
            from . import mcp
            name = line[len("/mcp login"):].strip()
            if not name:
                print(f"{C_DIM}usage: /mcp login <server>{C_RST}")
                continue
            print(f"{C_DIM}{mcp.login(name, emit=lambda m: print(f'{C_DIM}{m}{C_RST}'))}{C_RST}")
            continue
        if line == "/mcp":
            from . import mcp
            for ln in mcp.summary_lines():
                print(f"{C_DIM}  {ln}{C_RST}")
            continue
        if line == "/help":
            print(f"{C_DIM}/init /skills /mcp /mcp trust /mcp login <server> /reset /clear "
                  f"/compact /model /mode /exit · !cmd runs a shell command · "
                  f"@path attaches a file/dir{C_RST}")
            continue
        if line == "/init":
            agent.run_turn(INIT_PROMPT)
            agent.save()
            continue
        if line.startswith("!"):  # shell passthrough — run directly, don't call the model
            cmd = line[1:].strip()
            if cmd:
                from .tools import tool_bash
                print(f"{C_DIM}{tool_bash(cmd)}{C_RST}")
            continue
        agent.run_turn(line)
        agent.save()
