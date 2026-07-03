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

from . import compaction, guardrails, session
from .base_engine import BaseEngine
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
SUBAGENT_MAX_STEPS = 12
SUBAGENT_CTX_LIMIT = 16000
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


def _env_int(name, default=None):
    """Parse an int from env var `name`, or fall back to `default`. A non-numeric value
    warns and degrades to the default instead of raising — the CHAD_* budget knobs follow
    the repo's lenient-parse rule, so a typo can't abort Agent.__init__ (plan 052)."""
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        log.warning("ignoring non-integer %s=%r; using default %r", name, v, default)
        return default


def _env_float(name, default=None):
    """float sibling of `_env_int` — same lenient-parse contract."""
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        log.warning("ignoring non-float %s=%r; using default %r", name, v, default)
        return default


from .validate import VALIDATE, coerce_and_validate, legacy_validate, render_repair

# Validation (VALIDATE knob, legacy_validate baseline) lives in validate.py, the
# single source of truth shared with toolcall_parse.py — imported above.

# Plan 035 (measurement spike): env-gated per-step prefill telemetry. Resolved ONCE at
# import (like diag._DISABLED) so the hot loop pays only a truthiness check when unset —
# zero string/dict/IO work, per plan 020's "instrumentation must not tax the product"
# rule. Set CHAD_PREFILL_TRACE=path/to/trace.jsonl to capture one JSON row per engine
# generate() call (== one prefill event) for scripts/prefill_tax.py to analyze offline.
_PREFILL_TRACE = os.environ.get("CHAD_PREFILL_TRACE") or None


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
                 confirm=None, should_stop=None, thinking: bool = True,
                 max_gen_tokens: int = 8192, resume: list = None, persist: bool = False,
                 think_budget: int = None, turn_budget_tokens: int = None,
                 turn_budget_s: float = None, subagent: bool = False,
                 subagent_tools: str = "read-only", session_id: str = None):
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
            think_budget = _env_int("CHAD_THINK_BUDGET")
        self.think_budget = think_budget
        self.ctx_limit = ctx_limit  # prompt-token budget before compaction kicks in
        # Runaway-turn governor (plan 040): a per-turn budget on cumulative prefill tokens
        # (and optional wall-clock) that ends a turn which has burned through a checkpoint
        # WITHOUT landing+verifying a change — banking a deterministic progress note so the
        # caller can relaunch fresh (shedding both the ramble and the huge prefill a stuck
        # weak model drags around). Off entirely under CHAD_NO_GOVERNOR=1 (A/B family). The
        # token budget defaults to 3× the context limit — far above a normal task's use
        # (passing eval tasks: 14–35k prefill tokens; the pathological timeout tail:
        # 130–187k). Wall budget is off unless set: interactively the human is the wall
        # clock; evals/one-shot set it via --turn-budget-s / CHAD_TURN_BUDGET_S.
        self._no_governor = bool(os.environ.get("CHAD_NO_GOVERNOR"))
        if turn_budget_tokens is None:
            turn_budget_tokens = _env_int("CHAD_TURN_BUDGET_TOKENS", max(0, 3 * self.ctx_limit))
        self._turn_budget_tokens = turn_budget_tokens
        if turn_budget_s is None:
            turn_budget_s = _env_float("CHAD_TURN_BUDGET_S")
        self._turn_budget_s = turn_budget_s
        # Set when a turn hard-stops on budget (like last_plan_path): holds the progress
        # note so the caller (TUI / one-shot / evals) can relaunch a fresh turn seeded
        # with it. Reset at the start of every run_turn.
        self.budget_note: str | None = None
        self.messages = [{"role": "system",
                          "content": build_subagent_prompt() if subagent
                          else build_system_prompt()}]
        if resume:
            self.messages += [m for m in resume if m.get("role") != "system"]
        self._emit = emit or _default_emit
        self._confirm_cb = confirm  # callable(name, args)->bool; None => input() prompt
        self._should_stop = should_stop or (lambda: False)
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
            self.messages, self._render, self._emit, self.ctx_limit, prompt_ids)

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
        idxs = [i for i, m in enumerate(self.messages)
                if m.get("role") == "tool" and self._COLLAPSED not in m["content"]]
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

    def _render(self):
        return self.engine.tok.apply_chat_template(
            self.messages, tools=self._active_schemas(), add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

    def _stable_prefix_ids(self):
        """The byte-identical head of every session: system prompt + tool schemas,
        up to (but not including) the first user turn. Computed by diffing two
        first-turn renders with different user text — their common token prefix is
        exactly the system+tools block. This is what warm_prefix checkpoints to disk
        so a cold start skips re-prefilling it (the template can't render a system
        message alone — it raises 'No user query found' — hence the diff trick)."""
        sysm = self.messages[0]
        def render1(u):
            return self.engine.tok.apply_chat_template(
                [sysm, {"role": "user", "content": u}], tools=self._active_schemas(),
                add_generation_prompt=True, enable_thinking=self.thinking)
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
                     and not os.environ.get("CHAD_NO_DESTRUCTIVE_GUARD")
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
        sub = None
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
            if sub.interrupted:
                return "[task interrupted]"
        return result or "[task returned nothing]"

    def run_turn(self, user_text: str, stream=True):
        self.interrupted = False
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
        prompt = (_PLAN_PREFIX + expanded) if self.mode == "plan" else expanded
        self.messages.append({"role": "user", "content": prompt})
        log.info("TURN start | cwd=%s | mode=%s | attached=%s | query=%r",
                 os.getcwd(), self.mode, attached, redact(user_text[:200]))
        recent_sigs = []  # forge-style loop guard: detect repeated identical tool calls
        self._loop_nudges = 0
        self._validate_fails = 0  # typia self-repair rounds this turn (telemetry)
        unverified_edit = False  # files changed but not run/tested since
        verify_nudges = 0
        did_work = False  # a substantive tool (not plan/done) ran this turn
        empty_done_nudges = 0
        made_edit = False  # an edit/write/replace/insert actually landed this turn
        answer_nudges = 0
        truncation_nudges = 0  # times we pushed past a token-cap truncation this turn
        landing_nudges = 0  # one-shot "you're out of steps, land the edit" near the cap
        consecutive_failed_bash = 0  # back-to-back errored bash with no edit (thrash)
        thrash_nudges = 0
        think_cap_hits = 0  # soft think-cap firings this turn (plan 039; drives escalation)
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
        # and read_only_intent exempts genuine explain-only asks.
        _intent = classify_intent(user_text)
        action_task = self.mode != "plan" and _intent["action"]
        read_only_intent = _intent["read_only"]
        for step in range(self.max_steps):
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
                step, self.max_steps, made_edit, unverified_edit, landing_nudges)
            if land:
                landing_nudges += 1
                log.info("LANDING nudge at step %d (remaining=%d, made_edit=%s, "
                         "unverified_edit=%s)", step, self.max_steps - step,
                         made_edit, unverified_edit)
                self.messages.append({"role": "tool", "name": "edit", "content": land})
            # The engine diffs this full render against its live KV cache and prefills
            # only the appended tokens, so a plain re-render IS the cache-extension path
            # (truncated-turn divergence is handled inside engine._sync_to, not here).
            prompt_ids = self._render()
            _pre_compact_len = len(prompt_ids)
            prompt_ids = self._compact_if_needed(prompt_ids)
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
            stop_condition = None
            think_cap = None
            if self.think_budget and self.thinking:
                stuck = (think_cap_hits + self._loop_nudges + thrash_nudges
                         + verify_nudges)
                think_cap = guardrails.think_budget(stuck, base=self.think_budget)

                def stop_condition(text_so_far, n, _cap=think_cap):
                    # Still inside <think> (no close emitted) and over budget -> stop; the
                    # turn is then force-closed by close_unclosed_think, a prefix-safe append.
                    return n >= _cap and "</think>" not in text_so_far

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

            text, stats = self.engine.generate(
                prompt_ids, max_tokens=self.max_gen_tokens, on_token=on_token,
                should_stop=self._should_stop, on_prefill=on_prefill,
                on_prefill_progress=on_prefill_progress, stop_condition=stop_condition)
            if gen_count[0] % 16:  # flush the final (un-throttled) count for the ↓ readout
                self._emit("gen", str(gen_count[0]))
            # The generation ran to the token cap: the turn was cut off, not finished.
            # A truncated assistant turn is NOT a final answer (and on Ornith's
            # non-trimmable cache its decoded text won't re-tokenize identically, so it
            # also forces the next-step re-prefill). Tracked so the no-call branch can
            # tell "truncated mid-thought" apart from "deliberately answered."
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
            # strip any trailing special tokens the template will re-add
            text = text.replace("<|im_end|>", "").rstrip()

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
            if stats.stop_condition_fired:
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
                })
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
                kind, nudge = guardrails.nudge_for_no_calls(
                    text, hit_cap, made_edit, unverified_edit, read_only_intent,
                    action_task, truncation_nudges, answer_nudges, verify_nudges,
                    _has_open_tool_call(text))
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
                            "tool calls without making progress.]")
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
                    result = self._run_subagent(
                        args.get("description", ""), args.get("prompt", ""),
                        args.get("tools", "read-only"))
                    render_tool_result(self._emit, name, args, result)
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
                    try:
                        result = fn(args, self._should_stop)
                        if plan_write and result.startswith("[wrote"):
                            self.last_plan_path = os.path.abspath(args["path"])
                    except Exception as e:  # noqa: BLE001 - surface tool errors to model
                        result = f"[tool error: {type(e).__name__}: {e}]"
                render_tool_result(self._emit, name, args, result)
                log.info("TOOL %s(%s) -> %s", name, args_preview(args),
                         result_preview(result))
                self.messages.append({"role": "tool", "name": name, "content": result})
                # Update the guardrail bookkeeping flags (did_work / made_edit /
                # unverified_edit) for this tool result — see guardrails.update_work_flags.
                did_work, made_edit, unverified_edit = guardrails.update_work_flags(
                    name, args, result, did_work, made_edit, unverified_edit)
                # Thrash counter (C): count back-to-back failed bash with no edit between.
                consecutive_failed_bash = guardrails.update_thrash(
                    name, result, consecutive_failed_bash)

            # Governor progress (plan 040): a fresh edit LANDED, or a pending edit got
            # VERIFIED (unverified_edit cleared by a clean bash) this step — real forward
            # motion, so mark the current budget band as having made progress (resets the
            # checkpoint that would otherwise fire the governor).
            if (made_edit and not _gov_prev_made) or (_gov_prev_unverified and not unverified_edit):
                gov_progress = True

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
        log.info("END: hit max tool steps (%d) | did_work=%s unverified_edit=%s",
                 self.max_steps, did_work, unverified_edit)
        return "[stopped: hit max tool steps]"


def repl(engine: BaseEngine, yolo: bool, ctx_limit: int = 24000, resume: list = None,
         thinking: bool = True):
    agent = Agent(engine, yolo=yolo, ctx_limit=ctx_limit, thinking=thinking,
                  resume=resume, persist=True)
    print(f"{C_DIM}chad ready. model={engine.model_id}"
          f"{' + draft' if getattr(engine, 'draft', None) else ''}. "
          f"type a task, or /reset, /exit.{C_RST}")
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
                          persist=True)
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
