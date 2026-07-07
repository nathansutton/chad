"""Guardrail decision predicates for run_turn (extracted from agent.py).

These are the heuristics that separate "stops cleanly" from "loops forever / declares
false success": the loop guard, the verify-before-done / empty-done gating, the
tool-result bookkeeping (did_work / made_edit / unverified_edit), and the no-tool-call
nudge selection. They were previously inline in the 290-line run_turn loop and only
exercised by the slow model-backed eval suite.

Each function here is pure (or pure-ish: it returns a decision, the caller still owns
the counters and the message appends), so run_turn calls them in place of the old
inline boolean/branch expressions WITHOUT changing control flow or ordering — and they
can be unit-tested directly (see test_agent_guards.py). Thresholds, branch order, and
nudge text are byte-identical to the old inline code.
"""

import json
import re

# Catastrophic, near-never-intentional shell shapes. chad runs bash on the user's
# machine and reads UNTRUSTED repo files (`read`, `@mentions`) whose contents can
# drive a tool call — so in --yolo/auto mode a prompt-injected `rm -rf ~` would
# otherwise execute with no human in the loop. This denylist is deliberately tiny
# and high-confidence (recursive force-deletes of roots/home, raw disk/filesystem
# writes, fork bombs, curl|sh pipe-to-shell) so it almost never fires on real dev
# work; when it does, auto mode forces a confirm (or blocks, headless). Set
# CHAD_NO_DESTRUCTIVE_GUARD=1 to disable. NOT a security boundary — a sandbox is —
# just a seatbelt against the obvious catastrophe.
_DESTRUCTIVE_BASH = (
    # recursive force-rm whose target is rooted at / or ~ or $HOME (any depth), or
    # is a bare `*` / `.` (whole-cwd). A relative path like `build/` or `./out` is
    # NOT matched — only roots and home subtrees, where an injected delete is fatal.
    re.compile(r"\brm\s+(?:-\w+\s+)*-\w*[rR]\w*\s+(?:-\w+\s+)*(?:[/~]\S*|\$HOME\S*|[.*](?:\s|$))"),
    re.compile(r"\b(mkfs|fdisk|parted)\b"),
    re.compile(r"\bdd\b[^\n]*\bof=/dev/"),
    re.compile(r">\s*/dev/(sd|disk|nvme|hd)"),
    re.compile(r":\(\)\s*\{.*\|.*&.*\}"),       # fork bomb :(){ :|:& };:
    re.compile(r"\b(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(ba)?sh\b"),  # curl … | sh
)


def is_destructive_bash(command: str) -> bool:
    """True if a bash command matches the catastrophic denylist (see _DESTRUCTIVE_BASH).
    Pure and testable. Caller decides what to do (force-confirm / block)."""
    return any(p.search(command) for p in _DESTRUCTIVE_BASH)


def bash_result_verifies(result: str) -> bool:
    """A bash tool result clears the unverified-edit flag only on a clean run.

    A clean exit-0 command (including one with no output, `[no output]`) or any
    real stdout counts as a verification. The four `[`-prefixed sentinels below
    all mean the check did NOT actually pass, so they must NOT clear the flag:
    a non-zero exit, a timeout, a user interrupt (ctrl-c), or a launch failure.
    """
    return not result.startswith(
        ("[exit", "[timed out", "[interrupted", "[failed to launch"))


# Tools that count as real work (did_work) — includes the symbolic read/search/edit
# tools, but NOT planning/done. Kept as a named constant so the set is testable.
SUBSTANTIVE_TOOLS = ("grep", "glob", "read", "write", "edit", "bash",
                     "repo_map", "overview", "view_symbol", "find_symbol",
                     "find_refs", "replace_symbol", "insert_symbol", "rename_symbol")


def update_work_flags(name, args, result, did_work, made_edit, unverified_edit):
    """Update the (did_work, made_edit, unverified_edit) guardrail flags after one
    tool result; returns the new triple. Byte-identical to run_turn's old inline
    bookkeeping: a substantive tool counts as real work; a successful edit/write
    (text or symbolic) sets made_edit and — unless it's a pure prose/doc file —
    arms unverified_edit; a bash run that didn't error/timeout/interrupt clears it
    (a failing test keeps it dirty so the model re-runs)."""
    if name in SUBSTANTIVE_TOOLS:
        did_work = True
    landed_edit = (
        (name in ("write", "edit") and result.startswith(("[wrote", "[edited")))
        or (name in ("replace_symbol", "insert_symbol")
            and result.startswith(("[replaced", "[inserted")))
        or (name == "rename_symbol" and result.startswith("[renamed")))
    if landed_edit:
        made_edit = True
        # A pure prose/doc file (README, CLAUDE.md, notes) has nothing to "run",
        # so don't arm the verify-before-done nudge for it — otherwise /init and
        # doc edits waste steps being told to run a test that doesn't exist.
        # Code/config edits still require verification.
        is_doc = str(args.get("path", "")).lower().endswith(
            (".md", ".markdown", ".rst", ".txt"))
        if not is_doc:
            unverified_edit = True
    elif name == "bash" and bash_result_verifies(result):
        unverified_edit = False
    return did_work, made_edit, unverified_edit


def done_rejection(did_work, unverified_edit, empty_done_nudges, verify_nudges):
    """Whether a `done` should be rejected, in run_turn's order. Returns 'empty'
    (no real work yet — the markdown-code-fence failure mode), 'verify' (files
    changed but nothing run to verify them), or None (accept). The caller bumps the
    matching counter and appends the corresponding nudge."""
    if not did_work and empty_done_nudges < 2:
        return "empty"
    if unverified_edit and verify_nudges < 2:
        return "verify"
    return None


def nudge_for_no_calls(text, hit_cap, made_edit, unverified_edit, read_only_intent,
                       action_task, truncation_nudges, answer_nudges, verify_nudges,
                       open_tool_call):
    """Pick the nudge for a step that produced NO tool call, in the original priority
    order: (1) TRUNCATED — hit the token cap mid-thought, so it isn't an answer;
    (2) ANSWERED ON PAPER — produced/described code but never applied it; (3) UNVERIFIED
    EDIT — edited but never ran the check. Pure: returns (kind, nudge_text) or
    (None, None); the caller bumps the matching counter and appends the nudge. `kind`
    is one of 'truncated' / 'no-edit' / 'unverified-edit'. Byte-identical to the old
    inline branch (open_tool_call is run_turn's _has_open_tool_call(text))."""
    has_code = "```" in text
    if hit_cap and truncation_nudges < 2:
        if open_tool_call:
            # Cut off mid tool-call — almost always a `write` whose content
            # exceeded the token budget. Retrying it whole just truncates
            # again; guide it to land the file in bounded pieces.
            nudge = ("[your tool call was cut off at the length limit — the "
                     "content was too long to emit in one call. Do NOT retry it "
                     "whole. Create the file with `write` using only the FIRST "
                     "portion of the content, then append the rest with one or "
                     "more `edit` calls. Emit one complete tool call at a time.]")
        else:
            nudge = ("[your reply was cut off at the length limit before you "
                     "called any tool. Do NOT re-paste what you already wrote. "
                     "Take the next single concrete action now as a real "
                     "<tool_call> — e.g. write the file — one tool at a time.]")
        return "truncated", nudge
    if (not read_only_intent) and (action_task or has_code) \
            and not made_edit and answer_nudges < 2:
        nudge = ("[you described the change but did not apply it — markdown code "
                 "blocks are NOT executed and code in your reply does NOT touch "
                 "any file. To CREATE a new file use the `write` tool (path + full "
                 "content); to change an existing one use `edit`/`replace_symbol`. "
                 "Then run it with bash and call done. Do not answer with code in "
                 "prose.]")
        return "no-edit", nudge
    if unverified_edit and verify_nudges < 2:
        nudge = ("[not finished: you edited the file but the check has not passed. "
                 "Run the project's check/tests with bash; if it fails, read the "
                 "error, fix the code, and re-run. Don't stop until it passes, "
                 "then call done.]")
        return "unverified-edit", nudge
    return None, None


def landing_nudge(step, max_steps, made_edit, unverified_edit, landing_nudges):
    """Near the step cap with the task not cleanly landed, push the model to stop
    exploring and commit its highest-value edit before the hard cut-off. Fires at most
    once. Without this the run dies silently at max_steps with whatever it had — the
    demonstrated failure: 40 steps of environment/import probing, zero edits applied.
    'Cleanly landed' = an edit was made and has since been verified. Returns the nudge
    text (or None); the caller bumps landing_nudges and appends it as a tool message."""
    if landing_nudges >= 1:
        return None
    remaining = max_steps - step
    if remaining > 3:  # only inside the last 3 steps
        return None
    if made_edit and not unverified_edit:  # already landed and verified — let it finish
        return None
    if not made_edit:
        return (f"[only {remaining} step(s) left before this turn is force-stopped, and "
                "you have not applied a single edit yet. STOP exploring and verifying the "
                "environment. Make your highest-value edit now with edit/write/"
                "replace_symbol, verify it, and call done.]")
    return (f"[only {remaining} step(s) left before this turn is force-stopped. You "
            "edited but never ran the check. Run the project's tests now, fix the code "
            "if it fails, then call done — do not start any new exploration.]")


STEP_CAP_CEILING = 4  # absolute per-turn step ceiling = STEP_CAP_CEILING * max_steps


def extend_step_cap(step_cap, base_max_steps, landed_in_window, hard_ceiling):
    """At the step cap with the turn still landing AND verifying changes, grant another
    window instead of killing productive work (the demonstrated failure: a plan-
    implementation turn with steady verified edits force-stopped dead at step 40, an
    edit half-applied). Mirrors the governor's philosophy — never interrupt landed+
    verified progress — which the fixed cap used to override. Extends by half the base
    cap, never past hard_ceiling; a window that landed nothing verified does NOT extend
    (the turn ends and banks a progress note). The governor's token/wall budget remains
    the runaway backstop. Returns the new cap, or None to stop. Pure/testable."""
    if not landed_in_window:
        return None
    new_cap = min(step_cap + max(1, base_max_steps // 2), hard_ceiling)
    return new_cap if new_cap > step_cap else None


def update_thrash(name, result, consecutive_failed_bash):
    """Track a run of bash commands that errored back-to-back with no edit between them
    — the environment-thrash / flailing-probe signature (repeatedly guessing the test
    runner, or `python -c "import X"` checks that keep exiting non-zero). A failed bash
    increments the run; ANY landed edit OR a clean bash resets it; other tools leave it
    untouched (an interleaved read/grep is normal investigation). Returns the new count."""
    if name == "bash":
        return 0 if bash_result_verifies(result) else consecutive_failed_bash + 1
    if name in ("write", "edit", "replace_symbol", "insert_symbol", "rename_symbol"):
        return 0
    return consecutive_failed_bash


def bash_thrash_nudge(consecutive_failed_bash, thrash_nudges):
    """After 4+ consecutive failed bash commands with no edit, nudge (bounded to 2) to
    break the probe loop. The exact-call loop guard misses this because each failing
    command differs by a few chars (`from acr import A` / `from inflo import B`), so its
    signatures never repeat. Returns the nudge text or None."""
    if consecutive_failed_bash >= 4 and thrash_nudges < 2:
        return ("[several commands in a row have failed and you have not edited any file. "
                "Stop probing the environment. If you cannot run the tests, make your "
                "planned edit anyway and verify at the end; to check whether a symbol "
                "exists use find_symbol/overview, not `python -c import`. Take a different "
                "action now.]")
    return None


# --- soft think-cap (plan 039) -----------------------------------------------------
# <think> blocks are 36–41% of all generated tokens on the eval suite and decode is
# bandwidth-bound, so an unbounded reasoning run is the single largest wall-clock
# multiplier the harness controls. When armed, run_turn stops a step's <think> run once
# it exceeds this cap and force-closes the block (prefix-safe — see
# agent.close_unclosed_think), then continues. The cap ESCALATES with a turn's
# stuck-signals so a genuinely hard step gets more reasoning room instead of being
# chunked repeatedly into re-thinks.
THINK_CAP_RAMP = (1024, 2048, 4096)


def think_budget(stuck_level: int, base: int = 512) -> int:
    """Per-step <think>-token cap. `stuck_level` 0 => `base` (the cheap default); each
    increment climbs THINK_CAP_RAMP (clamped to the top), giving more reasoning room when
    run_turn has a concrete stuck-signal for this turn — a prior cap hit, or a loop /
    thrash / verify-fail nudge. Never returns below `base` (so a caller that sets a large
    base is respected). Pure and testable; run_turn owns `stuck_level` and `base`."""
    if stuck_level <= 0:
        return base
    return max(base, THINK_CAP_RAMP[min(stuck_level - 1, len(THINK_CAP_RAMP) - 1)])


# --- degenerate-repetition stop ------------------------------------------------------
# Greedy decode on a small quantized model can fall into a literal repetition loop: the
# dogfood traces caught a step that wrote a complete, correct answer in ~600 chars and
# then repeated "`CHAD_NO_TASK`, " ~1,300 times until the 8192-token cap — 225s of dead
# generation, twice in one task (blowing its timeout). Unlike the think-budget (an
# opt-in capability trade, plan 039), this guard only fires on output that is already
# garbage, so it is ON by default (CHAD_NO_REPEAT_GUARD=1 disables). run_turn checks the
# generation's tail every few tokens; on a hit it stops the step (prefix-safe — the
# generated tokens are already in the KV cache) and nudges the model out of the loop.
REPEAT_TAIL_CHARS = 2048    # window that must be fully periodic — long enough that no
                            # legitimate prose/code run trips it, short enough to fire
                            # a few hundred tokens into a runaway, not 8k tokens in
REPEAT_MAX_PERIOD = 256     # unit ≤ this ⇒ the window holds ≥ 8 repeats


def degenerate_tail(text: str, tail: int = REPEAT_TAIL_CHARS,
                    max_period: int = REPEAT_MAX_PERIOD) -> bool:
    """True when the last `tail` chars of `text` are one short unit repeated end-to-end
    — the degenerate-decode signature. The smallest period comes from the KMP prefix
    function (p = len - f[-1] ⇒ s[i] == s[i-p] for all i ≥ p, regardless of where the
    window cuts into the unit — a plain `(s+s).find(s)` doubling test would only catch
    periods that divide the window exactly). ~0.5ms of pure Python over a 2KB window;
    run_turn calls it every 16 tokens, i.e. a few times per second against a ~25ms/token
    decode, so the cost is noise."""
    if len(text) < tail:
        return False
    s = text[-tail:]
    f = [0] * len(s)  # KMP prefix function (longest proper border of s[:i+1])
    k = 0
    for i in range(1, len(s)):
        while k and s[i] != s[k]:
            k = f[k - 1]
        if s[i] == s[k]:
            k += 1
        f[i] = k
    return len(s) - f[-1] <= max_period


REPEAT_STOP_NUDGE = (
    "[your output degenerated into repeating the same text over and over; it was cut "
    "off. Do not continue that repetition. Give your final answer concisely, or make "
    "the next tool call, now.]")


def repeat_stop_abort(repeat_stops: int) -> bool:
    """Abort the turn after the 3rd repetition cut-off — the nudge isn't breaking the
    decode loop, and each retry costs another stall. Mirrors loop_should_abort."""
    return repeat_stops >= 3


# --- runaway-turn governor (plan 040) ----------------------------------------------
# chad's dominant failure mode is timeout, not wrong answers: on the polyglot sweep a
# PASSING task burns 14–35k prefill tokens; a FAILING one balloons to 130–187k before
# dying at the wall. Grinding a turn that's already 100k-prefill deep with no green test
# almost never converges — the cheapest good outcome is to STOP, bank what was learned,
# and (optionally) relaunch fresh. The existing guards are all *local* (repeat-call
# loop, consecutive failed bash, landing nudge near max_steps); this one watches the
# *global* trajectory: budget consumed vs progress made.
#
# It's a pure checkpoint state machine. run_turn tracks the cumulative prefill tokens
# (self.prefill_tokens) + wall clock and a per-band "did real work land+verify" signal,
# and consults turn_governor at each budget-fraction checkpoint. Soft = one strong nudge
# at ~50%; hard = end the turn with a deterministic progress note at ~80%. Because real
# work resets the checkpoint (progress=True => never fire), a genuinely slow-but-working
# turn is never interrupted — only the pathological no-progress tail binds.
GOV_SOFT_FRAC = 0.5   # first checkpoint: nudge if no progress yet
GOV_HARD_FRAC = 0.8   # second checkpoint: bank a note and end the turn
BUDGET_SENTINEL = "[budget]"  # run_turn return prefix on a hard governor stop

GOVERNOR_SOFT_NUDGE = (
    "[you have consumed half of this turn's budget without landing AND verifying a "
    "single change. Stop exploring. State your current single best hypothesis in one "
    "sentence, then act on it directly: make the edit and run the check. Do NOT re-read "
    "files you have already read or re-run commands you have already run.]")


def budget_fraction(tokens, token_budget, wall_s=0.0, wall_budget_s=None) -> float:
    """Fraction of the turn budget consumed = the max of the token-budget ratio and the
    wall-clock ratio (whichever is tighter drives the governor). A budget that is falsy
    (None/0) is ignored; if neither is set, returns 0.0 so the governor never fires (the
    off state). Pure and testable."""
    fracs = []
    if token_budget:
        fracs.append(tokens / token_budget)
    if wall_budget_s:
        fracs.append(wall_s / wall_budget_s)
    return max(fracs) if fracs else 0.0


def budget_band(frac: float) -> int:
    """Which checkpoint band a consumed-fraction falls in: 0 (below the soft mark),
    1 (soft..hard), 2 (at/over the hard mark). run_turn fires the governor only when the
    band *advances*, evaluating the just-completed band's progress."""
    if frac >= GOV_HARD_FRAC:
        return 2
    if frac >= GOV_SOFT_FRAC:
        return 1
    return 0


def turn_governor(band, progress, soft_fired, *, disabled=False):
    """Decision for a checkpoint the turn just crossed into. `band` is the band being
    entered (1 = soft ~50%, 2 = hard ~80%); `progress` is whether a change landed AND was
    verified during the band we're leaving; `soft_fired` whether the soft nudge already
    went out this turn. Returns 'hard' (end + bank a note), 'soft' (one nudge), or None.
    Real progress in the completed band resets the checkpoint — a slow-but-working turn is
    never interrupted. `disabled` (CHAD_NO_GOVERNOR) always returns None. Pure/testable."""
    if disabled or progress:
        return None
    if band >= 2:
        return "hard"
    if band == 1 and not soft_fired:
        return "soft"
    return None


def advance_governor(gov_band, new_band, progress, soft_fired):
    """Walk the governor across every budget checkpoint crossed in a SINGLE step
    (`gov_band` -> `new_band`). Returns `(decision, gov_band, progress)`: the first non-None
    `turn_governor` result (or None), the updated band, and the carried-over progress flag.

    A single step can leap two bands at once (e.g. a large re-prefill that consumes ~30%+ of
    the budget in one go, jumping 0 -> 2). The earned `progress` is credited to EVERY band
    crossed in this one step — you can't re-earn progress mid-jump, so a step that genuinely
    landed+verified a change must not be hard-stopped just because it also spanned two bands
    (plan 052). Progress is consumed (reset to False) once any band is crossed, so the next
    band must re-earn it; a step that crosses nothing leaves the flag untouched. Pure/testable."""
    decision = None
    crossed = gov_band < new_band
    while gov_band < new_band:
        decision = turn_governor(gov_band + 1, progress, soft_fired)
        gov_band += 1
        if decision:
            break
    if crossed:
        progress = False
    return decision, gov_band, progress


# Tool results that landed a file change / were run, used to reconstruct a progress note
# deterministically (no model call) from the transcript.
_EDIT_TOOLS = ("write", "edit", "replace_symbol", "insert_symbol", "rename_symbol")
_ERROR_PREFIXES = ("[exit", "[timed out", "[failed to launch", "[tool error", "[denied")


def progress_note(messages, max_lines: int = 20) -> str:
    """Synthesize a ≤`max_lines` progress note from the transcript with NO model call, so
    a hard-stopped turn can seed a fresh relaunch (sheds the ramble AND the huge prefill
    the stuck model was dragging around). Deterministic: pulls files edited and commands
    run from the assistant turns' own <tool_call> blocks (via parse_tool_calls), plus the
    last error seen in a tool result. Prefer facts the executor cannot reconstruct from a
    clean context — what was already tried and what failed last."""
    from .toolcall_parse import parse_tool_calls
    edited, commands = [], []
    last_error = None
    for m in messages:
        role = m.get("role")
        content = m.get("content", "") or ""
        if role == "assistant":
            for name, args in parse_tool_calls(content):
                if name == "bash":
                    cmd = str(args.get("command", "")).strip()
                    if cmd and cmd not in commands:
                        commands.append(cmd)
                elif name in _EDIT_TOOLS:
                    p = str(args.get("path", "")).strip()
                    if p and p not in edited:
                        edited.append(p)
        elif role == "tool":
            if content.startswith(_ERROR_PREFIXES) or "Traceback" in content:
                last_error = content
    lines = ["Progress so far (auto-summarized — the previous attempt ran out of budget):"]
    if edited:
        lines.append("Files already edited: " + ", ".join(edited[-8:]))
    if commands:
        lines.append("Commands already tried (do not blindly repeat):")
        lines += [f"  $ {c[:120]}" for c in commands[-4:]]
    if last_error:
        lines.append("Last error seen:")
        lines += ["  " + ln[:120] for ln in last_error.strip().splitlines()[-4:]]
    if len(lines) == 1:
        lines.append("(no edits, commands, or errors were recorded before the budget ran out)")
    return "\n".join(lines[:max_lines])


def loop_signature(calls) -> str:
    """Canonical signature of a tool-call set, for the repeated-call loop guard."""
    return json.dumps(calls, sort_keys=True)


def is_repeat_loop(seen_before: int) -> bool:
    """True on the 3rd+ identical call-set (seen_before is the count BEFORE this
    occurrence was recorded, so >=2 means this is at least the third)."""
    return seen_before >= 2


def loop_should_abort(loop_nudges: int) -> bool:
    """After incrementing the loop-nudge counter, more than 2 nudges -> abort."""
    return loop_nudges > 2
