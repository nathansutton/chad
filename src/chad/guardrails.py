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
