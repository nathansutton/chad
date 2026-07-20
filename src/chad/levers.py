"""Named harness levers, each individually disableable at runtime.

A harness change that ships hardcoded can only be attributed by a full sweep with the
code reverted. That is the trap the iter-1/2/3 bundles fell into: fourteen fixes land as
three diffs, the rate moves, and nothing tells you which fix moved it — or whether one of
them is a regression the others are masking. Leave-one-out ablation is the cure, and it
needs exactly one thing from the harness: a way to switch a single behavior off without
touching the tree.

So every behavioral lever gets a name here and a `levers.enabled("name")` guard at its
site. `CHAD_DISABLE=a,b` turns off a and b; the private ablation harness enumerates this
registry and runs the slice once per lever to produce a per-lever delta.

Unknown names in `CHAD_DISABLE` are a hard error, not a warning. A typo'd lever in an
ablation script would otherwise run the *unmodified* harness and report a delta of zero,
which reads as "this lever does nothing" — the single most expensive way for this
machinery to lie to you. `validate_env()` runs it at startup, not at first use.

Levers default ON: the registry describes what can be switched off, not what is off.

Two fields carry the judgment a bare on/off switch can't:

`group` — the harness iteration that introduced the change, so `ablate.py --group iter3`
can price one bundle without paying for the others.

`kind` — whether the OFF state is a state you could ever ship.
  BEHAVIOR       the change adds a behavior; without it the agent is merely less helped.
  REGRESSION_GUARD  the change fixes a demonstrated bug, and OFF restores that bug. A
                 grep that reports "[no matches]" on a tree it never finished walking is
                 a lie, not a configuration. These exist to be *measured*, never shipped
                 off, and `--levers` says so out loud so nobody wires one into a preset.
"""

import os
from dataclasses import dataclass

BEHAVIOR = "behavior"
REGRESSION_GUARD = "regression-guard"


@dataclass(frozen=True)
class Lever:
    description: str
    group: str
    kind: str = BEHAVIOR


# Keep each description accurate enough that `chad --levers` is a readable inventory of
# what the harness does beyond a bare model call. It is the artifact a reviewer reads to
# judge whether a benchmark number came from the model or the scaffolding.
LEVERS: dict[str, Lever] = {
    # --- iter-2: implemented from the night-7 forensics; only ever measured on the 7
    #     tasks it was derived from, never on the other 23. ------------------------
    "verify_requires_execution": Lever(
        "A bash result only clears the unverified-edit flag if the command actually ran "
        "code: reject trivial syntax/compile/version probes and display-only commands. "
        "OFF restores the spoofable verify gate (`sed -n | cat -A` exiting 0 disarmed "
        "every guard and shipped an IndentationError).",
        "iter2", REGRESSION_GUARD),
    "bail_nudge": Lever(
        "Nudge a turn that produced no tool call and no real content — empty after "
        "</think>, or a preamble that only announces a next step. Backend-agnostic.",
        "iter2"),
    "investigation_gate": Lever(
        "After ~6 read-only steps with no landed edit, steer the model to act before the "
        "step cap kills the turn with an empty patch.",
        "iter2"),
    "edit_loop_break": Lever(
        "After 2 consecutive edits that failed to land, stop the model re-trying "
        "variations and tell it to read the real lines / replace the whole symbol.",
        "iter2"),
    "subagent_no_respawn": Lever(
        "Refuse to re-spawn a sub-agent with an identical (description, prompt) after one "
        "capped out — it will cap out again and burn the parent's budget.",
        "iter2"),
    "subagent_budget_note": Lever(
        "A sub-agent that hits its step cap returns its progress note instead of "
        "discarding it (it used to answer '1 line' after reading the gold file).",
        "iter2"),
    "grep_zero_match_notice": Lever(
        "grep states its searched scope and never hides truncation on the zero-match "
        "path. OFF restores the bare '[no matches]' LIE on a tree grep never finished "
        "walking — the root trigger of the django-14007 failure.",
        "iter2", REGRESSION_GUARD),
    "syntaxgate_revert": Lever(
        "An edit that introduces an indentation/syntax error is REVERTED, not merely "
        "warned about. OFF restores the wrong-indent edit death loop.",
        "iter2", REGRESSION_GUARD),
    "structural_reindent": Lever(
        "When a replace_lines/insert_lines splice would break Python indentation, "
        "recompute the block's indentation from its own syntax (colons/dedents), "
        "ignoring the model's whitespace, and take it if the file then parses. OFF "
        "falls back to fit+snap, which can't fix a multi-LEVEL block a weak model "
        "mis-indents (the observed 9B failure).",
        "iter5"),

    # --- iter-3: implemented straight off the iter-2 traces. NEVER MEASURED. -------
    "progress_note_rich": Lever(
        "The relaunch progress note carries the model's last working hypothesis, the "
        "failing-check signature, and the files already examined — not just file names. "
        "OFF restores the amnesic note that made a relaunch re-derive the investigation.",
        "iter3"),
    "grep_filter_before_cap": Lever(
        "tool_grep applies the dir/skip/isfile filter BEFORE the file cap, so a deep "
        "locale tree can't exhaust the budget before the source tree is reached. OFF "
        "restores a grep that reports zero matches for files it never looked at.",
        "iter3", REGRESSION_GUARD),
    "repeat_coarse_tier": Lever(
        "Degenerate-decode detection at block scale (12KB window / 3072-char period), "
        "catching the paragraph-sized reasoning loop the 256-char fine tier is blind to.",
        "iter3"),
    "edit_fail_kind": Lever(
        "Distinguish a no-op edit (old == new) from an unmatched `old` and prescribe the "
        "right remedy. OFF tells a model that pasted identical text to go re-read and "
        "paste it verbatim — which is exactly what just failed.",
        "iter3"),
    "revert_rearm_gate": Lever(
        "A clean `git checkout/reset --hard/stash/clean -f` un-sets made_edit, re-arming "
        "the no-empty-diff gate. OFF lets edit→test→revert→prose ship an empty diff.",
        "iter3", REGRESSION_GUARD),
    "done_spec_recheck": Lever(
        "Before ACCEPTING the first `done` of a task turn, inject one message telling the "
        "model to re-read the task and confirm every required deliverable exists at the "
        "exact path/format the task asked for (check with ls/cat/test), fixing anything "
        "missing or wrong first. Fires once per turn. Targets the 'declared victory with "
        "budget to spare, wrong output format' losses (TB2 sam-cell-seg wrote directories "
        "where files were required; bn-fit-modify 8/9). OFF accepts the first done as-is.",
        "iter3"),

    # --- uncommitted at the time of writing; likewise unmeasured. ------------------
    "subagent_compact_window": Lever(
        "Compaction's drop-oldest pass bounds by a trailing window rather than the last "
        "user query, so it is not dead code in a sub-agent (whose only user message is "
        "at index 1). OFF makes a sub-agent fall through to the pass that shreds the "
        "search results it just gathered.",
        "iter3", REGRESSION_GUARD),
    "backend_retry": Lever(
        "Re-issue a generation once on a transient backend error (5xx / mid-stream error "
        "chunk). Safe because the prompt is rebuilt from `messages` each step.",
        "iter3"),

    # --- iter-6: line-addressed edits corrupting multi-line structures;
    #     derived from the measured 9B/35B dogfood (10 ignored parse warnings, LOOP
    #     ABORT with a severed def signature). ---------------------------------------
    "syntax_revert": Lever(
        "An edit that turns a cleanly-parsing Python file into ANY SyntaxError is "
        "REVERTED (generalizes syntaxgate_revert beyond IndentationError), with the "
        "reject naming the severed multi-line statement and echoing the current lines. "
        "Applies to edit/replace_lines/insert_lines/replace_symbol/insert_symbol "
        "(non-Python languages and whole-file write are gated separately: "
        "ts_edit_revert / write_gate, iter8). OFF restores warn-and-land: the 073 "
        "dogfood landed ~10 corrupting line edits over ignored parse warnings and "
        "aborted with the file broken.",
        "iter6", REGRESSION_GUARD),
    "edit_result_echo": Lever(
        "replace_lines/insert_lines results echo the changed region with its POST-edit "
        "line numbers plus a shift note, so follow-up edits re-anchor on numbers the "
        "model has actually seen instead of reusing the ones from a pre-edit read (the "
        "073 stale-number failure that conflated two adjacent defs).",
        "iter6"),
    "stale_file_guard": Lever(
        "replace_lines/insert_lines reject-once, with a fresh numbered view of the "
        "target region, when the file changed on disk since the model last saw it "
        "(bash/sed, git checkout, test runs) — line numbers minted against the old "
        "content are blind. The reject refreshes the anchor, so it never locks out.",
        "iter6"),

    # --- iter-7: parse-clean semantic drift in whole-unit rewrites,
    #     derived from the first successful post-073 dogfood (replace_symbol dropped
    #     an argparse line whose Namespace attr was still read → --agentic crashed,
    #     with no signal at edit time). ----------------------------------------------
    "edit_drift_warn": Lever(
        "After a Python edit lands, warn in the same result when it DROPPED a binding "
        "(def/class/param/assignment/self-attr) or a CLI-flag string that the rest of "
        "the file still uses — the parse-clean drift class the syntax gates can't see. "
        "Warn not reject: a reject would make legitimate remove-a-feature edits "
        "order-dependent. OFF restores silent drift (the measured --context-tokens "
        "AttributeError shipped without a word).",
        "iter7"),

    # --- iter-8: the validate-before-write choke point, from the lydia
    #     teardown + the two-corpus trace sweep (320 dogfood sessions / 304 benchmark
    #     trajectories): broken code LANDED 4x more often than it was rejected, 51/55
    #     benchmark landings came through warn-only write, and non-Python files had no
    #     revert at all (vm.js / ars.R compounded to reward-zero tasks). ---------------
    "ts_edit_revert": Lever(
        "Extends the 073 edit revert beyond Python: a targeted edit that takes a file "
        "with zero tree-sitter ERROR/MISSING nodes to one with any is REVERTED. Dirty "
        "baselines stay editable; no-grammar files are never blocked. OFF restores "
        "warn-and-land on non-Python code — the measured make-mips-interpreter vm.js "
        "break that six follow-up edits then compounded to a reward-zero task.",
        "iter8", REGRESSION_GUARD),
    "write_gate": Lever(
        "Whole-file write refuses content that would newly break the parse (existing "
        "clean file -> broken, or a new Python file that doesn't parse); an "
        "already-broken file stays overwritable as the repair path, and a new "
        "tree-sitter-language file only warns (grammar-quirk risk). OFF restores the "
        "warn-only write that delivered 51 of the 55 landed syntax breaks in the "
        "benchmark trace sweep.",
        "iter8", REGRESSION_GUARD),
    "broken_streak_steer": Lever(
        "After 2+ consecutive landed mutations that leave a Python file unparseable "
        "(the sanctioned already-broken path), the parse warning escalates to a "
        "whole-file-rewrite / restore-known-good steer. The 079 dogfood sweep measured "
        "a 14-landing broken streak that plain per-edit warnings never interrupted.",
        "iter8"),
    "write_diff_note": Lever(
        "An overwrite's result carries '(+a -d lines vs previous)' so the model can "
        "check the change's size against its intent — the cheap core of the diff-echo "
        "idea; targets the lost-track-of-file-state no-op/loop episodes.",
        "iter8"),

    # --- (TB2 deadline awareness): the adapter now passes the wall budget
    #     down (--turn-budget-s = cap-60), so the governor arms on every TB2 run. This
    #     lever is the wrap-up NUDGE that rides on top of it. -------------------------
    "wrapup_window": Lever(
        "One-shot wall-clock steering note in the turn's final stretch (remaining <= "
        "max(120s, 15% of the wall budget)): stop exploring and land your best answer "
        "before the force-stop. Distinct from the governor's no-progress hard-stop — this "
        "fires even on a productive turn so it commits a scored partial instead of being "
        "SIGKILLed mid-edit. Only active when a wall budget is configured; off with the "
        "governor (CHAD_NO_GOVERNOR).",
        "iter9"),

    # --- (TB2 think-spiral salvage): close-and-continue force-closes a runaway
    #     <think> at CHAD_THINK_CEILING and keeps decoding the action in-step (the engine
    #     mechanism, env-gated, has no lever — off by default like CHAD_THINK_BUDGET). This
    #     lever is the ESCALATION on top of it. --------------------------------------
    "no_think_escalation": Lever(
        "After 2 consecutive steps that hit the gen cap (or were salvaged) yet produced no "
        "tool call — the think-spiral signature — run the next step with <think> disabled "
        "to force an action, then restore. Mechanical 'act now', replacing a third prose "
        "nudge that demonstrably does not land. Only active when the close-and-continue "
        "ceiling is armed (CHAD_THINK_CEILING), so default chad is unaffected.",
        "iter9"),

    # --- (TB2.1 done-audit): the n=1 autopsy's largest fail bucket (20/43)
    #     was dones claiming SPECIFIC verification the hidden checker rejected — the
    #     model verifies a WEAKER predicate than the task's own wording, with huge
    #     unused budget (kv-store-grpc done at 84s of 900), and done_spec_recheck was
    #     ON for every one of them. -----------------------------------------------
    "done_audit": Lever(
        "Bounce a would-otherwise-be-accepted `done` ONCE per turn (turns that did real "
        "work, with wall runway to spare) with the task statement's own requirement "
        "lines quoted verbatim plus stat-level facts about every path the task names, "
        "then accept the next `done` unconditionally (the 070 anti-spiral latch). "
        "Supersedes the generic done_spec_recheck steer while enabled — stacking both "
        "would force two bounces per turn. OFF restores the done_spec_recheck path.",
        "iter10"),

    # --- (turn-level think budget): the 086 ceiling watches any ONE generation
    #     and is blind to a turn that death-by-a-thousand-cuts its way to a wall-death
    #     across many separate thinks under that ceiling (dna-assembly: 123k total across
    #     13 thinks, no single one >= 6k). This watches CUMULATIVE turn think-spend
    #     instead. Only active when a wall budget is configured (self._turn_budget_s) —
    #     off entirely in interactive/unmetered runs, like wrapup_window. -------------
    "turn_think_budget": Lever(
        "A cumulative per-turn reasoning-token budget (wall- and decode-speed-aware, "
        "clamped 8k-24k): a one-shot soft steer at half spent, then a no-think "
        "THROTTLE once spent — one forced no-think action step per 3k further think "
        "tokens (guardrails.turn_think_throttle), so thinking restores when the model "
        "stops over-spending; a blanket rest-of-turn mute regressed run1 passes with "
        "garbled no-think tails (plan 107). Acts only at step boundaries on fresh "
        "generations, never an in-flight one (086's lesson). Off in plan mode, for "
        "read-only-intent turns, and below a 300s wall budget (short relaunch tails "
        "belong to hard_wrapup). Only active with a wall budget configured.",
        "iter11"),

    # --- (103 hard wrap-up): the 085 wrapup_window nudge only lands at a step
    #     boundary and fired 3/89 (0 rescued) because the model is usually buried in one
    #     long generation when the window opens. This aborts that generation mid-stream. --
    "hard_wrapup": Lever(
        "At the wall deadline (within max(90s, 10% of the budget)) ABORT the in-flight "
        "generation mid-stream and force ONE time-boxed, no-think landing turn that writes "
        "each deliverable to its exact path before the harness SIGKILLs the task — the "
        "mid-generation backstop the 085 step-boundary nudge (3/89, 0 rescued) can't be. "
        "Only active with a wall budget configured; inert in interactive/unmetered runs.",
        "iter12"),

    # --- from the LangChain harness-tuning playbook. -------------------------------
    "compact_notice": Lever(
        "After compaction, inject an in-band message telling the model its context was "
        "trimmed and that earlier content must be re-read, not recalled.",
        "playbook"),
    "compact_offload": Lever(
        "On compaction, spill the untrimmed transcript to a file and give the model its "
        "path, so trimmed content is recoverable rather than lost.",
        "playbook"),
    "plan_review": Lever(
        "After the plan file is written in plan mode, inject a message asking the model "
        "to re-read and check it against a rubric before calling done.",
        "playbook"),
    "profile_prompt": Lever(
        "Append the active model profile's prompt block to the system prompt "
        "(model-specific accommodations; see profiles.py).",
        "playbook"),
}


class UnknownLever(ValueError):
    """A CHAD_DISABLE entry that names no registered lever."""


def _disabled() -> frozenset[str]:
    raw = os.environ.get("CHAD_DISABLE", "")
    names = frozenset(n.strip() for n in raw.split(",") if n.strip())
    unknown = names - LEVERS.keys()
    if unknown:
        raise UnknownLever(
            f"CHAD_DISABLE names unregistered lever(s): {sorted(unknown)}. "
            f"Known levers: {sorted(LEVERS)}")
    return names


def enabled(name: str) -> bool:
    """True unless `name` appears in CHAD_DISABLE. Raises if `name` is not registered —
    a guard on an unregistered lever is invisible to the ablation driver, so it must not
    silently pass."""
    if name not in LEVERS:
        raise UnknownLever(f"unregistered lever {name!r}; add it to levers.LEVERS")
    return name not in _disabled()


def validate_env() -> None:
    """Raise on a typo'd CHAD_DISABLE now, at startup, rather than at the first compaction
    — which on a long sweep is forty minutes of wall-clock into a run that is already
    measuring the wrong thing. Called from the CLI and the eval runner."""
    _disabled()


def active() -> list[str]:
    """Registered levers currently enabled, sorted. Recorded in eval rows so a banked
    result carries the harness configuration that produced it."""
    off = _disabled()
    return sorted(n for n in LEVERS if n not in off)


def groups() -> list[str]:
    """Harness iterations present in the registry, in registration order."""
    seen = []
    for lv in LEVERS.values():
        if lv.group not in seen:
            seen.append(lv.group)
    return seen


def in_group(group: str) -> list[str]:
    return sorted(n for n, lv in LEVERS.items() if lv.group == group)


def as_dict() -> dict:
    """JSON-serializable registry for `chad --levers` / the ablation driver."""
    return {n: {"description": lv.description, "group": lv.group, "kind": lv.kind}
            for n, lv in LEVERS.items()}
