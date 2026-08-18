"""Named harness levers, each individually disableable at runtime.

A harness change that ships hardcoded can only be attributed by a full sweep with the
code reverted. That is the trap the early bundles fell into: fourteen fixes land as
three diffs, the rate moves, and nothing tells you which fix moved it — or whether one of
them is a regression the others are masking. Leave-one-out ablation is the cure, and it
needs exactly one thing from the harness: a way to switch a single behavior off without
touching the tree.

So every behavioral lever gets a name here and a `levers.enabled("name")` guard at its
site. `CHAD_ENABLE=a,b` turns on a and b (`CHAD_ENABLE=all` turns on everything);
`CHAD_DISABLE=a,b` subtracts from whatever is enabled, so the leave-one-out ablation
idiom is `CHAD_ENABLE=all CHAD_DISABLE=x`.

Unknown names in either variable are a hard error, not a warning. A typo'd lever in an
ablation script would otherwise run the *unmodified* harness and report a delta of zero,
which reads as "this lever does nothing" — the single most expensive way for this
machinery to lie to you. `validate_env()` runs it at startup, not at first use.

Levers default OFF (since 1.10.0), with one scoped exception: `LEAN_DEFAULTS` below,
which re-arms the bash-route levers under CHAD_LEAN because that arm deletes the tools
those guarantees were living in. Every lever here was born from a plausible failure
narrative, but the measured record says the bare model + tool loop matches or beats the
full stack on the benchmark that motivated them (the 2026-08 clean-slate arms: BARE-0
within noise of FULL, at substantially less token spend). Default-on made the scaffold
the null hypothesis; it is now the alternative, and a lever ships ON only with a
pre-registered, positive, measured contrast behind it. The registry and the firing
telemetry stay so that any lever remains one env var away from a fair re-trial.

Two fields carry the judgment a bare on/off switch can't:

`group` — the harness iteration that introduced the change, so `ablate.py --group iter3`
can price one bundle without paying for the others.

`kind` — what the ON state buys.
  BEHAVIOR       the change adds a behavior; without it the agent is merely less helped.
  REGRESSION_GUARD  the change fixes a once-demonstrated bug, and OFF restores that
                 bug's possibility. A grep that reports "[no matches]" on a tree it
                 never finished walking is a lie, not a configuration — but the
                 clean-slate arms measured no aggregate score cost to running without
                 these, so they follow the same default-OFF, evidence-to-ship rule.
                 The classification survives so `chad levers` can say exactly what
                 enabling one buys back.
"""

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass

BEHAVIOR = "behavior"
REGRESSION_GUARD = "regression-guard"

# How a lever's use is observable in a trace (plan 123: exists / exercised / improved).
#   EVENT    the lever has a discrete action site that calls `fired()` when its
#            behavior actually runs — a steer injected, an edit reverted, a gate
#            biting. "Configured" and "exercised" are different claims, and only a
#            firing event ties an outcome delta to a lever that actually did
#            something on that task.
#   PASSIVE  the lever's whole effect is static prompt content; "fired" would mean
#            "the system prompt was built", which the config record already says.
EVENT = "event"
PASSIVE = "passive"


@dataclass(frozen=True)
class Lever:
    description: str
    group: str
    kind: str = BEHAVIOR
    fires: str = EVENT


# Keep each description accurate enough that `chad levers` is a readable inventory of
# what the harness does beyond a bare model call. It is the artifact a reviewer reads to
# judge how much of the agent's behavior comes from the model and how much from the
# scaffolding around it.
LEVERS: dict[str, Lever] = {
    # --- group "iter2": implemented from the overnight forensics; only ever measured
    #     on the 7 tasks it was derived from, never on the other 23. --------------
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
    "verification_matrix": Lever(
        "After ~8 exploratory-bash steps since the last change (even AFTER an edit has "
        "landed — the case investigation_gate can't see), pull the turn into a bounded "
        "verification phase: for each task requirement, close it with a real causal run "
        "OR an explicit 'unverified', then `done`. Reuses done_audit's requirement "
        "extractor; targets the 69%-of-fails non-convergence (ran-out) bucket while the "
        "honest-unverified escape keeps it from thrashing on the genuinely-unverifiable.",
        "iter16"),
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
        "walking — the root trigger of the measured zero-match stall.",
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

    # --- group "iter3": implemented straight off the "iter2" traces. NEVER MEASURED.
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
        "budget to spare, wrong output format' losses — writing directories where files "
        "were required, or passing 8 of 9 checks on a format mismatch. OFF accepts the "
        "first done as-is.",
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

    # --- group "iter6": line-addressed edits corrupting multi-line structures;
    #     derived from the measured 9B/35B dogfood (10 ignored parse warnings, LOOP
    #     ABORT with a severed def signature). ---------------------------------------
    "syntax_revert": Lever(
        "An edit that turns a cleanly-parsing Python file into ANY SyntaxError is "
        "REVERTED (generalizes syntaxgate_revert beyond IndentationError), with the "
        "reject naming the severed multi-line statement and echoing the current lines. "
        "Applies to edit/replace_lines/insert_lines/replace_symbol/insert_symbol "
        "(non-Python languages and whole-file write are gated separately: "
        "ts_edit_revert / write_gate). OFF restores warn-and-land: the "
        "dogfood landed ~10 corrupting line edits over ignored parse warnings and "
        "aborted with the file broken.",
        "iter6", REGRESSION_GUARD),
    "edit_result_echo": Lever(
        "replace_lines/insert_lines results echo the changed region with its POST-edit "
        "line numbers plus a shift note, so follow-up edits re-anchor on numbers the "
        "model has actually seen instead of reusing the ones from a pre-edit read (the "
        "measured stale-number failure that conflated two adjacent defs).",
        "iter6"),
    "stale_file_guard": Lever(
        "replace_lines/insert_lines reject-once, with a fresh numbered view of the "
        "target region, when the file changed on disk since the model last saw it "
        "(bash/sed, git checkout, test runs) — line numbers minted against the old "
        "content are blind. The reject refreshes the anchor, so it never locks out.",
        "iter6"),

    # --- group "iter7": parse-clean semantic drift in whole-unit rewrites,
    #     derived from the first successful syntax-gate dogfood (replace_symbol dropped
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

    # --- group "iter8": the validate-before-write choke point, from a 624-session
    #     trace sweep: broken code LANDED 4x more often than it was rejected, 51/55
    #     landed breaks came through warn-only write, and non-Python files had no
    #     revert at all (a JS and an R file each compounded into an unrecoverable
    #     tree). ---------------------------------------------------------------------
    "ts_edit_revert": Lever(
        "Extends the Python edit revert to every parseable language: a targeted edit "
        "that takes a file "
        "with zero tree-sitter ERROR/MISSING nodes to one with any is REVERTED. Dirty "
        "baselines stay editable; no-grammar files are never blocked. OFF restores "
        "warn-and-land on non-Python code — the measured JS break that six follow-up "
        "edits then compounded into an unrecoverable tree.",
        "iter8", REGRESSION_GUARD),
    "write_gate": Lever(
        "Whole-file write refuses content that would newly break the parse (existing "
        "clean file -> broken, or a new Python file that doesn't parse); an "
        "already-broken file stays overwritable as the repair path, and a new "
        "tree-sitter-language file only warns (grammar-quirk risk). OFF restores the "
        "warn-only write that delivered 51 of the 55 landed syntax breaks in the "
        "trace sweep.",
        "iter8", REGRESSION_GUARD),
    "broken_streak_steer": Lever(
        "After 2+ consecutive landed mutations that leave a Python file unparseable "
        "(the sanctioned already-broken path), the parse warning escalates to a "
        "whole-file-rewrite / restore-known-good steer. The dogfood sweep measured "
        "a 14-landing broken streak that plain per-edit warnings never interrupted.",
        "iter8"),
    "write_diff_note": Lever(
        "An overwrite's result carries '(+a -d lines vs previous)' so the model can "
        "check the change's size against its intent — the cheap core of the diff-echo "
        "idea; targets the lost-track-of-file-state no-op/loop episodes.",
        "iter8"),

    # --- group "iter9" (deadline awareness): an unattended runner passes the wall
    #     budget down (CHAD_TURN_BUDGET_S), so the governor arms on every run. This
    #     lever is the wrap-up NUDGE that rides on top of it. -------------------------
    "wrapup_window": Lever(
        "One-shot wall-clock steering note in the turn's final stretch (remaining <= "
        "max(120s, 15% of the wall budget)): stop exploring and land your best answer "
        "before the force-stop. Distinct from the governor's no-progress hard-stop — this "
        "fires even on a productive turn so it commits a usable partial instead of being "
        "SIGKILLed mid-edit. Only active when a wall budget is configured; off with the "
        "governor (CHAD_NO_GOVERNOR).",
        "iter9"),

    # --- Think-spiral salvage: close-and-continue force-closes a runaway
    #     <think> at the think ceiling and keeps decoding the action in-step (the engine
    #     mechanism has no lever of its own — it is a constructor default, so it shows up
    #     neither here nor under CHAD_DISABLE). This lever is the ESCALATION on top of it.
    #     ---------------------------------------------------------------------------
    "no_think_escalation": Lever(
        "After 2 consecutive steps that hit the gen cap (or were salvaged) yet produced no "
        "tool call — the think-spiral signature — run the next step with <think> disabled "
        "to force an action, then restore. Mechanical 'act now', replacing a third prose "
        "nudge that demonstrably does not land. Requires the close-and-continue ceiling to "
        "be armed, which is now the DEFAULT in the full arm (it was env-only when this "
        "lever shipped, and the old note that 'default chad is unaffected' no longer "
        "holds) — and OFF by default in the lean arm, where this lever is therefore inert "
        "unless CHAD_THINK_CEILING re-arms it.",
        "iter9"),

    # --- Done-audit: the largest measured fail bucket (20/43)
    #     was dones claiming SPECIFIC verification that did not hold up — the model
    #     verifies a WEAKER predicate than the task's own wording, with huge unused
    #     budget (one run declared done at 84s of a 900s wall), and done_spec_recheck
    #     was ON for every one of them. ---------------------------------------------
    "done_audit": Lever(
        "Bounce a would-otherwise-be-accepted `done` ONCE per turn (turns that did real "
        "work, with wall runway to spare) with the task statement's own requirement "
        "lines quoted verbatim plus stat-level facts about every path the task names, "
        "then accept the next `done` unconditionally (the anti-spiral latch). "
        "Supersedes the generic done_spec_recheck steer while enabled — stacking both "
        "would force two bounces per turn. OFF restores the done_spec_recheck path.",
        "iter10"),

    # --- (turn-level think budget): the per-generation think ceiling watches any ONE
    #     generation and is blind to a turn that death-by-a-thousand-cuts its way to a
    #     wall-death across many separate thinks under that ceiling (measured: 123k
    #     total across 13 thinks, no single one >= 6k). This watches CUMULATIVE
    #     instead. Only active when a wall budget is configured (self._turn_budget_s) —
    #     off entirely in interactive/unmetered runs, like wrapup_window. -------------
    "turn_think_budget": Lever(
        "A cumulative per-turn reasoning-token budget (wall- and decode-speed-aware, "
        "clamped 8k-24k): a one-shot soft steer at half spent, then a no-think "
        "THROTTLE once spent — one forced no-think action step per 3k further think "
        "tokens (guardrails.turn_think_throttle), so thinking restores when the model "
        "stops over-spending; a blanket rest-of-turn mute regressed run1 passes with "
        "garbled no-think tails. Acts only at step boundaries on fresh "
        "generations, never an in-flight one (the per-generation ceiling's lesson). "
        "Off in plan mode, for "
        "read-only-intent turns, and below a 300s wall budget (short relaunch tails "
        "belong to hard_wrapup). Only active with a wall budget configured.",
        "iter11"),

    # --- (hard wrap-up): the wrapup_window nudge only lands at a step boundary and
    #     fired in 3 measured runs, rescuing none, because the model is usually buried
    #     in one long generation when the window opens. This aborts it mid-stream. -----
    "hard_wrapup": Lever(
        "At the wall deadline (within max(90s, 10% of the budget)) ABORT the in-flight "
        "generation mid-stream and force ONE time-boxed, no-think landing turn that writes "
        "each deliverable to its exact path before the runner SIGKILLs the task — the "
        "mid-generation backstop the step-boundary nudge (3 firings, 0 rescued) can't "
        "be. "
        "Only active with a wall budget configured; inert in interactive/unmetered runs.",
        "iter12"),

    # --- group "iter13": lever *interactions*. Two otherwise-winnable tasks were lost not
    #     to the model but to guardrails fighting each other — a garbled tool
    #     call accepted as the final answer once the shared nudge counter and the
    #     done-audit latch were both spent, and the investigation gate counting a
    #     `git merge`/`apt-get` ops workflow as "investigation" and demanding an edit
    #     at a point where there was nothing yet to edit. --------------------------
    "garble_never_final": Lever(
        "A step whose text contains tool-call markers is NEVER accepted as a final "
        "answer and never reaches the done-audit final-answer twin: garbles get their "
        "own nudge counter (cap 6, not shared with cap-truncations), a canonical "
        "call exemplar from the 2nd consecutive garble, a scrub of the previous "
        "garbled message body (so the model stops few-shotting its own broken "
        "dialect), and — if nudges run out — a banked-note hard stop. OFF restores "
        "the shared-counter failure: a run of garbles burns the shared counter + "
        "audit latch and the last is accepted as the answer with most of the wall "
        "budget still left.",
        "iter13", REGRESSION_GUARD),
    "audit_absent_rebounce": Lever(
        "When the done-audit's first bounce stat'ed task-mentioned paths as ABSENT "
        "and they are STILL absent at the accepting done/final-answer (runway > "
        "120s), bounce one final time naming only the missing paths; the next done "
        "is then accepted unconditionally (hard cap 2 bounces/turn). The first "
        "audit's acceptance promise is reworded to stay truthful while this is on.",
        "iter13"),
    "gate_ops_exempt": Lever(
        "A bash step whose command is not provably read-only counts as ACTION for "
        "the investigation gate (resets the read-only streak): `git merge`, "
        "`apt-get install`, redirects etc. are ops progress, not investigation. "
        "The gate keeps firing on genuine grep/read loops. OFF restores the gate "
        "harassing ops-heavy tasks that legitimately run many non-read commands.",
        "iter13"),

    # --- group "iter14": tool-result economics (2026-07). Two ideas from a teardown of a
    #     token-optimizing agent harness: normalize typographic unicode when matching
    #     edits, and never re-send file content the model provably already has; both
    #     map cleanly onto a prefill-dominated local loop. ---------------------------
    "edit_typo_match": Lever(
        "Fourth edit-match rung: when exact / escape-normalized / whitespace-flexible "
        "matching all miss, retry the whitespace-flexible match with typographic "
        "punctuation folded to ASCII on BOTH sides (curly quotes, en/em dashes, "
        "ellipsis, non-breaking space) — the drift when a model re-types prose or "
        "docstrings it saw rendered. Still requires a unique match; never fires on "
        "code that matched an earlier rung. OFF loses only the recovery.",
        "iter14"),
    "subagent_evidence_warn": Lever(
        "A sub-agent that returns a confident, non-empty report having dispatched ZERO "
        "tools answered from model memory, not the repo; its folded result gets an "
        "explicit verify-before-relying warning appended. Warn-not-reject on purpose: "
        "a local re-spawn doubles GPU cost, and a false reject breaks the turn while a "
        "false accept merely restores the pre-warning status quo. OFF folds the "
        "unverified report back silently.",
        "iter14"),
    "dup_result_elide": Lever(
        "A read-only tool result byte-identical to a tool message still in the "
        "transcript is replaced with a short pointer to the earlier copy instead of "
        "re-appending the full body to the prefill. Equality against the LIVE "
        "transcript is the safety proof: compaction rewrites or drops old messages, "
        "which breaks equality, so content is only elided while it is genuinely "
        "still in context. OFF re-appends duplicate output verbatim.",
        "iter14"),

    # --- group "iter15": the audit-silent early-quit class. The no-empty-diff hard stop
    #     sits ABOVE the done-audit in both accept paths, so a done with no
    #     landed+verified change ends the turn before the audit can engage: a
    #     progress note carrying the model's own completion claim gets banked, and
    #     each relaunch re-dones into the same stop until the continue allowance
    #     is exhausted — gate order, not the audit's own scoping, kept the
    #     task-grounded steer away from exactly the turns that needed it. --------
    "audit_churn_handoff": Lever(
        "When the no-empty-diff gate is about to hard-stop a done/final-answer (no "
        "landed+verified change on an action task) and the done-audit has not fired "
        "this turn, bounce ONCE with the audit steer (task requirement lines quoted "
        "+ stat-level path facts, truthful promise: acceptance follows a landed+"
        "verified change) instead of ending the turn — the steer lands in context "
        "with the turn's work rather than in a banked progress note the relaunch "
        "re-confirms. The next empty-diff done hard-stops exactly as before (churn "
        "capped, not replaced); latch and bounce caps shared with done_audit. "
        "Inert unless done_audit is also enabled. OFF restores the immediate stop.",
        "iter15"),

    "bash_auto_background": Lever(
        "On timeout, move a still-running bash command to the background instead of "
        "killing its process group: its output keeps streaming to a spill file whose "
        "path the result names, and an `[exit <code> at HH:MM:SS]` footer marks "
        "completion. The spill file IS the interface — no new tool, no task registry "
        "for the model to learn, since it can already read and grep a path. A killed "
        "install/build/download costs a full re-run of work already paid for, and "
        "`cmd &` (what the kill message suggests when off) hands back a process with "
        "nowhere to put its output. Bounded: at most 2 concurrent, an absolute "
        "lifetime each, and every one of them terminated when chad exits or is "
        "signalled; a user interrupt still kills outright. OFF restores the "
        "kill-and-advise path exactly.",
        "iter15"),
    "steer_verify_specific": Lever(
        "One system-prompt block steering the FINAL verification at the task's own "
        "stated acceptance check (run the named command/test end-to-end and read its "
        "output) instead of a proxy like file-existence or grep-counting, with a "
        "WRONG/CORRECT pair. The measured class: of 42 failed runs that took a "
        "done-audit bounce, 38 did re-verify afterwards — with a weaker or wrong "
        "check — and still shipped a bad solution, most with over 80% of the wall "
        "unused; a harness-side predicate cannot judge check quality, so this steers "
        "the choice upstream where it is made. Rides the "
        "cached prefix; OFF removes the block byte-exactly.",
        "iter15", fires=PASSIVE),
    "scoped_destructive_guard": Lever(
        "Scope the destructive-bash seatbelt's recursive-rm screen to targets whose "
        "loss is actually catastrophic — filesystem root, top-level directories, "
        "home trees at any depth, whole-cwd/parent globs — instead of ANY absolute "
        "path, and have the headless block tell the model the guard (not a person) "
        "refused and how to narrow the delete. In a container every real path is "
        "absolute, so the legacy shape denied ordinary scoped deletes "
        "(`rm -rf /tmp/test-deploy`) as '[denied by user]' — the measured cost: 152 "
        "denials across 26 sessions, one turn re-phrasing the same delete 30 times. "
        "Also screens every rm target in a compound command, not just the first "
        "(a strictly wider net than legacy on multi-target commands). mkfs / "
        "dd-to-device / fork-bomb / curl|sh screens are untouched. OFF restores the "
        "any-absolute-path shape and the bare denial text.",
        "iter15"),
    "late_continue_replenish": Lever(
        "Keep granting auto-continue relaunches beyond the base allowance while a "
        "QUARTER of the task wall remains, instead of half. Step-capped turns that "
        "kept working but never landed a verified change ended their task with "
        "12-50% of the wall budget stranded once the 0.5 replenish line passed — "
        "attempts the wall could still have paid for. The absolute relaunch ceiling "
        "(6) and the minimum-remaining-wall floor for any relaunch are unchanged. "
        "OFF restores the 0.5 threshold exactly.",
        "iter15"),
    "capped_think_credit": Lever(
        "Credit a generation that ended inside <think> — no closing tag, whatever "
        "stopped it — as reasoning in full, not zero. The turn think budget only "
        "counted think tokens when the model closed the block or a stop CONDITION "
        "fired, so a generation truncated at the raw token cap mid-reasoning (the "
        "reasoning_length_stop telemetry already names it) was invisible to the "
        "budget meant to bound it — and those are the largest generations there "
        "are, one full cap each. OFF restores the blind accounting; no threshold "
        "or clamp changes either way, only what the budget can see.",
        "iter15"),

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
        "playbook", fires=PASSIVE),

    # --- from the ai-codex teardown: an index the model already has beats one it must
    #     fetch. chad had the ranked map but only as a tool; this puts a digest in-prompt.
    "workspace_map": Lever(
        "Inject a ranked repo_map digest into the system-prompt dynamic tail instead of "
        "a flat file listing, so the model orients without a reflexive step-1 repo_map "
        "call. Degrades to the flat listing when repomap is unavailable (see prompt.py).",
        "ai-codex", fires=PASSIVE),

    # --- group "ambient": facts in the result channel. Trace measurement showed the
    #     model routes ~83% of searching through bash and ignores the symbolic tools
    #     under maximal prompt steering; these levers stop asking it to change routes
    #     and put the harness's state knowledge in the results it already reads.
    #     Facts only, never admonitions — the clean-slate arms priced the admonition
    #     class at ~0.
    "env_manifest": Lever(
        "Append a session-start environment manifest (toolchains present with "
        "versions, notable absences, package managers, and the search/text toolbox "
        "the shell route does its looking with) to the system-prompt tail, "
        "answering the which/--version/pip-list probe class up front. Measured "
        "fail-enriched: 3.33 probes/trial in fails vs 2.46 in passes, 37% of probes "
        "failing outright. Stamped once at session start and never updated in place "
        "(prefix-resident by design); post-install state is session_ledger's job.",
        "ambient", fires=PASSIVE),
    "session_ledger": Lever(
        "Append one cumulative fact line to landed-mutation results and done bounces "
        "— files edited (with symbols where known), files written, and when "
        "something was last actually run (`last verifying run: N calls ago "
        "(pytest → exit 1)`). State facts only, past tense, with provenance — never "
        "an instruction, so it cannot be quoted as verification the model didn't "
        "perform. Targets the wrong-verify done class (the #1 measured fail bucket) "
        "as a state-visibility gap, and blind re-reads (21.8% of reads; fail 1.20 "
        "vs pass 0.86/trial). Elided when state is unchanged since last emission.",
        "ambient"),
    "bash_read_skeleton": Lever(
        "The first time a source file's content comes back through bash "
        "cat/sed/head or `read`, append a one-line symbol map from the tags cache "
        "(`[file] x.py: parse() 30-71 · _tune() 73-88`); when a bash grep for a "
        "known symbol returns nothing, append its real definition site. Symbol "
        "intelligence delivered on the route the model actually uses — it never "
        "has to learn a tool name. Once per file per session.",
        "ambient"),
    "edit_miss_diagnose": Lever(
        "An `old string not found` says WHY it missed instead of only offering the "
        "closest line: the first line and column where `old` stops matching the file "
        "(both rendered with whitespace visible), or — when every line `new` would add "
        "is already in the file — that the site may already be edited. OFF leaves the "
        "generic 'copy it exactly (mind indentation)', which names one hypothesis; "
        "measured, a run spent four calls on that hypothesis for a stray leading quote "
        "and re-sent an already-applied edit reading the same text as an indent "
        "problem.",
        "ambient"),
    "bash_empty_diagnose": Lever(
        "A bash command that printed nothing gets one line saying why, from facts "
        "the harness already holds: a `sed -n 'A,Bp'` whose range starts past the "
        "end of the file is told the file's real length, and an empty pipeline is "
        "told which filter stage matched nothing. Bare `[no output]`/`[exit 1]` "
        "results were 5% of the tool calls in a measured lean run, and each one "
        "cost a whole round trip to re-derive (the `wc -l` the model had to spend "
        "its next call on). The `read`/`grep` tools already refuse to report an "
        "empty result without its scope; this is that contract on the bash route, "
        "where a lean arm does all of its reading.",
        "ambient", REGRESSION_GUARD),
    "bash_line_clip": Lever(
        "Cap any single line of bash output at BASH_LINE_CHARS, spilling the full text "
        "to a file when the clip is large. head+tail budgets the total and assumes "
        "output is made of LINES; a source map or minified bundle is one line of tens "
        "of thousands of chars, so the budget returns 20k chars sliced out of the "
        "middle of base64. Measured: a search that reached a gitignored build dir "
        "returned a 92k-char .js.map line, ~5k tokens of it prefilled for a 147s stall "
        "and zero information. `tool_grep` has capped its rows since it shipped; OFF "
        "restores the uncapped bash route.",
        "ambient", REGRESSION_GUARD),
    "bash_trim_keep_failures": Lever(
        "When a bash result is too long and gets head+tail trimmed, the failure rows "
        "in the dropped MIDDLE ride along verbatim (capped, deduped). Head+tail "
        "assumes the middle is filler, which holds for build logs and not for test "
        "runners: ava prints each failing test's NAME and assertion in the body and "
        "only the counts at the end, so a trimmed run measured out as `2 tests "
        "failed` with both names cut. OFF restores counts-without-names.",
        "ambient", REGRESSION_GUARD),
    "rg_replace_flag_note": Lever(
        "An `rg -r/--replace` result is labelled as rewritten output. ripgrep's `-r` "
        "takes a REPLACEMENT (it recurses by default), so a model reaching for grep's "
        "recursive flag writes `rg -rn PATTERN` and silently gets rows whose matched "
        "text has been replaced by `n`. Measured twice in one run, the second time "
        "convincing the model its own test file had been corrupted — it spent three "
        "calls investigating and closed with a wrong explanation. The rows on screen "
        "are not the file's text, and only the harness can see that from the flags.",
        "ambient", REGRESSION_GUARD),
    "verify_baseline": Lever(
        "Record the project test command's outcome the first time it runs BEFORE "
        "any edit lands, and recall it — exit code plus the runner's own summary "
        "rows, verbatim — on a failing verifying run afterwards. A measured run "
        "spent nine turns re-running subsets and grepping for `test.failing` "
        "markers to work out which of two failures pre-dated its first edit, a "
        "question no observation can answer once the edit has landed. Adds the "
        "prompt line that asks for the pre-edit run in the first place; without "
        "it there is nothing to recall.",
        "ambient"),

    # --- group "ctxengine": the uniform-symbols work. -------------------------------
    "post_edit_diagnostics": Lever(
        "Append the language server's post-edit typecheck errors (~50 tokens, errors "
        "only, capped) to edit/symbol-edit results — the semantic tier above "
        "syntaxgate's parse check. A type error surfaced in the edit result replaces "
        "the multi-thousand-token failed test run the model would otherwise need to "
        "discover it. Never spawns a server (an edit must not pay a cold start): "
        "only a server already warmed by find_refs/hover/disambiguation is "
        "consulted; absent one, silence.",
        "ctxengine"),

    # --- group "contract": one line-number rendering across every surface that
    #     shows file lines, so a number the model sees anywhere is a valid edit
    #     anchor and carries the same freshness bookkeeping. Read and the edit
    #     echoes already share it; these levers extend it to the remaining
    #     surfaces. ------------------------------------------------------------
    "grep_anchor": Lever(
        "grep renders matches grouped per file in read's numbered format (a "
        "`path:` header, right-aligned line numbers, `*` marking match rows when "
        "context lines are shown) and records each rendered file as seen, so a "
        "grep hit is a first-class line-edit anchor with the same staleness "
        "bookkeeping a read gets. OFF keeps the flat `path:line: text` rows. "
        "Fires per grep that rendered anchors, with file and match counts.",
        "contract"),
    "catalog_clip": Lever(
        "Each `<available_skills>` row is trimmed to whole sentences within a "
        "260-char budget (first sentence always kept, later ones while they fit, "
        "'…' when any were dropped). The catalog is the largest block in the "
        "system prompt and survives every turn in the warm prefix: on a host with "
        "62 installed skills the descriptions alone measure 19.0k chars — median "
        "305 per row, so it is the whole distribution, not a few outliers — and "
        "this takes it to 9.8k with 34 rows clipped. A row is TRIGGER text the "
        "model matches a task against, so the contrast this has to win is skill "
        "selection, not prompt size; the detail it drops is procedure, which "
        "activation serves in full. Fires per render that clipped, with the count.",
        "contract"),
    "read_range_footer": Lever(
        "Skeleton and clipped reads name the exact un-shown line ranges in "
        "directly pasteable form (a concrete `read(path, offset=N, limit=M)` per "
        "elided span) instead of describing the paging mechanism in prose. Fires "
        "per read that emitted a concrete continuation.",
        "contract"),

    # --- group "safety": blast-radius containment for unattended mutation. ---------
    "yolo_seatbelt": Lever(
        "In yolo mode on macOS, each bash command's shell child runs under a "
        "Seatbelt profile (sandbox-exec) that denies file writes outside the "
        "workspace, temp dirs, tool caches, and ~/.chad — reads, network, and exec "
        "stay open, and ~/.chad/checkpoints (the undo history) is carved back out "
        "so a sandboxed command cannot delete it. The model process is never "
        "sandboxed, only the spawned shell. The startup probe proves the profile "
        "ENFORCES (an allowed write must land, a denied write must not) before any "
        "confinement is claimed; either failure runs unconfined with a loud log. "
        "Fires when a denial is detected in command output: each fire is a write "
        "the command-pattern denylist did not catch.",
        "safety"),
    "edit_checkpoint": Lever(
        "Before a file-mutating tool lands, commit a workspace snapshot to a "
        "shadow git repo under ~/.chad/checkpoints (the project's own .git is "
        "never touched or required); /undo and /restore in the TUI revert from "
        "it. Fires once per snapshot taken and once per restore performed.",
        "safety"),
    "seatbelt_protect_git": Lever(
        "Opt-in tier on top of yolo_seatbelt: the workspace's git metadata "
        "(.git, and a worktree's external gitdir + common dir) is write-DENIED "
        "inside the otherwise-writable workspace, so an unreviewed command "
        "cannot destroy project history (`rm -rf .git`) — and with "
        "edit_checkpoint on, the undo snapshots stay tamper-proof from inside "
        "the sandbox too. The cost is real: every .git-writing git command "
        "(commit, add, checkout) EPERMs — sized at <=2.65% of 91,910 real "
        "session commands, concentrated in 14% of sessions — which is why this "
        "is its own lever, not part of the base profile. Inert unless "
        "yolo_seatbelt is also enabled; a denial it causes surfaces through "
        "yolo_seatbelt's own firing (the profile cannot say which rule bit).",
        "safety", fires=PASSIVE),
    "bash_env_guard": Lever(
        "Spawned bash children get a FILTERED copy of the environment: variable "
        "names shaped like credentials (…_TOKEN, …_SECRET, …_PASSWORD, "
        "…_API_KEY, …_ACCESS_KEY(_ID), …_PRIVATE_KEY, …_CREDENTIALS) are "
        "dropped, closing the gap where a filesystem sandbox still hands every "
        "command the operator's cloud keys. Name-pattern only — values are "
        "never read or logged. OFF (the default) inherits the parent env "
        "untouched, today's behavior; commands that legitimately need a "
        "credential need the lever off. Fires per spawn that actually dropped "
        "something, with the count.",
        "safety"),
}


class UnknownLever(ValueError):
    """A CHAD_ENABLE/CHAD_DISABLE entry that names no registered lever."""


def _parse(var: str) -> frozenset[str]:
    raw = os.environ.get(var, "")
    names = frozenset(n.strip() for n in raw.split(",") if n.strip())
    unknown = names - LEVERS.keys() - {"all"}
    if unknown:
        raise UnknownLever(
            f"{var} names unregistered lever(s): {sorted(unknown)}. "
            f"Known levers: {sorted(LEVERS)} (or 'all')")
    return names


# Levers the lean arm (CHAD_LEAN) starts with ON. Not an exception to the
# default-OFF contract so much as its consequence: default-OFF is a statement about
# levers that ADD behavior on top of a full tool surface, and lean deletes the tool
# surface. `read` and `grep` carry four of these behaviors themselves — grep refuses
# to report an empty result without its scope (`grep_zero_match_notice`), read names
# the ranges it did not show (`read_range_footer`) — and hiding those tools silently
# repeals the contract rather than testing it. Worse, it repeals it invisibly: a lever
# bound to a hidden tool cannot fire, so an ablation reads it as "exercised and
# useless" when it was never reachable. These four put the same guarantees on the bash
# route, which is the only route lean has.
#
# CHAD_ENABLE still adds and CHAD_DISABLE still subtracts, so `CHAD_LEAN=1
# CHAD_DISABLE=bash_empty_diagnose` is a valid leave-one-out arm and `active()` keeps
# reporting exactly what ran.
LEAN_DEFAULTS = frozenset({
    "env_manifest",             # bash is the only route, so its toolbox IS the surface
    "bash_read_skeleton",       # the only structure signal left once repo_map is gone
    "bash_empty_diagnose",      # grep_zero_match_notice / read_range_footer, on bash
    "bash_trim_keep_failures",  # bash IS the test channel here
    "verify_baseline",          # ditto, and the pre-edit run is otherwise unrecorded
    "bash_line_clip",           # no tool_grep row cap on this route either
    "edit_miss_diagnose",       # `edit` is the ONLY editor in this arm
    "rg_replace_flag_note",     # ripgrep is the whole search surface here
})


def _enabled_set() -> frozenset[str]:
    """Levers ON right now: LEAN_DEFAULTS (only under CHAD_LEAN) plus CHAD_ENABLE
    ('all' = everything), minus CHAD_DISABLE. Read live, never cached — the eval
    harness flips these between tasks in-process."""
    from . import config
    on = _parse("CHAD_ENABLE")
    if "all" in on:
        on = frozenset(LEVERS)
    elif config.flag("CHAD_LEAN"):
        on = on | LEAN_DEFAULTS
    off = _parse("CHAD_DISABLE")
    if "all" in off:
        off = frozenset(LEVERS)
    return on - off


def enabled(name: str) -> bool:
    """True only if `name` is switched on via CHAD_ENABLE (and not subtracted by
    CHAD_DISABLE). Raises if `name` is not registered — a guard on an unregistered
    lever is invisible to the ablation driver, so it must not silently pass."""
    if name not in LEVERS:
        raise UnknownLever(f"unregistered lever {name!r}; add it to levers.LEVERS")
    return name in _enabled_set()


_fire_counts: Counter = Counter()


def fired(name: str, step: int | None = None, **detail) -> None:
    """Record that a lever's behavior actually ran — not that it was configured.

    Call this at the ACTION site, inside the `enabled()` branch, at the moment the
    lever's distinctive effect is produced (a nudge returned, an edit reverted, a
    gate refusing, a result modified) — never at the bare `enabled()` check, and for
    predicate-style levers only when the lever CHANGED the outcome vs the legacy path.
    An arm's per-lever firing counts are what let an outcome delta be split into
    "exercised and useless" vs "never fired on these tasks" (plan 123).

    Emits one line to the session log: `LEVER {"lever": ..., "step": ..., ...}` —
    machine-parseable, one JSON object per event. A fire while the lever is disabled
    is a call-site bug (the guard leaked); it is recorded with `"disabled": true`
    rather than crashing a live task, and the test suite treats it as a failure.
    """
    if name not in LEVERS:
        raise UnknownLever(f"unregistered lever {name!r}; add it to levers.LEVERS")
    _fire_counts[name] += 1
    evt: dict = {"lever": name}
    if step is not None:
        evt["step"] = step
    if name not in _enabled_set():
        evt["disabled"] = True
    if detail:
        evt.update(detail)
    logging.getLogger("chad").info("LEVER %s", json.dumps(evt, sort_keys=True, default=str))


def fire_counts() -> dict[str, int]:
    """Per-lever firing counts for this process, for tests and end-of-run summaries."""
    return dict(_fire_counts)


def validate_env() -> None:
    """Raise on a typo'd CHAD_ENABLE/CHAD_DISABLE now, at startup, rather than at the
    first compaction — which on a long sweep is forty minutes of wall-clock into a run
    that is already measuring the wrong thing. Called from the CLI and the eval runner."""
    _enabled_set()


def active() -> list[str]:
    """Registered levers currently enabled, sorted. Recorded in eval rows so a banked
    result carries the harness configuration that produced it. Empty on a default
    (bare) run."""
    return sorted(_enabled_set())


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
    """JSON-serializable registry for `chad levers` / the ablation driver."""
    return {n: {"description": lv.description, "group": lv.group, "kind": lv.kind}
            for n, lv in LEVERS.items()}
