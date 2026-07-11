"""Named harness levers, each individually disableable at runtime.

A harness change that ships hardcoded can only be attributed by a full sweep with the
code reverted. That is the trap the iter-1/2/3 bundles fell into: fourteen fixes land as
three diffs, the rate moves, and nothing tells you which fix moved it — or whether one of
them is a regression the others are masking. Leave-one-out ablation is the cure, and it
needs exactly one thing from the harness: a way to switch a single behavior off without
touching the tree.

So every behavioral lever gets a name here and a `levers.enabled("name")` guard at its
site. `CHAD_DISABLE=a,b` turns off a and b; `evals/ablate.py` enumerates this registry
and runs the slice once per lever to produce a per-lever delta.

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
