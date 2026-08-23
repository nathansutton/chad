"""The bash-route levers chad ships, each individually disableable.

chad 2.0.0 is the lean harness: one tool surface (bash / edit / write /
write_todos / done) and the result-channel behaviors below, all ON by
default. Everything else the 1.x lever registry carried was measured against the
bare loop and never beat it (the 2026-08 clean-slate arms and the pack campaign
that followed: every pack inside the two-bare-arm null band), so it was removed
rather than left OFF behind switches.

What remains is the set whose mechanism is a generally-true property of the bash
route itself — a result never lies about what it searched, a truncated test run
keeps its failure rows, an edit miss explains itself — and each keeps a name and
a `levers.enabled()` guard for exactly one reason: leave-one-out ablation.
`CHAD_DISABLE=a,b` switches named levers off (`CHAD_DISABLE=all` runs the bare
loop); there is no enable list because ON is the default.

Unknown names in CHAD_DISABLE are a hard error, not a warning: a typo'd lever in
an ablation script would otherwise run the unmodified harness and report a delta
of zero — the most expensive way for this machinery to lie. `validate_env()`
runs the check at startup, not at first use.
"""

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Lever:
    description: str


# Keep each description accurate enough that `chad levers` is a readable
# inventory of what the harness does beyond a bare model call.
LEVERS: dict[str, Lever] = {
    "env_manifest": Lever(
        "Session-start toolchain inventory in the system prompt tail: which "
        "compilers/interpreters/package managers/search tools are present (with "
        "versions) and which commonly-probed ones are absent — answers the "
        "which/--version probe class in zero round trips."),
    "bash_read_skeleton": Lever(
        "A one-line symbol map appended the first time a source file's content "
        "comes back through bash cat/sed/head, and a definition pointer when a "
        "bash grep for a known symbol comes back empty — structure made ambient "
        "in the channel a shell-native model actually uses."),
    "bash_empty_diagnose": Lever(
        "Why a bash command produced nothing: a sed range past the end of the "
        "file, or the pipeline stage whose filter matched nothing. A bare "
        "'[no output]' costs a wasted round trip every time."),
    "bash_trim_keep_failures": Lever(
        "When long bash output is head/tail-trimmed, the failure rows from the "
        "omitted middle are kept verbatim — a trimmed test run must not lose "
        "the names of the tests that failed."),
    "verify_baseline": Lever(
        "The pre-edit outcome of the project's own test command, recalled on a "
        "failing post-edit run — answers 'was this failing before I started?' "
        "without re-deriving it."),
    "bash_line_clip": Lever(
        "Per-line cap on bash output, so a single minified/source-map line "
        "can't spend the whole output budget on base64. The full text goes to "
        "a spill file, never silently destroyed."),
    "edit_miss_diagnose": Lever(
        "A failed edit explains itself: whether the change looks ALREADY "
        "applied, or the first line/column where the sent text diverges from "
        "the file — instead of a bare 'old string not found'."),
    "trim_spill": Lever(
        "When compaction head/tail-trims an older tool result, the full "
        "original goes to a spill file and the collapse marker names the path "
        "— the one lossy path in the harness becomes a loan, not a deletion."),
    "result_spill": Lever(
        "When the per-result backstop cap clips a tool result, the full body "
        "goes to a spill file and the notice names the path, so the model can "
        "grep for what was dropped instead of re-running the call."),
    "rg_replace_flag_note": Lever(
        "One line naming what an `rg -r` result actually is (rewritten output, "
        "not the file's text) — `-rn` is `--replace n`, not grep's recursive "
        "flag, and the output looks plausible enough to act on."),
}


class UnknownLever(ValueError):
    """A CHAD_DISABLE entry that names no registered lever."""


def _disabled() -> frozenset[str]:
    raw = os.environ.get("CHAD_DISABLE", "")
    names = frozenset(n.strip() for n in raw.split(",") if n.strip())
    unknown = names - LEVERS.keys() - {"all"}
    if unknown:
        raise UnknownLever(
            f"CHAD_DISABLE names unregistered lever(s): {sorted(unknown)}. "
            f"Known levers: {sorted(LEVERS)} (or 'all')")
    if "all" in names:
        return frozenset(LEVERS)
    return names


def enabled(name: str) -> bool:
    """True unless `name` is switched off via CHAD_DISABLE. Raises if `name` is
    not registered — a guard on an unregistered lever is invisible to the
    ablation driver, so it must not silently pass. Read live, never cached —
    the eval harness flips CHAD_DISABLE between tasks in-process."""
    if name not in LEVERS:
        raise UnknownLever(f"unregistered lever {name!r}; add it to levers.LEVERS")
    return name not in _disabled()


_fire_counts: Counter = Counter()


def fired(name: str, step: int | None = None, **detail) -> None:
    """Record that a lever's behavior actually ran — not that it was configured.

    Call this at the ACTION site, inside the `enabled()` branch, at the moment
    the lever's distinctive effect is produced. Emits one line to the session
    log: `LEVER {"lever": ..., "step": ..., ...}` — machine-parseable, one JSON
    object per event. A fire while the lever is disabled is a call-site bug (the
    guard leaked); it is recorded with `"disabled": true` rather than crashing a
    live task, and the test suite treats it as a failure.
    """
    if name not in LEVERS:
        raise UnknownLever(f"unregistered lever {name!r}; add it to levers.LEVERS")
    _fire_counts[name] += 1
    evt: dict = {"lever": name}
    if step is not None:
        evt["step"] = step
    if name in _disabled():
        evt["disabled"] = True
    if detail:
        evt.update(detail)
    logging.getLogger("chad").info("LEVER %s", json.dumps(evt, sort_keys=True, default=str))


def fire_counts() -> dict[str, int]:
    """Per-lever firing counts for this process, for tests and end-of-run summaries."""
    return dict(_fire_counts)


def validate_env() -> None:
    """Raise on a typo'd CHAD_DISABLE now, at startup, rather than forty minutes
    into a run that is already measuring the wrong thing."""
    _disabled()


def active() -> list[str]:
    """Registered levers currently enabled, sorted. Recorded in eval rows so a
    banked result carries the harness configuration that produced it."""
    off = _disabled()
    return sorted(n for n in LEVERS if n not in off)


def as_dict() -> dict:
    """JSON-serializable registry for `chad levers`."""
    return {n: {"description": lv.description} for n, lv in LEVERS.items()}
