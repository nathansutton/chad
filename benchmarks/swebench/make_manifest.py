"""Generate `legacy-manifest.tsv` — the LEGACY-FULL arm's lever configuration.

    uv run --project ../chad-legacy python benchmarks/swebench/make_manifest.py \
        --tree ../chad-legacy --out benchmarks/swebench/legacy-manifest.tsv

`CHAD_ENABLE=all` is not a configuration. The arm needs a list that says, for every
lever the tree ships, whether it is on and why — frozen by sha before the first scored
trial, so a reader can check the arm that ran against the arm that was pre-registered.

The generator reads the registry rather than a hand-kept list, so a lever that exists in
the tree and has no verdict here is an error instead of a silent omission. The verdicts
are the judgement; everything else on the row is read out of the tree.

TWO KINDS OF "NO"
-----------------
`state` is what the arm is configured to do, and only a CONFLICT turns a lever off: a
lever that would fight the runner's own confinement, or whose prerequisite the runner
has already taken away. Cost is not a reason — the steelman rule gives LEGACY-FULL its
best honest configuration, and "this feature is expensive" is a result, not a setting.

`fires_here` is a prediction, pre-registered: whether the lever can fire at all under
this campaign's conditions (mode=yolo, no TTY, one task per session, a Python-only
corpus). A lever that cannot fire stays ON — trimming the arm on a prediction would be
the quiet kind of mistake — but saying so in advance is what stops a zero firing count
from being explained after the fact.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# lever -> (state, fires_here, reason). Every lever in the tree must appear.
#
# ON + yes      the arm's ordinary configuration; nothing more to say than what the
#               registry description already says, so the reason names the trigger.
# ON + no       reachable in principle, but this campaign cannot produce its trigger.
# OFF           a conflict with the runner, stated.
VERDICTS: dict[str, tuple[str, str, str]] = {
    # --- off: conflicts with the runner's own confinement --------------------
    "yolo_seatbelt": (
        "OFF", "no",
        "CONFLICT: Seatbelt profiles do not nest. The runner confines every trial "
        "through a PATH shim (sandbox.py), identically for both arms; an inner profile "
        "would fail to apply and turn every bash call into an error. run.py sets "
        "CHAD_NO_SEATBELT=1 and asserts it."),
    "seatbelt_protect_git": (
        "OFF", "no",
        "CONFLICT: an opt-in tier on top of yolo_seatbelt, which is off. The runner's "
        "profile already denies writes outside the trial workspace."),

    # --- on, but cannot fire here: said before scoring, not after ------------
    "profile_prompt": (
        "ON", "no",
        "INERT (measured): profiles.resolve() returns the `qwen38` profile for the "
        "shipped pack, and that profile's prompt block is empty by design — the hook "
        "exists, nothing has been earned into it. Contributes 0 chars to the prompt."),
    "plan_review": (
        "ON", "no",
        "Plan mode is never entered: every trial runs mode='yolo'."),
    "catalog_clip": (
        "ON", "no",
        "Trims rows of the `<available_skills>` block, and skill discovery is off for "
        "both arms (CHAD_NO_SKILLS=1, see run.py) so there is no catalog to trim."),
    "structural_reindent": (
        "ON", "yes",
        "Python-only corpus, which is exactly this lever's language."),
    "ts_edit_revert": (
        "ON", "no",
        "Extends the Python revert to other parseable languages; both repos are pure "
        "Python, so `syntax_revert` covers every file the arm can edit."),
    "backend_retry": (
        "ON", "no",
        "Retries a transient 5xx from a remote backend. The engine is in-process; there "
        "is no HTTP path to fail."),
    "subagent_no_respawn": ("ON", "yes", "Fires when a capped sub-agent is re-requested."),
    "rg_replace_flag_note": (
        "ON", "yes",
        "ripgrep is on PATH and `rg -r` is a shape the model reaches for."),

    # --- on: the ordinary configuration --------------------------------------
    "verify_requires_execution": (
        "ON", "yes",
        "The done-gate's anti-spoof rung. Its `_EXECUTES_RE` carries the ported "
        "widening (PORTS.md) so a real test run is not refused."),
    "bail_nudge": ("ON", "yes", "Fires on a step with no tool call and no content."),
    "investigation_gate": ("ON", "yes", "Fires after ~6 read-only steps with no edit."),
    "verification_matrix": ("ON", "yes", "Fires after ~8 exploratory-bash steps."),
    "edit_loop_break": ("ON", "yes", "Fires after 2 consecutive edits that did not land."),
    "subagent_budget_note": ("ON", "yes", "Fires when a sub-agent hits its step cap."),
    "grep_zero_match_notice": ("ON", "yes", "Fires on any zero-match grep."),
    "syntaxgate_revert": ("ON", "yes", "Fires on an edit that breaks indentation."),
    "progress_note_rich": ("ON", "yes", "Fires on a governor relaunch."),
    "grep_filter_before_cap": ("ON", "yes", "Fires on a grep over a deep tree; django is one."),
    "repeat_coarse_tier": ("ON", "yes", "Fires on block-scale degenerate decode."),
    "edit_fail_kind": ("ON", "yes", "Fires on a no-op or unmatched edit."),
    "revert_rearm_gate": ("ON", "yes", "Fires on a clean `git checkout/reset --hard/stash`."),
    "done_spec_recheck": ("ON", "yes", "Fires before the first accepted `done` of a task turn."),
    "subagent_compact_window": ("ON", "yes", "Fires when a sub-agent's transcript compacts."),
    "syntax_revert": ("ON", "yes", "Fires on an edit that introduces any Python SyntaxError."),
    "edit_result_echo": ("ON", "yes", "Fires on every replace_lines/insert_lines result."),
    "stale_file_guard": ("ON", "yes", "Fires on a line edit against an unseen or changed file."),
    "edit_drift_warn": ("ON", "yes", "Fires when a Python edit drops a binding."),
    "write_gate": ("ON", "yes", "Fires on a whole-file write that would break the parse."),
    "broken_streak_steer": ("ON", "yes", "Fires after 2+ landed mutations leaving a file unparseable."),
    "write_diff_note": ("ON", "yes", "Fires on every overwrite of an existing file."),
    "wrapup_window": (
        "ON", "yes",
        "Needs a wall budget; run.py sets CHAD_TURN_BUDGET_S to the trial wall cap for "
        "both arms, so this fires in the final stretch of a long trial."),
    "no_think_escalation": ("ON", "yes", "Fires after 2 consecutive gen-capped steps with no tool call."),
    "done_audit": ("ON", "yes", "Fires once per turn that did real work."),
    "turn_think_budget": ("ON", "yes", "Needs CHAD_TURN_BUDGET_S, which run.py sets."),
    "hard_wrapup": ("ON", "yes", "Needs CHAD_TURN_BUDGET_S, which run.py sets."),
    "garble_never_final": ("ON", "yes", "Fires when a final answer carries tool-call markers."),
    "audit_absent_rebounce": ("ON", "yes", "Fires on a second done-audit with paths still absent."),
    "gate_ops_exempt": ("ON", "yes", "Fires on a non-read-only bash step under the investigation gate."),
    "edit_typo_match": ("ON", "yes", "Fires on the fourth edit-match rung."),
    "subagent_evidence_warn": ("ON", "yes", "Fires on a confident sub-agent report with zero tool calls."),
    "dup_result_elide": ("ON", "yes", "Fires on a repeated byte-identical read-only result."),
    "audit_churn_handoff": ("ON", "yes", "Fires when the no-empty-diff gate is about to hard-stop."),
    "bash_auto_background": (
        "ON", "yes",
        "Fires on a bash timeout. The spill file is written by chad's own process, "
        "which is outside the runner's sandbox, so the path stays readable."),
    "steer_verify_specific": ("ON", "yes", "Static prompt block; present in every render."),
    "scoped_destructive_guard": ("ON", "yes", "Fires on a recursive rm whose target is not catastrophic."),
    "late_continue_replenish": ("ON", "yes", "Needs CHAD_TURN_BUDGET_S, which run.py sets."),
    "capped_think_credit": ("ON", "yes", "Fires on a generation that ended inside <think>."),
    "compact_notice": ("ON", "yes", "Fires on compaction; a 60-step trial on django reaches it."),
    "compact_offload": ("ON", "yes", "Fires on compaction; the spill is written outside the sandbox."),
    "workspace_map": (
        "ON", "yes",
        "The repo_map digest in the prompt tail — a headline part of the surface under "
        "test. Built per workspace, so its cost lands in the arm's turn-1 numbers."),
    "env_manifest": ("ON", "yes", "Static prompt block; present in every render."),
    "session_ledger": ("ON", "yes", "Fires on every landed mutation and done bounce."),
    "bash_read_skeleton": ("ON", "yes", "Fires the first time a source file comes back."),
    "edit_miss_diagnose": ("ON", "yes", "Fires on an `old string not found`."),
    "bash_empty_diagnose": ("ON", "yes", "Fires on a bash command that printed nothing."),
    "bash_line_clip": ("ON", "yes", "Fires on an over-long output line."),
    "bash_trim_keep_failures": ("ON", "yes", "Fires on a head/tail-trimmed bash result."),
    "verify_baseline": (
        "ON", "yes",
        "Records the project's test command before the first edit. The prompt block it "
        "adds is part of the arm; the runner tells neither arm a test command."),
    "post_edit_diagnostics": (
        "ON", "yes",
        "Needs a language server. Checked before scoring: the pinned pyright resolves "
        "from the uv cache with no network (`uvx --offline --from pyright==<pin>`), and "
        "node is on PATH. The LSP runs in chad's process, outside the sandbox."),
    "grep_anchor": ("ON", "yes", "Fires on every grep with matches."),
    "read_range_footer": ("ON", "yes", "Fires on every skeleton or clipped read."),
    "edit_checkpoint": (
        "ON", "yes",
        "Snapshots into a shadow repo with its own --git-dir, so the workspace's own "
        "git state — which IS the prediction — is never touched. It supplies its own "
        "committer identity, so the runner's GIT_CONFIG_GLOBAL=/dev/null does not stop "
        "it. Cost per edit on an 11k-file tree is a result, not a reason to switch off."),
    "bash_env_guard": (
        "ON", "yes",
        "Filters credential-shaped variables out of the child environment. The runner's "
        "sandbox denies network, so this is belt and braces on a machine that runs the "
        "campaign for weeks."),
}

HEADER = ("# LEGACY-FULL lever manifest — generated by make_manifest.py, frozen by sha\n"
          "# in PREREG.md before the first scored trial. Columns:\n"
          "#   lever  state  fires_here  group  kind  reason\n"
          "# state='ON' is what run.py passes as CHAD_ENABLE. fires_here is a\n"
          "# PRE-REGISTERED PREDICTION, not a setting: 'no' means this campaign cannot\n"
          "# produce the lever's trigger, said in advance so a zero count is not\n"
          "# explained afterwards. Only a conflict with the runner turns a lever OFF.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True, help="the legacy checkout to read levers from")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tree = os.path.abspath(args.tree)
    sys.path.insert(0, os.path.join(tree, "src"))
    from chad import levers

    known = set(levers.LEVERS)
    missing = sorted(known - set(VERDICTS))
    extra = sorted(set(VERDICTS) - known)
    if missing or extra:
        for name in missing:
            print(f"no verdict for lever {name!r} (it exists in {tree})", file=sys.stderr)
        for name in extra:
            print(f"verdict for {name!r}, which the tree does not ship", file=sys.stderr)
        return 2

    sha = subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=False).stdout.strip()
    rows = []
    for name in sorted(known):
        state, fires, reason = VERDICTS[name]
        lever = levers.LEVERS[name]
        rows.append("\t".join((name, state, fires, lever.group, lever.kind, reason)))

    on = sum(1 for n in known if VERDICTS[n][0] == "ON")
    cannot_fire = sorted(n for n in known if VERDICTS[n][1] == "no")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(HEADER)
        fh.write(f"# tree: {tree}\n# tree_sha: {sha}\n")
        fh.write(f"# {len(known)} levers: {on} ON, {len(known) - on} OFF; "
                 f"{len(cannot_fire)} predicted unable to fire here "
                 f"({', '.join(cannot_fire)})\n")
        fh.write("\n".join(rows) + "\n")
    print(f"wrote {args.out}: {len(known)} levers, {on} ON, "
          f"{len(cannot_fire)} predicted inert")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
