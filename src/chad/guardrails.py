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
import os
import re

from . import levers
from .toolcall_parse import strip_think

# First-person, forward-looking leads that ANNOUNCE a next step ("Let me find where
# this is defined…") — used to detect a turn that stated intent to act but emitted no
# tool call. "let me know" is excluded below (a question to the user, not a self-directed
# action).
_INTENT_LEADS = (
    "let me ", "let's ", "let us ", "i'll ", "i will ", "i need to ", "i'm going to ",
    "i am going to ", "i should ", "i want to ", "i plan to ", "next, i", "first, i",
    "now i", "now, i",
)


def _announces_unfulfilled_action(stripped: str) -> bool:
    """True when the model's (think-stripped) content is a bare preamble that ANNOUNCES a
    next step but takes none — no tool call, no code. Keyed off an intent lead in the
    FINAL sentence so it doesn't fire on a genuine answer that merely mentions next steps.
    Conservative companion to the empty-content check in `nudge_for_no_calls`."""
    t = stripped.strip().lower().replace("\n", " ")
    if not t:
        return False
    for sep in (". ", "! ", "? "):
        if sep in t:
            t = t.rsplit(sep, 1)[-1]
    tail = t[-200:]
    if "let me know" in tail or "let us know" in tail:
        return False
    return any(lead in tail for lead in _INTENT_LEADS)

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


# A shell command that proves NOTHING about runtime behavior — a syntax/compile check,
# a byte-compile, or a --version/--help probe. These must NOT clear unverified_edit: the
# demonstrated failure is a model that "verifies" a real Django bug-fix with
# `python -c "compile(open('f').read(),'f','exec')"` → exit 0 "Syntax OK", then ships a
# patch whose FAIL_TO_PASS test (`None != 0`) still fails. Verifying means *running the
# code/tests*, not proving it parses.
_TRIVIAL_CHECK_RE = re.compile(
    r"py_compile|compileall|compile\s*\(|ast\.parse\s*\(|--version\b|--help\b"
    r"|pip[0-9.]*\s+(?:install|download|uninstall)", re.I)


def _is_trivial_check(command: str) -> bool:
    """True when a bash command is only a syntax/compile/version probe (see above)."""
    return bool(command) and bool(_TRIVIAL_CHECK_RE.search(command))


# Commands that plausibly EXECUTE the project — a test runner, an interpreter, a build
# that runs code, a script. Only these can clear unverified_edit. The demonstrated
# failure (sphinx-7440): after an edit broke the file, `sed -n '307,308p' … | cat -A`
# exited 0 with output and "verified" the edit — disarming the verify nudge, the done
# rejection AND the landing nudge at once; the patch shipped with an IndentationError.
# Display/plumbing commands (sed/cat/ls/grep/echo/find/git…) prove nothing about
# runtime behavior no matter how cleanly they exit.
_EXECUTES_RE = re.compile(
    r"(?:^|[;&|(]\s*|\bsudo\s+|\benv\s+(?:\w+=\S+\s+)*)"
    r"(?:python[0-9.]*|pytest|py\.test|tox|nox|unittest|make|cmake|ctest|cargo|go"
    r"|node|npm|npx|yarn|pnpm|deno|bun|mvn|gradlew?|ant|rake|rspec|ruby|phpunit|php"
    r"|dotnet|swift|julia|Rscript|perl|lua|java|sbt|stack|sh|bash|zsh|\./\S+)\b")


def _is_executing_command(command: str) -> bool:
    """True when a bash command plausibly runs code (see `_EXECUTES_RE`). An empty
    command (legacy callers) is trusted, preserving the old behavior for them."""
    if not command:
        return True
    return bool(_EXECUTES_RE.search(command))


def bash_result_verifies(result: str, command: str = "") -> bool:
    """A bash tool result clears the unverified-edit flag only on a clean run that
    actually exercised the code.

    Three gates, all required: (1) the result is not an error sentinel — the four
    `[`-prefixed prefixes below mean the check did NOT pass (non-zero exit, timeout,
    ctrl-c, launch failure); (2) the command is not a trivial syntax/compile/version
    probe (`_is_trivial_check`) — parsing clean is not the tests passing; (3) the
    command actually EXECUTES something (`_is_executing_command`) — a display command
    like `sed -n | cat -A` or `ls` exiting 0 is not verification (the sphinx-7440
    false-green: it disarmed every guard while the file didn't even parse)."""
    if result.startswith(("[exit", "[timed out", "[interrupted", "[failed to launch")):
        return False
    # Gates (2) and (3) together ARE the iter-2 anti-spoof fix; ablating it means
    # accepting any clean exit as verification, which is the pre-fix behavior.
    if not levers.enabled("verify_requires_execution"):
        return True
    if _is_trivial_check(command):
        return False
    if not _is_executing_command(command):
        return False
    return True


# Tools that count as real work (did_work) — includes the symbolic read/search/edit
# tools, but NOT planning/done. Kept as a named constant so the set is testable.
SUBSTANTIVE_TOOLS = ("grep", "glob", "read", "write", "edit", "bash",
                     "repo_map", "overview", "view_symbol", "find_symbol",
                     "find_refs", "replace_symbol", "insert_symbol", "rename_symbol")


# Working-tree DISCARD commands: after one of these runs cleanly, the edits the model
# made are gone and the tree is clean again. The demonstrated hole (matplotlib-20676):
# the model edited → ran pytest (clearing unverified_edit) → `git checkout .` reverted
# the file → answered in prose, and the no-empty-diff done-gate (which keys on made_edit)
# waved it through with an EMPTY diff — the exact outcome the gate exists to stop. So a
# clean revert must un-set made_edit, re-arming the gate. High-confidence forms only
# (whole-tree/path discards, hard reset, stash-save, forced clean) — a branch-creating
# `git checkout -b` or a `git stash pop`/`apply` (which RESTORE work) do not match.
_REVERT_PATTERNS = (
    re.compile(r"\bgit\s+checkout\s+(?:--(?:\s|$)|\.|HEAD\b|-f\b)"),
    re.compile(r"\bgit\s+checkout\s+\S+\.\w"),        # git checkout path/to/file.py
    re.compile(r"\bgit\s+restore\b"),
    re.compile(r"\bgit\s+reset\s+--hard\b"),
    re.compile(r"\bgit\s+stash\b(?!\s+(?:pop|apply|list|show|drop|branch))"),
    re.compile(r"\bgit\s+clean\s+-\w*f"),
)


def reverts_working_tree(command: str) -> bool:
    """True when a bash command DISCARDS working-tree edits (git checkout/restore of
    paths, reset --hard, stash-save, clean -f). Used to re-arm the no-empty-diff gate:
    a clean revert un-lands whatever the model had edited (see `update_work_flags`)."""
    return bool(command) and any(p.search(command) for p in _REVERT_PATTERNS)


# Command heads that only OBSERVE state. Used by the investigation gate: a bash step
# whose every segment starts with one of these (and redirects nothing to a file) is
# investigation; anything else — `git merge`, `apt-get install`, `mkdir`, `tar x`,
# a redirect — is ACTION and must reset the read-only streak (otherwise the gate can
# count an entire git/ops workflow as "investigation" and demand an edit at a decision
# point where there is nothing to edit yet).
_READONLY_HEADS = frozenset((
    "ls", "cat", "head", "tail", "less", "more", "grep", "egrep", "fgrep", "rg",
    "find", "file", "stat", "wc", "which", "type", "pwd", "echo", "printf", "du",
    "df", "ps", "env", "printenv", "sort", "uniq", "cut", "tr", "diff", "cmp",
    "md5sum", "sha1sum", "sha256sum", "strings", "xxd", "hexdump", "od",
    "readlink", "realpath", "basename", "dirname", "test", "[", "true", "false",
    "date", "whoami", "id", "uname", "hostname", "tree", "awk", "sed", "jq",
    "column", "nl", "tac", "sleep",
))
_READONLY_GIT_SUBS = frozenset((
    "log", "status", "diff", "show", "describe", "rev-parse", "ls-files",
    "ls-remote", "ls-tree", "blame", "reflog", "shortlog", "grep", "cat-file",
    "rev-list", "name-rev", "var", "count-objects",
))
# Harmless stderr plumbing stripped before the "any redirect ⇒ mutating" check.
_STDERR_REDIR_RE = re.compile(r"2>\s*&1|2>\s*/dev/null|&>\s*/dev/null|>\s*/dev/null")
_SEGMENT_SPLIT_RE = re.compile(r"\|\||&&|[;|]")


def is_readonly_bash(command: str) -> bool:
    """Conservatively true when a bash command only OBSERVES state: every pipeline
    segment's head is in the read-only allowlist (`cd`/`env`-style prefixes skipped;
    `git <readonly-sub>` allowed; `sed -i` excluded) and nothing is redirected to a
    file. Anything unrecognized is NOT read-only — for the investigation gate that
    is the safe direction (an ops step wrongly counted as investigation harasses the
    model; a read wrongly counted as action merely delays the gate)."""
    if not command.strip():
        return True
    cleaned = _STDERR_REDIR_RE.sub("", command)
    if ">" in cleaned or "<(" in cleaned:
        return False
    for seg in _SEGMENT_SPLIT_RE.split(cleaned):
        words = seg.strip().split()
        # skip wrappers/prefixes that don't decide the verb
        while words and (words[0] == "cd" or "=" in words[0] or words[0] in
                         ("env", "sudo", "command", "builtin", "time", "nice")):
            if words[0] == "cd":  # `cd x && grep …`: drop `cd` + its argument
                words = words[2:]
            else:
                words = words[1:]
        if not words:
            continue
        head = words[0]
        if head == "git":
            if len(words) < 2 or words[1] not in _READONLY_GIT_SUBS:
                return False
        elif head not in _READONLY_HEADS:
            return False
        elif head == "sed" and any(w.startswith("-i") or w.startswith("--in-place")
                                   for w in words[1:]):
            return False
    return True


def audit_absent_paths(task_text, paths=None):
    """The task-mentioned paths that do not currently exist — recorded at the first
    done-audit bounce so `audit_rebounce` can later check exactly these (and nothing
    else) at accept time."""
    if paths is None:
        paths = audit_extract_paths(task_text)
    return [p for p in paths if not os.path.exists(p)]


def audit_rebounce(absent_paths, runway_s):
    """The one final done-audit bounce (levers.audit_absent_rebounce), or None to
    accept. Fires only when a path the FIRST audit already reported absent is STILL
    absent and there is real runway left — the miss it closes is a task-named
    deliverable the audit stat'ed as absent, where the acceptance promise forced the
    done through and the file was never written. Caller owns the 2-bounce cap and the
    still-once-per-turn latch semantics."""
    if not levers.enabled("audit_absent_rebounce") or not absent_paths:
        return None
    if runway_s is not None and runway_s <= AUDIT_MIN_RUNWAY_S:
        return None
    still = [p for p in absent_paths if not os.path.exists(p)]
    if not still:
        return None
    return ("[final audit — the task statement names path(s) that still do not "
            "exist:\n" + "\n".join(f"  - {p}" for p in still) +
            "\nIf the task requires them, create them at exactly these paths now "
            "(then re-verify). If the task genuinely wants them absent, call done "
            "again. Your next `done` will be accepted unconditionally.]")


def update_work_flags(name, args, result, did_work, made_edit, unverified_edit):
    """Update the (did_work, made_edit, unverified_edit) guardrail flags after one
    tool result; returns the new triple. A substantive tool counts as real work; a
    successful edit/write (text or symbolic) sets made_edit and — unless it's a pure
    prose/doc file — arms unverified_edit; a bash run that didn't error/timeout/interrupt
    clears it (a failing test keeps it dirty so the model re-runs). A bash command that
    cleanly REVERTS the working tree (`reverts_working_tree`) un-sets made_edit AND
    unverified_edit — the edit is gone, so the done-gate must treat the turn as having
    landed nothing (matplotlib-20676: revert-then-prose shipped an empty diff)."""
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
    elif name == "bash":
        cmd = str(args.get("command", ""))
        errored = result.startswith(
            ("[exit", "[timed out", "[interrupted", "[failed to launch"))
        if (levers.enabled("revert_rearm_gate")
                and reverts_working_tree(cmd) and not errored):
            made_edit = False
            unverified_edit = False
        elif bash_result_verifies(result, cmd):
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


# Injected once before the first `done` is accepted (levers.done_spec_recheck). The
# hidden verifier scores container end-state, so a model that "finished" but wrote its
# output to the wrong path or in the wrong shape fails silently with budget to spare —
# the single largest wrong-done bucket (TB2 sam-cell-seg wrote directories where files
# were required; bn-fit-modify passed 8/9 on a format mismatch). This asks the model to
# self-check deliverables against the literal task text before it stops.
DONE_SPEC_RECHECK = (
    "[before you finish — one verification pass, read-only. Re-read the ORIGINAL task "
    "above and list every concrete deliverable it names (each output file, its exact "
    "path, and the exact format/content it must contain). For EACH one, CHECK it now with "
    "read-only commands — `ls` the path, `cat`/`head` the contents — and confirm it "
    "exists, is a file (not a directory), and matches what was asked (a valid answer at "
    "the wrong path or in the wrong format scores zero). Do NOT edit or rewrite a "
    "deliverable that already checks out: correct output left alone stays correct, and "
    "re-editing working files only risks breaking them. Call `done` the moment every "
    "deliverable verifies. ONLY if a check actually fails — something missing, misplaced, "
    "or malformed — make the single targeted fix it needs, then call done.]"
)


def done_spec_recheck(did_work, unverified_edit, recheck_done, read_only_intent):
    """Whether to inject the deliverable recheck before accepting `done`. Fires at most
    once per turn (recheck_done), only on a real action turn that produced work and is
    not an explain-only ask, and only after the empty/verify gates have cleared (so it
    doesn't stack with them). Pure/testable; the lever gate is checked at the call site."""
    return (not recheck_done and did_work and not unverified_edit
            and not read_only_intent)


# The deliverable recheck (done_spec_recheck) is a verify-then-done pass; a genuine fix
# it turns up is one or two targeted edits (wrong path -> move, wrong shape -> one
# rewrite). More landed edits than this means the model is re-editing already-correct
# output into a thrash instead of finishing — the poly_two_bucket regression: 3 edits /
# 16k gen tokens / 621s (7x baseline wall) after the recheck fired on an answer that had
# already passed the verify gate. Cap the post-recheck fix edits and keep the result.
RECHECK_MAX_FIX_EDITS = 2


def recheck_spiral(post_recheck_edits):
    """Whether the deliverable recheck has driven more landed edits than a real fix
    should take (> RECHECK_MAX_FIX_EDITS), i.e. the model is thrashing on already-correct
    work rather than verifying-and-finishing. The caller ends the turn keeping the result
    that was ship-ready when the recheck fired. Pure/testable."""
    return post_recheck_edits > RECHECK_MAX_FIX_EDITS


# --- Done-audit ----------------------------------------------------------
#
# The TB2.1 autopsy's largest fail bucket (20/43) was `done`s whose final message claimed
# SPECIFIC verification the hidden checker then rejected — kv-store-grpc done at 84s of
# 900 ("confirmed via socket test"), overfull-hbox at 205/750 ("zero overfull warnings").
# The model runs *a* check, but a weaker predicate than the task's own wording, and the
# generic done_spec_recheck steer above was ON for every one of them. The audit is the
# task-grounded successor: quote the task statement's own requirement lines back (quoting
# beats paraphrase — it re-anchors on the real predicate and is deterministic), plus
# stat-level facts about every path the task names. One bounce per turn, ever; the next
# `done` is accepted unconditionally (the 070 anti-spiral lesson).

AUDIT_MIN_RUNWAY_S = 120.0   # never start an audit with less wall runway than this
AUDIT_MAX_PATHS = 8          # paths statted/quoted in the steer
AUDIT_MAX_REQ_LINES = 8      # requirement lines quoted in the steer
AUDIT_LINE_CHARS = 400       # per quoted line — several corpus tasks are ONE long
                             # paragraph-line (overfull-hbox 366 chars, mteb-retrieve
                             # 398) whose acceptance criteria sit past 240

# Absolute paths (/app/server.py). The lookbehind rejects s/…/ regex bodies, URL //,
# :/ and expression contexts (argv[1]/2, f(x)/2) so none of those read as deliverables.
_AUDIT_ABS_PATH_RE = re.compile(r"(?<![\w./:\])])/(?:[\w.+~-]+/)*[\w.+~-]*\w")
# Bare filenames with a known file extension (main.tex, expected.csv) — an extension
# whitelist, because `json.load`/`re.sub`-style attribute access and run-on sentences
# ("videos.Your") are indistinguishable from filenames by shape alone. A missed exotic
# extension just means one less statted fact, never a wrong one. Name >= 2 chars keeps
# e.g./i.e. out; a leading alnum keeps template fragments ({x}_pb2.py) out.
_AUDIT_BARE_EXTS = (
    "py|js|ts|c|h|cpp|cc|hpp|rs|go|java|rb|pl|r|sh|bash|html|htm|css|json|jsonl|csv|"
    "tsv|txt|md|rst|tex|toml|yaml|yml|xml|ini|cfg|conf|sql|db|sqlite|proto|ttl|sparql|"
    "scm|vim|fasta|ipynb|log|lock|gz|tar|zip|png|jpg|jpeg|svg|mp4|wav|pdf|bin|dat|env|"
    "service|socket|patch|diff")
_AUDIT_BARE_FILE_RE = re.compile(
    r"\b([A-Za-z0-9][\w-]+\.(?:" + _AUDIT_BARE_EXTS + r"))\b")
# Imperative / acceptance-criteria vocabulary that marks a requirement line.
_AUDIT_REQ_RE = re.compile(
    r"\b(must|should|write|writes|written|save|saves|saved|creat(?:e|es|ed)|exactly|"
    r"format|ensur(?:e|es)|generat(?:e|es)|output|outputs|place|install|match|matches|"
    r"requir(?:e|es|ed|ements)|do not|don'?t|only|byte-for-byte|run|runs|running|keep)"
    r"\b", re.IGNORECASE)
# Structural cue: a numbered/bulleted line in a task statement is almost always a
# requirement item, keywords or not (kv-store-grpc "5. Run the server.py file…",
# sparql-university's three numbered criteria).
_AUDIT_LIST_ITEM_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+\S")

DONE_AUDIT_DEMAND = (
    "Re-verify each requirement above against the ACTUAL files/system with fresh "
    "commands — do not rely on your memory of earlier checks, and check the task's "
    "exact wording (right path, right format, right behavior), not a weaker version "
    "of it. Fix anything that fails. Then call done again; it will be accepted.")


def audit_task_text(user_text: str) -> str:
    """The pristine task statement for audit extraction. cli.py appends harness notes to
    the task on auto-continue relaunches (the banked progress note) and on the review
    pass (REVIEW_PASS_PROMPT); their paths and `$ command` lines are session noise, not
    task requirements — Part B's build-pmars on.1 bounce quoted `/bin/sh` and
    `/dev/null` straight out of the note. Both appendices open with a fixed marker;
    truncate at the first one found."""
    for marker in ("\n\n[" + PROGRESS_NOTE_HEADER, REVIEW_PASS_PROMPT):
        i = user_text.find(marker)
        if i != -1:
            user_text = user_text[:i]
    return user_text


def audit_extract_paths(task_text):
    """Paths the task statement names, in document order, deduped, capped. Absolute
    paths first-class; bare `name.ext` mentions are included unless they are just the
    basename of an absolute path already found (input.csv vs /app/input.csv)."""
    paths = []
    for m in _AUDIT_ABS_PATH_RE.finditer(task_text):
        p = m.group(0)
        # A real path names something: all-digit-and-punctuation matches (a stray "/2")
        # are arithmetic, not deliverables.
        if p not in paths and re.search(r"[A-Za-z]", p):
            paths.append(p)
    for m in _AUDIT_BARE_FILE_RE.finditer(task_text):
        p = m.group(1)
        # Substring dedupe, not just basename: /app/x.tar.gz already covers a bare
        # "x.tar" the whitelist re-finds inside it.
        if p not in paths and not any(p in a for a in paths):
            paths.append(p)
    return paths[:AUDIT_MAX_PATHS]


def audit_requirement_lines(task_text, paths):
    """The task statement's own requirement lines: any line that names an extracted path
    or uses imperative/acceptance vocabulary. Quoted verbatim (truncated per line) so the
    steer re-anchors the model on the real predicate, not a paraphrase. When over the
    cap, path-bearing lines win — they are the concrete deliverables."""
    candidates = []  # (doc_index, line, names_a_path)
    for i, raw in enumerate(task_text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        if len(line) > AUDIT_LINE_CHARS:
            line = line[:AUDIT_LINE_CHARS] + "…"
        if any(p in raw for p in paths):
            candidates.append((i, line, True))
        elif _AUDIT_REQ_RE.search(line) or _AUDIT_LIST_ITEM_RE.match(raw):
            candidates.append((i, line, False))
    take = [c for c in candidates if c[2]][:AUDIT_MAX_REQ_LINES]
    take += [c for c in candidates if not c[2]][:AUDIT_MAX_REQ_LINES - len(take)]
    return [line for _, line, _ in sorted(take)]


def audit_path_facts(paths, turn_start_epoch):
    """Deterministic stat-level facts, one per path — pure fact-gathering, no judgment.
    Phrasing stays NEUTRAL (absence is evidence to weigh, not an assertion of failure:
    a task may legitimately delete a path it mentions). No content checks — content is
    the model's job in the re-verify."""
    facts = []
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            facts.append(f"{p}: mentioned in the task, currently absent")
            continue
        if os.path.isdir(p):
            facts.append(f"{p}: exists (directory)")
            continue
        # -1s slack for filesystem mtime granularity vs the epoch snapshot.
        when = ("modified during this turn" if st.st_mtime >= turn_start_epoch - 1
                else "not modified during this turn")
        facts.append(f"{p}: exists, {st.st_size} bytes, {when}")
    return facts


def done_audit(task_text, turn_state):
    """The one-shot done-audit steer, or None to accept the done untouched. Fires only
    with wall runway to spare — an audit that pushes the task into the wall converts a
    wrong-done into a wall-death (same score, worse autopsy) — and only when the task
    text yields something concrete to audit against. The caller owns the once-per-turn
    latch, the action-task/plan-mode scoping, and the message append.

    turn_state: turn_start_epoch (time.time() at turn start, for mtime facts), wall_s
    (elapsed), wall_budget_s (None when no budget is configured — runway then unlimited),
    step_walls (recent per-step wall seconds, for the runway margin)."""
    budget = turn_state.get("wall_budget_s")
    if budget:
        remaining = budget - turn_state.get("wall_s", 0.0)
        recent = (turn_state.get("step_walls") or [])[-5:]
        mean_step = sum(recent) / len(recent) if recent else 0.0
        if remaining <= max(AUDIT_MIN_RUNWAY_S, 2.0 * mean_step):
            return None
    paths = audit_extract_paths(task_text)
    req_lines = audit_requirement_lines(task_text, paths)
    if not req_lines:
        return None
    # The acceptance promise must stay truthful: with the absent-path re-bounce
    # armed (levers.audit_absent_rebounce) one further existence-only check may run,
    # so say so; otherwise keep the original unconditional promise (the anti-spiral
    # latch depends on the model believing it).
    if levers.enabled("audit_absent_rebounce"):
        promise = ("your NEXT `done` will be accepted after at most one further "
                   "check that the paths below exist.")
    else:
        promise = "your NEXT `done` will be accepted without further audit."
    parts = ["[done-audit — one-time check before this `done` is accepted; " + promise
             + " The hidden grader checks "
             "the task's OWN requirements, not the checks you happened to run. From the "
             "task statement:"]
    parts += [f"  > {ln}" for ln in req_lines]
    facts = audit_path_facts(paths, turn_state.get("turn_start_epoch", 0.0))
    if facts:
        parts.append("Facts about paths the task mentions (checked on disk just now — "
                     "evidence to weigh, not verdicts; a path the task removes may "
                     "rightly be absent):")
        parts += [f"  - {f}" for f in facts]
    parts.append(DONE_AUDIT_DEMAND + "]")
    return "\n".join(parts)


GARBLE_NUDGE_CAP = 6  # per-turn garble re-nudges (own counter, separate from truncation)

# Placeholder that replaces a garbled assistant message body once the NEXT step
# garbles too — the wrong dialect must not stay in context as a few-shot example
# (repeated garbles in a row each condition on the last). Costs one prefix-cache
# invalidation on a rare path.
GARBLE_SCRUBBED = "[a malformed tool call was removed here — it did not run]"

# Appended to the garble nudge from the 2nd consecutive garbled step: the model is
# clearly stuck in a wrong dialect, so show the contract instead of describing it.
TOOLCALL_EXEMPLAR = (
    "\nThis is the ONLY valid format — copy its shape exactly:\n"
    "<tool_call>\n"
    '{"name": "bash", "arguments": {"command": "ls /app"}}\n'
    "</tool_call>\n"
    "No <function=...>, no <parameter=...> tags, nothing else inside the block.")


def nudge_for_no_calls(text, hit_cap, made_edit, unverified_edit, read_only_intent,
                       action_task, truncation_nudges, answer_nudges, verify_nudges,
                       open_tool_call, garbled_call=False, garble_nudges=0,
                       consecutive_garbles=0):
    """Pick the nudge for a step that produced NO tool call, in the original priority
    order: (1) GARBLE/TRUNCATED — a tool-call attempt that parsed to nothing, or the
    token cap hit mid-thought, so it isn't an answer; (2) ANSWERED ON PAPER —
    produced/described code but never applied it; (3) UNVERIFIED EDIT — edited but
    never ran the check. Pure: returns (kind, nudge_text) or (None, None); the caller
    bumps the matching counter and appends the nudge. `kind` is one of 'garble' /
    'truncated' / 'no-edit' / 'unverified-edit'. (open_tool_call is run_turn's
    _has_open_tool_call(text): an unbalanced <tool_call>/<function> that parsed to
    nothing.)"""
    has_code = "```" in text
    # A tool-call attempt that parsed to zero calls is never a final answer, whether
    # or not the token cap was hit. Causes: (a) hit_cap — a `write` whose content
    # overran the budget (guide it to write in bounded pieces); (b) an unclosed or
    # closed-but-unparseable block — sampling glitch, premature EOS, or a slide into
    # a foreign XML tool-call dialect. With the garble_never_final lever these use
    # their OWN counter (garble_nudges, cap 6) — a shared `truncation_nudges < 2` lets
    # an unrelated step-0 cap-hit spend the garble budget, after which a run of garbles
    # can be ACCEPTED as the final answer with most of the wall budget still left. This
    # fires before the bare-stall branch so a garbled call isn't misread as an empty
    # stall.
    if open_tool_call and hit_cap:
        # A call cut off AT the token cap is a length problem (a too-long `write`),
        # not a dialect garble — it stays in the truncation family/counter.
        if truncation_nudges < 2:
            nudge = ("[your tool call was cut off at the length limit — the "
                     "content was too long to emit in one call. Do NOT retry it "
                     "whole. Create the file with `write` using only the FIRST "
                     "portion of the content, then append the rest with one or "
                     "more `edit` calls. Emit one complete tool call at a time.]")
            return "truncated", nudge
    elif open_tool_call or garbled_call:
        if levers.enabled("garble_never_final"):
            allowed, kind = garble_nudges < GARBLE_NUDGE_CAP, "garble"
        else:  # legacy (ablation arm): shared counter, kind counted as truncation
            allowed, kind = truncation_nudges < 2, "truncated"
        if allowed:
            if open_tool_call:
                nudge = ("[your last tool call was malformed and did not run — it opened a "
                         "<tool_call> (or <function=…>) that was never properly closed, so no "
                         "tool executed and nothing happened. Re-emit it now as ONE complete, "
                         "well-formed <tool_call> block with valid JSON arguments.]")
            else:
                # garbled_call: the block WAS closed but nothing inside parsed — mixed
                # JSON/XML dialects, invalid JSON the repair pass couldn't reconstruct,
                # etc. Without this branch the garble is accepted as a final answer —
                # e.g. a `{"name": "bash", … </parameter></function></tool_call>` mash
                # that ends the task with budget still left.
                nudge = ("[your last tool call was malformed and did not run — the "
                         "<tool_call> block did not contain one valid JSON object, so no "
                         "tool executed and nothing happened. Re-emit it now as ONE "
                         "complete <tool_call> block: a single JSON object with \"name\" "
                         "and \"arguments\", no XML tags inside.]")
            if kind == "garble" and consecutive_garbles >= 2:
                nudge = nudge[:-1] + TOOLCALL_EXEMPLAR + "]"
            return kind, nudge
    if hit_cap and truncation_nudges < 2:
        # Cap hit but the call (if any) was balanced — a plain mid-thought truncation.
        nudge = ("[your reply was cut off at the length limit before you "
                 "called any tool. Do NOT re-paste what you already wrote. "
                 "Take the next single concrete action now as a real "
                 "<tool_call> — e.g. write the file — one tool at a time.]")
        return "truncated", nudge
    # Bare stall: the turn produced no tool call and no real content — the model
    # reasoned (or not) and then STOPPED, either with empty content after </think> or
    # with a preamble that only announces a next step ("Let me find where this is
    # defined…"). That is never a valid final answer. Observed on the llama.cpp backend,
    # where the served quant emits EOS right after the think-block close, before the tool
    # call the MLX path would go on to produce — but the check is backend-agnostic (a
    # premature stop from any engine lands here). Bounded by answer_nudges so a model
    # that keeps stalling still terminates. read_only intent is exempt (an explain-only
    # ask is allowed to answer briefly), matching the answered-on-paper branch below.
    stripped = strip_think(text).strip()
    if levers.enabled("bail_nudge") \
            and (not read_only_intent) and not made_edit and answer_nudges < 2 \
            and ((not stripped) or _announces_unfulfilled_action(stripped)):
        nudge = ("[you stopped after thinking without taking any action — no tool call "
                 "and no answer. Do not stop here. Emit your next concrete step now as a "
                 "real <tool_call>: grep/read to locate the code, then edit/write to "
                 "change it, then run the check. One tool call at a time.]")
        return "no-edit", nudge
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


def investigation_gate(readonly_streak, made_edit, gate_nudges, threshold=6):
    """Fire a one-shot steering nudge when the model has spent a run of purely read-only
    steps (grep/read/search — no edit landed) and clearly has enough context to act. The
    demonstrated failure: it named the exact one-line fix at step 0, then grep-looped for
    "a better fix" into the step cap and shipped an EMPTY patch; and, separately, it
    delegated the same read-only sub-agent twice, both capping out, without ever editing.
    This converts investigation into an edit before the loop-abort / step-cap kills the
    turn with nothing applied. Returns nudge text or None. Bounded by gate_nudges."""
    if not levers.enabled("investigation_gate"):
        return None
    if made_edit or gate_nudges >= 2 or readonly_streak < threshold:
        return None
    return (f"[you've spent {readonly_streak} steps investigating and applied no edit — "
            "you have enough context now. STOP searching. Make your single highest-value "
            "edit with edit/write/replace_symbol, then run the project's real test to "
            "verify it. If you already know the fix, apply it this step.]")


def edit_failed_to_land(result: str) -> bool:
    """True when an edit/symbolic-edit tool result means the change did NOT apply — a
    no-op (old==new / replacement leaves file unchanged), an unmatched `old` string, or
    an ambiguous match. Used to count consecutive dead edits so the harness can escalate
    instead of letting the model re-emit near-identical edits until it burns out (the
    demonstrated failure: it had the correct fix, reverted it, then looped on a no-op edit
    for its remaining steps and never called done)."""
    return result.startswith((
        "[no-op edit", "[old string not found", "[old string appears",
        "[old string matches", "[no such file", "[edit rejected", "[write rejected"))


def edit_fail_kind(result: str) -> str | None:
    """Classify WHY an edit failed to land, so the loop-break nudge can prescribe the
    right fix. 'noop' = old and new were identical (the model pasted the same code in
    both fields — it read the code but never changed it; telling it to "re-read and paste
    verbatim" is exactly wrong, that's what it already did). 'nomatch' = `old` didn't
    match / was ambiguous (re-read IS the right fix). None = the result isn't a failure.
    The demonstrated conflation (pytest-10356): 2/3 reps died on repeated old==new no-ops
    while being told to re-read and paste verbatim."""
    if result.startswith("[no-op edit"):
        return "noop"
    if result.startswith("[write rejected"):
        # The write gate refused the content itself: the syntax error is in
        # what the model SENT, so neither re-reading the file nor switching edit tools
        # helps — it must fix its own content.
        return "badsyntax"
    if result.startswith("[edit rejected"):
        # Three reverted-edit shapes, each with its own remedy: a structural
        # parse break (stop patching fragments → whole statement / replace_symbol), a
        # stale file (the reject already echoed fresh numbers → just re-send with them),
        # and an indentation break (stop hand-indenting → switch tools).
        if "unparseable" in result:
            return "structure"
        if "changed on disk" in result:
            return "stale"
        return "indent"     # syntaxgate reverted an indentation break — re-quoting won't help
    if edit_failed_to_land(result):
        return "nomatch"
    return None


_NOMATCH_BREAK = (
    "[your last edits changed nothing — your `old`/target text does not match "
    "the file as it is now. Do NOT re-try more variations. First `read` the exact "
    "current lines (or `view_symbol`), then either copy that text verbatim into "
    "`old`, or replace the entire enclosing function with `replace_symbol`. Land "
    "one real change, then run the test.]")
_NOOP_BREAK = (
    "[your last edits were NO-OPS: your `old` and `new` are identical, so nothing "
    "changed. Re-reading and pasting the same lines again will not help — that is what "
    "just failed. Decide the ACTUAL change: `old` = the current lines, `new` = those "
    "lines MODIFIED to fix the bug (they must differ). If you're unsure what to change, "
    "state the fix in one sentence first, then make `new` reflect it. Land one real "
    "change, then run the test.]")
_INDENT_BREAK = (
    "[STOP hand-indenting: your last edits were REJECTED for breaking indentation, and "
    "re-sending edit/replace_lines with different whitespace is exactly the loop that keeps "
    "failing — a small model cannot reliably re-transcribe indentation. Switch tools: use "
    "`replace_symbol` to rewrite the ENTIRE enclosing function (send the whole function; "
    "its indentation is handled for you), or `insert_lines` to add a single line at a read "
    "line-number. Do NOT send another hand-indented edit.]")
_STRUCTURE_BREAK = (
    "[STOP patching fragments: your last edits were REJECTED because they would leave the "
    "file unparseable — you are cutting into a multi-line structure (a def signature, "
    "call, or literal), and re-trying slightly different line ranges is the loop that "
    "keeps failing. Make ONE whole-unit change instead: `replace_symbol` with the "
    "complete new function, or a single `replace_lines` covering the ENTIRE statement "
    "(the reject message printed its exact line span — use those numbers). Never edit "
    "part of a multi-line statement.]")
_BADSYNTAX_BREAK = (
    "[STOP re-sending broken content: your last writes were REJECTED because the content "
    "you sent does not parse — the error is inside YOUR content, not the file on disk. "
    "Re-reading the file or switching tools will not help. Look at the exact line the "
    "reject message quoted, fix that syntax error in your draft, and re-send the "
    "COMPLETE corrected file in one write.]")


def edit_loop_break(noop_edit_streak, break_nudges, kind=None):
    """After ~2 consecutive edits that failed to land (see `edit_failed_to_land`), stop
    the model re-trying variations of a broken edit. `kind` (from `edit_fail_kind` on the
    latest failure) tailors the remedy: 'noop' → the change was empty, tell it to make
    `new` actually differ; anything else → the `old` text doesn't match, tell it to
    re-read and paste verbatim / use `replace_symbol`. One-shot-ish (bounded by
    break_nudges). Returns nudge text or None."""
    if not levers.enabled("edit_loop_break"):
        return None
    if noop_edit_streak < 2 or break_nudges >= 2:
        return None
    # Ablating the classifier means every failure gets the nomatch remedy, which is the
    # pre-iter-3 conflation: a model that pasted `old == new` is told to go re-read and
    # paste verbatim — precisely what it just did.
    if not levers.enabled("edit_fail_kind"):
        kind = None
    if kind == "noop":
        return _NOOP_BREAK
    if kind == "indent":
        return _INDENT_BREAK
    if kind == "structure":
        return _STRUCTURE_BREAK
    if kind == "badsyntax":
        return _BADSYNTAX_BREAK
    return _NOMATCH_BREAK


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


# --- soft think-cap -----------------------------------------------------
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
# opt-in capability trade), this guard only fires on output that is already
# garbage, so it is ON by default (CHAD_NO_REPEAT_GUARD=1 disables). run_turn checks the
# generation's tail every few tokens; on a hit it stops the step (prefix-safe — the
# generated tokens are already in the KV cache) and nudges the model out of the loop.
REPEAT_TAIL_CHARS = 2048    # window that must be fully periodic — long enough that no
                            # legitimate prose/code run trips it, short enough to fire
                            # a few hundred tokens into a runaway, not 8k tokens in
REPEAT_MAX_PERIOD = 256     # unit ≤ this ⇒ the window holds ≥ 8 repeats
# Coarse tier: catches paragraph/block-scale loops the fine tier's 256-char period can't
# see. The demonstrated miss (django-14404): a ~2–3KB reasoning paragraph repeated ~90×
# across five dead 8192-token completions — period far over 256, so the fine detector
# never fired and the loop burned the whole turn's budget. A much larger window with a
# proportionally larger period demands ≥4 exact repeats before firing, so it stays clear
# of legitimate long output (which is never 12KB of an identically-repeated block).
REPEAT_COARSE_TAIL_CHARS = 12288
REPEAT_COARSE_MAX_PERIOD = 3072


def _is_periodic_tail(text: str, tail: int, max_period: int) -> bool:
    """True when the last `tail` chars of `text` are one unit of length ≤ `max_period`
    repeated end-to-end. The smallest period comes from the KMP prefix function
    (p = len - f[-1] ⇒ s[i] == s[i-p] for all i ≥ p, regardless of where the window cuts
    into the unit — a plain `(s+s).find(s)` doubling test would only catch periods that
    divide the window exactly)."""
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


def degenerate_tail(text: str, tail: int = REPEAT_TAIL_CHARS,
                    max_period: int = REPEAT_MAX_PERIOD) -> bool:
    """True when `text`'s tail is one unit repeated end-to-end — the degenerate-decode
    signature — at either of two scales: a short unit in a 2KB window (the token-loop
    case) OR a large block (paragraph/tool-call-scale) in a 12KB window (the reasoning-
    loop case the fine tier is blind to). ~0.5ms for the fine window, ~3ms for the coarse
    one, of pure Python; run_turn calls it every 16 tokens, i.e. a few times per second
    against a ~25ms/token decode, so the cost is noise."""
    if _is_periodic_tail(text, tail, max_period):
        return True
    if not levers.enabled("repeat_coarse_tier"):
        return False
    return _is_periodic_tail(text, REPEAT_COARSE_TAIL_CHARS, REPEAT_COARSE_MAX_PERIOD)


REPEAT_STOP_NUDGE = (
    "[your output degenerated into repeating the same text over and over; it was cut "
    "off. Do not continue that repetition. Give your final answer concisely, or make "
    "the next tool call, now.]")


def repeat_stop_abort(repeat_stops: int) -> bool:
    """Abort the turn after the 3rd repetition cut-off — the nudge isn't breaking the
    decode loop, and each retry costs another stall. Mirrors loop_should_abort."""
    return repeat_stops >= 3


# --- runaway-turn governor ----------------------------------------------
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
Progress is consumed (reset to False) once any band is crossed, so the next
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
# Read/search tools whose target the note records as "already examined" so a relaunch
# doesn't re-walk the tree it already mapped (the demonstrated leak: django-14007/-14404
# reps burned their whole budget re-exploring because the note carried only edited files).
_READ_TOOLS = ("read", "grep", "glob", "view_symbol", "find_symbol", "find_refs",
               "repo_map", "overview")
_ERROR_PREFIXES = ("[exit", "[timed out", "[failed to launch", "[tool error", "[denied")
# Lines in a tool result worth carrying as the failing-test signature — the concrete
# assertion/exception the next attempt must make pass, not just "[exit 1]".
_FAILURE_RE = re.compile(
    r"^(?:E\s|.*\b(?:Error|Exception|assert|FAILED|AssertionError)\b)", re.I)

_TOOLCALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)


def _last_hypothesis(content: str) -> str | None:
    """The model's own prose from one assistant turn, with the <think> block and any
    <tool_call> blocks stripped — its working diagnosis. Returns None when there's no
    substantive prose (a pure tool-call turn) or when the prose has degenerated into a
    repetition loop (carrying that forward would only re-seed the loop)."""
    prose = _TOOLCALL_BLOCK_RE.sub("", strip_think(content)).strip()
    if len(prose) < 40 or degenerate_tail(prose):
        return None
    return prose


# First line of every progress note. Also the marker audit_task_text keys on to strip
# the note from a relaunched turn's user text (cli.py wraps it as `{task}\n\n[{note}]`),
# so a wording change here must keep the two in sync — hence the constant.
PROGRESS_NOTE_HEADER = (
    "Progress so far (auto-summarized — the previous attempt ran out of budget):")


# Matches a hypothesis that ASSERTS the work is already finished (the poison a
# rejected-done relaunch must not inherit as its leading "fact"), NOT a diagnosis that
# merely contains a success-ish word. Every completion word is anchored to a subject or
# to "already": a bare `build\s+is\s` matched "reBUILD IS failing", and a standalone
# `verified`/`successful` matched "SUCCESSFULLY reproduced the crash" — both real
# diagnoses, dropped on the exact path meant to preserve them (eng-review outside-voice
# F1). \b on the subjects also stops "build" from matching inside "rebuild".
_COMPLETION_CLAIM_RE = re.compile(
    r"(already\s+(?:built|complete|done|installed|present|correct|verified|passe?[sd])"
    r"|\b(?:build|task|project|everything|it)\s+is\s+"
    r"(?:complete|done|built|ready|correct|passing)"
    r"|task\s+is\s+(?:done|complete))", re.IGNORECASE)


def progress_note(messages, max_lines: int = 24, rejected_claim: str | None = None) -> str:
    """Synthesize a ≤`max_lines` progress note from the transcript with NO model call, so
    a hard-stopped turn can seed a fresh relaunch (sheds the ramble AND the huge prefill
    the stuck model was dragging around). Deterministic: pulls the model's last working
    hypothesis, the files edited/examined and commands run (from the assistant turns' own
    <tool_call> blocks via parse_tool_calls), and the last failing-test signature. Prefer
    facts the executor CANNOT reconstruct from a clean context — the diagnosis it reached,
    what it already tried, and what failed last. The old note carried only file names, so
    a relaunch re-derived the whole investigation and re-spent the budget it was meant to
    save (django-14007/-14404: the correct fix was stated in prose, then lost).

    `rejected_claim` (plan 107 follow-up — the build-pov-ray poisoning loop): when the
    turn ended via a REJECTED completion claim (done / final answer blocked by the
    no-empty-diff gate), pass the claim text. The note then (a) leads with an explicit
    warning that the claim was rejected, and (b) drops a working hypothesis that itself
    asserts completion — otherwise every fresh relaunch inherits "the build is already
    complete and verified" as its leading fact, re-confirms it, and re-dones (observed
    6 relaunches in a row, task 0/1 with 86% of the wall unused)."""
    from .toolcall_parse import parse_tool_calls
    edited, commands, examined = [], [], []
    last_error = None
    hypothesis = None
    for m in messages:
        role = m.get("role")
        content = m.get("content", "") or ""
        if role == "assistant":
            h = _last_hypothesis(content)
            if h:
                hypothesis = h  # keep the most recent substantive reasoning
            for name, args in parse_tool_calls(content):
                if name == "bash":
                    cmd = str(args.get("command", "")).strip()
                    if cmd and cmd not in commands:
                        commands.append(cmd)
                elif name in _EDIT_TOOLS:
                    p = str(args.get("path", "")).strip()
                    if p and p not in edited:
                        edited.append(p)
                elif name in _READ_TOOLS:
                    tgt = str(args.get("path") or args.get("pattern")
                              or args.get("name") or args.get("symbol") or "").strip()
                    if tgt and tgt not in examined:
                        examined.append(tgt)
        elif role == "tool":
            if content.startswith(_ERROR_PREFIXES) or "Traceback" in content:
                last_error = content
    # Ablating `progress_note_rich` reverts to the pre-iter-3 note: file names and
    # commands only. Everything the executor cannot reconstruct from a clean context —
    # the diagnosis, the failing signature, what was already looked at — is what the
    # lever adds, and therefore what its delta measures.
    rich = levers.enabled("progress_note_rich")
    lines = [PROGRESS_NOTE_HEADER]
    if rejected_claim:
        lines.append(
            "WARNING: the previous attempt ended by CLAIMING the task was complete "
            f"(\"{rejected_claim.strip()[:160]}\") but the claim was REJECTED — it had "
            "landed no verified change. Treat every success statement below as "
            "unverified and likely wrong: re-verify against the task statement with "
            "fresh commands and MAKE the required change before calling done.")
        # A hypothesis that itself asserts completion is exactly the poison the
        # warning exists for — never carry it forward as the leading "fact".
        if hypothesis and _COMPLETION_CLAIM_RE.search(hypothesis):
            hypothesis = None
    if rich and hypothesis:
        # The single most valuable thing to carry: the diagnosis the stuck attempt
        # reached. Keep the tail (the conclusion), clipped so it can't blow the budget.
        lines.append("Working hypothesis from the previous attempt (verify, don't assume):")
        lines += ["  " + ln[:160] for ln in hypothesis.splitlines()[-4:] if ln.strip()]
    if edited:
        lines.append("Files already edited: " + ", ".join(edited[-8:]))
    if rich and last_error:
        lines.append("Last failing check — make THIS pass:")
        err_lines = last_error.strip().splitlines()
        sig = [ln for ln in err_lines if _FAILURE_RE.match(ln.strip())][-3:]
        lines += ["  " + ln[:140] for ln in (sig or err_lines[-3:])]
    if rich and examined:
        lines.append("Already examined (don't re-explore): " + ", ".join(examined[-8:]))
    if commands:
        lines.append("Commands already tried (do not blindly repeat):")
        lines += [f"  $ {c[:120]}" for c in commands[-4:]]
    if len(lines) == 1:
        lines.append("(no edits, commands, or errors were recorded before the budget ran out)")
    return "\n".join(lines[:max_lines])


# --- deadline wrap-up window + early-finish review -----------------------
# The governor's HARD stop (above) only fires on a NO-PROGRESS band — a turn that keeps
# landing+verifying is never interrupted. But TB2 scores container end-state, and a
# still-working turn that gets SIGKILLed at the wall ships whatever half-applied mess it
# was mid-edit on. The wrap-up window is the complementary lever: a one-shot WALL-CLOCK
# nudge, fired regardless of progress, once the turn is inside its final stretch — "land
# your best answer NOW, then call done" — so a productive-but-slow turn commits a scored
# partial instead of being cut off mid-think. It sits beside the governor check and is
# only meaningful when a wall budget (turn_budget_s) is configured (evals / one-shot).
WRAPUP_MIN_S = 120.0     # never start the wrap-up window with less than this much runway
WRAPUP_FRAC = 0.15       # or the last 15% of the wall budget, whichever is larger


def wrapup_window_nudge(wall_s, wall_budget_s, wrapup_fired) -> str | None:
    """One-shot steering note when the turn's wall clock has entered its final wrap-up
    window: remaining time <= max(WRAPUP_MIN_S, WRAPUP_FRAC * budget). Returns the nudge
    text (with the approximate seconds left) or None (lever off, no wall budget, already
    fired, or not yet inside the window). Pure/testable; the caller owns `wrapup_fired`."""
    if not levers.enabled("wrapup_window"):
        return None
    if wrapup_fired or not wall_budget_s:
        return None
    remaining = wall_budget_s - wall_s
    threshold = max(WRAPUP_MIN_S, WRAPUP_FRAC * wall_budget_s)
    if remaining > threshold:
        return None
    return _wrapup_text(max(0, int(remaining)))


def _wrapup_text(secs: int) -> str:
    """The final-stretch landing steer, shared by the 085 soft nudge (wrapup_window_nudge)
    and the 103 hard-deadline landing (wrapup_landing_steer) so both say exactly the same
    thing — write each deliverable to its exact path, imperfect is fine, one check, done."""
    return (f"[about {secs}s left before this turn is force-stopped. STOP exploring and "
            "STOP reading now — you are out of time to investigate further. Save your "
            "best available answer immediately: write each deliverable to the EXACT path "
            "the task named, even if it is imperfect, then run one quick check and call "
            "done. A landed partial answer can score; being SIGKILLed mid-edit scores "
            "nothing.]")


# --- hard wrap-up abort (103) ---------------------------------------------
# wrapup_window_nudge above is a SOFT nudge that only lands if a step boundary happens to
# fall inside the window; the TB2.1 autopsy showed the model is usually buried inside one
# 80-100s generation when the window opens, so it fired 3/89 and rescued 0. This is the
# backstop: a wall-clock stop_condition (agent.py) cuts the in-flight generation INSIDE
# the margin, then forces one time-boxed no-think landing turn. The margin must fit: close
# the cut generation + one landing generation + a couple of fast tool calls.
HARD_WRAPUP_MIN_MARGIN_S = 90.0   # never leave less than this to land, even on a tiny wall
HARD_WRAPUP_MARGIN_FRAC = 0.10    # or 10% of the wall budget, whichever is larger
HARD_WRAPUP_LAND_TOK_CAP = 1024   # the landing generation's hard token ceiling (kilo 097)
HARD_WRAPUP_LAND_TOK_FLOOR = 256  # ...but never box it below this, even past the wall


def land_margin(wall_budget_s) -> float:
    """Seconds before the wall at which to abort-and-land: max(90, 10% of budget). Sits
    INSIDE wrapup_window_nudge's window (max(120s, 15%)) — the soft nudge still fires first
    at any step boundary that falls in the window; this is the mid-generation backstop for
    when none does. Pure/testable. 0/None budget -> the floor (never a negative margin)."""
    return max(HARD_WRAPUP_MIN_MARGIN_S, HARD_WRAPUP_MARGIN_FRAC * (wall_budget_s or 0.0))


def landing_max_tokens(remaining_wall_s, decode_tps) -> int:
    """Token box for the single landing generation: min(1024, remaining_wall * tps * 0.5),
    floored so it stays a sane positive cap even if the wall is already blown. Halving the
    wall-implied token count leaves room for the landing's own tool dispatch, and the 1024
    cap keeps a doomed landing from itself becoming a long generation. Pure/testable."""
    room = (remaining_wall_s or 0.0) * max(1.0, decode_tps or 0.0) * 0.5
    return int(min(HARD_WRAPUP_LAND_TOK_CAP, max(HARD_WRAPUP_LAND_TOK_FLOOR, room)))


def wrapup_landing_steer(remaining_s) -> str:
    """The steer appended when the hard deadline cut a generation (103): 085's wrap-up text
    verbatim (it already says exactly the right thing). Not window-gated — the caller only
    calls this once it has already decided the deadline fired."""
    return _wrapup_text(max(0, int(remaining_s)))


MIN_RELAUNCH_WALL_S = 120.0   # below this remaining wall, an auto-continue relaunch can't
                              # complete a useful round before the task SIGKILL — skip it.


def relaunch_budget(total_wall_s, task_elapsed_s) -> float | None:
    """Wall budget for an auto-continue relaunch: the time that REMAINS of the TASK-level
    budget (total - elapsed), not a fresh full budget. The harness SIGKILLs the whole chad
    process at the task deadline, so a relaunch that inherits the full budget calibrates its
    governor / wrap-up / hard-abort windows against a clock that never elapses before the
    kill — the relaunched turn then rides to a mid-work SIGKILL with nothing landed (the
    exact 3/89-wrapup-fired blind spot). Returns the remaining seconds, or None when too
    little remains to bother relaunching (or no wall budget is configured). Pure/testable."""
    if not total_wall_s:
        return None
    remaining = total_wall_s - task_elapsed_s
    if remaining < MIN_RELAUNCH_WALL_S:
        return None
    return remaining


AUTO_CONTINUE_TOTAL_CAP = 6       # absolute relaunch ceiling per task (base + extras)
AUTO_CONTINUE_REPLENISH_FRAC = 0.5  # grant extras only while this fraction of the task
                                    # wall is still unspent


def replenish_continue(total_wall_s, elapsed_s, used_continues,
                       cap: int = AUTO_CONTINUE_TOTAL_CAP,
                       frac: float = AUTO_CONTINUE_REPLENISH_FRAC) -> bool:
    """Grant an auto-continue relaunch beyond the base allowance? The fixed base of 2
    is wall-blind: build-pov-ray (TB2.1 v1.0.0 run) burned 3 step-capped turns in 637s
    and gave up with 94.7% of a 12000s budget unused (plan 107 F3). While more than
    `frac` of the task wall remains unspent, keep granting fresh attempts, bounded by
    `cap` total relaunches so a pathological fast-stall loop still terminates well
    before the harness SIGKILL. No wall budget -> never (interactive runs keep the
    explicit base allowance only). Pure/testable; the caller counts usage."""
    if not total_wall_s:
        return False
    if used_continues >= cap:
        return False
    return (total_wall_s - elapsed_s) > frac * total_wall_s


def hard_wrapup_deadline(wall_budget_s, already_fired, plan_mode, read_only) -> float | None:
    """Should the hard wrap-up abort ARM for this step? Returns the land_margin (seconds
    before the wall at which to cut the generation) when it should, else None. Lever-gated
    here (like wrapup_window_nudge) so ablation can switch it off, and inert without a wall
    budget, once fired, in plan mode, or on a read-only-intent turn — a Q&A turn has no
    artifact to land. Pure/testable; the caller owns the latch and the clock."""
    if not levers.enabled("hard_wrapup"):
        return None
    if already_fired or not wall_budget_s or plan_mode or read_only:
        return None
    return land_margin(wall_budget_s)


# --- turn-level cumulative think budget -----------------------------------
# The 086 ceiling above (close-and-continue) watches any ONE generation; it is blind to a
# turn that burns its budget across MANY separate thinks each under that ceiling —
# dna-assembly: 123k total generated tokens across 13 big thinks, no single one hit the
# 6k ceiling. This watches the turn's CUMULATIVE reasoning-token spend instead, and acts
# only at STEP boundaries on FRESH generations — never re-opening or extending an
# in-flight completion (086's own gate: relieving the cap mid-think taught the model to
# re-open thinks, inflating cumulative think 2.54x). Two thresholds: a one-shot soft
# steer at half spent, then a PERSISTENT no-think for the rest of the turn once spent —
# distinct from 086's no_think_escalation, which is a ONE-SHOT (thinking restores after a
# single forced action).
TURN_THINK_BUDGET_LO = 8000    # floor: a *turn* budget, not a per-generation one, so it
                               # must clear passes' own median spend (10.7k) comfortably
TURN_THINK_BUDGET_HI = 24000   # ceiling: clamp regardless of wall/decode-speed inputs
TURN_THINK_BUDGET_FRAC = 0.35  # the budget's TIME cost should stay <= ~35% of the wall
TURN_THINK_MIN_WALL_S = 300.0  # below this wall budget the mechanism is inert: a short
                               # auto-continue tail clamps to LO and half-fires on its
                               # first step, churning against hard_wrapup's landing
                               # (plan 107 F2 — the regex-log relaunch signature)
TURN_THINK_REARM_TOK = 3000    # past exhaustion, one forced no-think step is owed per
                               # this many FURTHER think tokens spent (plan 107 F1)


def turn_think_budget(wall_budget_s, decode_tps, frac: float = TURN_THINK_BUDGET_FRAC,
                      lo: int = TURN_THINK_BUDGET_LO, hi: int = TURN_THINK_BUDGET_HI) -> int:
    """B_turn = clamp(frac * wall_budget_s * decode_tps, lo, hi): wall-aware — a 900s task
    cannot afford what a 3600s task can, so the budget scales with the turn's OWN wall
    budget and the engine's actual recent decode speed, not a flat constant. Falls back to
    `hi` (never a tighter bind) when either input is unknown/zero — no wall budget
    configured, or no generation has completed yet this turn (decode_tps unmeasured) — so
    the mechanism never guesses a tight budget from absent signal. Pure and testable."""
    if not wall_budget_s or not decode_tps:
        return hi
    return int(min(hi, max(lo, frac * wall_budget_s * decode_tps)))


TURN_THINK_BUDGET_STEER = (
    "[reasoning budget for this turn is half spent; prefer acting over further analysis "
    "— make your next tool call rather than continuing to think.]")


def turn_think_budget_check(turn_think_tokens: int, budget: int, half_fired: bool,
                            exhausted: bool):
    """Threshold latch for the cumulative per-turn reasoning budget: 'half' fires ONCE as
    a soft steer at budget/2; 'exhausted' fires ONCE as the transition into the throttled
    state (see turn_think_throttle — the TB2.1 v1.0.0 run showed a blanket rest-of-turn
    no-think mute degrades Ornith's tool-call syntax and capability over long tails,
    plan 107). Once exhausted, always returns (None, half_fired, True) — nothing further
    to escalate. Returns (decision, half_fired, exhausted). Pure/testable; the caller
    owns the state."""
    if exhausted:
        return None, half_fired, True
    if turn_think_tokens >= budget:
        return "exhausted", True, True
    if not half_fired and turn_think_tokens >= budget / 2:
        return "half", True, False
    return None, half_fired, False


def turn_think_throttle(turn_think_tokens: int, budget: int, nothink_paid: int,
                        rearm: int = TURN_THINK_REARM_TOK) -> bool:
    """Should THIS step render with thinking off? A duty-cycle throttle for the
    post-exhaustion state: crossing the budget owes one forced no-think action step,
    and every further `rearm` think tokens spent owes one more. The caller counts the
    no-think steps it has already paid (`nothink_paid`). Self-correcting by design —
    a model that keeps burning reasoning is throttled toward always-off (the old
    persistent-mute behavior), while one that stops over-thinking gets its thinking
    back once the owed steps are paid. The v1.0.0 full run's evidence for restoring
    rather than muting: break-filter-js spent 26 straight no-think steps after
    exhaustion and regressed a run1 pass with garbled landing tool calls (plan 107 F1).
    Pure/testable; never touches an in-flight generation (086 contract)."""
    if turn_think_tokens < budget:
        return False
    owed = 1 + (turn_think_tokens - budget) // max(1, rearm)
    return nothink_paid < owed


# Injected as the review turn's task preamble. A fresh Agent on a
# reset KV cache re-derives the deliverables from a clean context — the poisoned context
# that convinced the first attempt it was done is gone — so it can catch the confident-
# wrong `done` (fix-code-vulnerability claimed its OWN tests passed; adaptive-rejection-
# sampler called done on a hallucinated file). It must NOT rewrite already-correct work
# (the "second attempt wrecks correct state" risk); verify first, fix only real failures.
REVIEW_PASS_PROMPT = (
    "\n\n[A previous attempt believes it already finished this task. Do NOT trust that — "
    "independently verify the deliverables against the instruction above from scratch. "
    "Re-read the task and list every concrete deliverable it names (each output file, its "
    "exact path, and the exact format/content it must contain). For EACH one, inspect what "
    "is actually on disk NOW — `ls` the path, `cat`/`head` the contents — and run whatever "
    "check the task implies (the real tests, not a syntax probe). If a deliverable already "
    "checks out, do NOT rewrite it: correct output left alone stays correct, and re-editing "
    "working files only risks breaking them. ONLY fix what actually fails verification — "
    "something missing, at the wrong path, or in the wrong format — with the single "
    "targeted change it needs. Call done once every deliverable verifies.]"
)


def review_pass_should_fire(clean_end, turn_budget_s, elapsed_s, spare_frac=0.30) -> bool:
    """Whether to launch a fresh-context review turn after a task's turns settle: the task
    ended CLEANLY (no budget/step hard-stop banked a note) AND more than `spare_frac` of
    the wall budget is still unspent. Off when no wall budget is configured (interactive /
    unmetered runs never trigger it). Pure/testable; the caller owns the lever/flag gate
    and the actual relaunch."""
    if not clean_end or not turn_budget_s:
        return False
    return elapsed_s < (1.0 - spare_frac) * turn_budget_s


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
