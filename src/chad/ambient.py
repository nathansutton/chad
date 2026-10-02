"""Ambient state for the result channel.

Trace measurement showed the model routes its searching through `bash` under
any tool surface — routing follows the model's trained prior, and steering text
does not move it. The design here leans into that: the harness's knowledge is
made ambient IN the bash route — facts appended to results the model already
reads, never a new tool, never an admonition, never a blocked command.

Each lever has its own trace-measured target (all default ON; see levers.py):

  env_manifest        session-start toolchain/package/search-toolbox inventory in
                      the system prompt tail. Target: the environment-probe class
                      (which/--version/pip list…), measured fail-enriched at
                      3.33/trial in fails vs 2.46 in passes, 37% failing.
  bash_read_skeleton  a one-line symbol map appended the first time a source
                      file's content comes back through bash cat/sed/head,
                      and a definition pointer when a bash grep for a
                      known symbol comes back empty. Target: blind re-reads and
                      the product story (structure ambient in the channel a
                      shell-native model actually uses).
  bash_empty_diagnose why a bash command produced nothing — a sed range past the
                      end of the file, or the pipeline stage whose filter matched
                      nothing. Target: bare `[no output]`/`[exit 1]` results,
                      5% of the tool calls in a measured lean run and a wasted
                      round trip every time.
  verify_baseline     the pre-edit outcome of the project's own test command,
                      recalled on a failing post-edit run. Target: the nine turns
                      a measured run spent working out which failures pre-dated
                      its first edit.

STATE is module-global and session-scoped: one chad process is one session.
`Agent.__init__` resets it for a fresh session. Bookkeeping runs even while a
lever is disabled — the eval harness flips CHAD_DISABLE between tasks
in-process, and a lever enabled mid-session must see true state. With every
lever disabled the annotate path returns its input byte-identically.

The confabulation rule: every line states observable facts in
past tense with its provenance ("last verifying run: …"), never an instruction
and never a claim about correctness — an ambient line must not be quotable as
verification the model didn't perform.
"""

from __future__ import annotations

import functools
import os
import re
import shlex
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, Callable

from . import levers
from .guardrails import _RUNNER_WRAPPER_RE

if TYPE_CHECKING:
    from .repomap import Tag

# Appended ambient text is bounded so it can never bloat a result the clip cap
# already sized (annotation happens after _clip_tool_result on purpose — a
# clipped result must still carry its facts).
_LEDGER_MAX = 300
_SKELETON_MAX = 220
_SKELETON_MIN_DEFS = 3     # a 2-symbol file's structure is visible at a glance
# The definition pointer's lookup runs between the model's tool call and its result, and
# a cold whole-repo parse measured 17.5 s on an 11k-file repo. A decoration line gets a
# second; past that the result goes back without it.
_DEF_POINTER_BUDGET_S = 1.0
_MANIFEST_PROBE_TIMEOUT = 1.5
_MANIFEST_MAX_VERSIONS = 16

# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------

_calls = 0                       # tool results seen (main agent)
_edited: dict[str, set] = {}     # rel path -> symbol names (empty set = file-level)
_wrote: list = []                # rel paths written whole (order kept, deduped)
_last_run: dict | None = None    # {"call": int, "head": str, "exit": int}
_baselines: dict = {}            # run key -> the run's outcome BEFORE any edit landed
_skeleton_shown: set = set()     # abs paths whose skeleton line already rode a result
_def_pointer_seen: dict[str, str] = {}  # identifier -> its pointer line, "" = none
_manifest_cache: str | None = None


def reset() -> None:
    """Fresh session (a new main Agent). Clears all ambient bookkeeping."""
    global _calls, _last_run, _manifest_cache
    _calls = 0
    _baselines.clear()
    _edited.clear()
    _wrote.clear()
    _last_run = None
    _skeleton_shown.clear()
    _def_pointer_seen.clear()
    _manifest_cache = None


def snapshot() -> dict:
    """The session facts worth carrying across a resume, as JSON-ready data: what was
    edited and written, the last verifying run and the pre-edit baselines, and which
    files' skeleton lines the model has already been shown. Saved with the conversation
    (`Agent.save`) so a resumed session answers "was this failing before I started?" and
    does not re-decorate a file the transcript already explains. The definition-pointer
    memo is left out on purpose: it caches where a symbol was (or that it was nowhere)
    at lookup time, and a session that ended may have moved it."""
    return {
        "calls": _calls,
        "edited": {rel: sorted(syms) for rel, syms in _edited.items()},
        "wrote": list(_wrote),
        "last_run": dict(_last_run) if _last_run else None,
        "baselines": {key: dict(base) for key, base in _baselines.items()},
        "skeleton_shown": sorted(_skeleton_shown),
    }


def restore(state: dict) -> None:
    """Rebuild the facts `snapshot` recorded, on top of a fresh session. Each entry is
    taken only in the shape `snapshot` writes it; anything else (an older file, a hand
    edit) is skipped, so a damaged entry costs its fact and never the resume."""
    global _calls, _last_run
    calls = state.get("calls")
    if isinstance(calls, int) and calls >= 0:
        _calls = calls
    edited = state.get("edited")
    if isinstance(edited, dict):
        for rel, syms in edited.items():
            if isinstance(rel, str) and rel and isinstance(syms, list):
                _edited[rel] = {s for s in syms if isinstance(s, str)}
    wrote = state.get("wrote")
    if isinstance(wrote, list):
        _wrote.extend(rel for rel in wrote if isinstance(rel, str) and rel not in _wrote)
    last_run = state.get("last_run")
    if (isinstance(last_run, dict) and isinstance(last_run.get("call"), int)
            and isinstance(last_run.get("exit"), int)):
        _last_run = {"call": last_run["call"], "head": str(last_run.get("head") or "bash"),
                     "exit": last_run["exit"], "key": str(last_run.get("key") or "")}
    baselines = state.get("baselines")
    if isinstance(baselines, dict):
        for key, base in baselines.items():
            if (isinstance(key, str) and key and isinstance(base, dict)
                    and isinstance(base.get("call"), int) and isinstance(base.get("exit"), int)):
                _baselines[key] = {"call": base["call"], "exit": base["exit"],
                                   "summary": str(base.get("summary") or "")}
    shown = state.get("skeleton_shown")
    if isinstance(shown, list):
        _skeleton_shown.update(p for p in shown if isinstance(p, str))


# ---------------------------------------------------------------------------
# bookkeeping (always on — see module docstring)
# ---------------------------------------------------------------------------

_ERROR_SENTINELS = ("[exit", "[timed out", "[interrupted", "[failed to launch")
_EXIT_RE = re.compile(r"^\[exit (-?\d+)\]")
_EDIT_TOOLS = ("edit",)


def _rel(path: str) -> str:
    try:
        rel = os.path.relpath(path)
    except ValueError:
        return path
    return path if rel.startswith("..") else rel


def _cmd_head(command: str) -> str:
    """The program the verifying run actually invoked, for the ledger's provenance
    note — derived from the same regex the verify gate uses, so the two never
    disagree about what counts as executing."""
    from . import guardrails
    m = guardrails._EXECUTES_RE.search(command)
    if not m:
        return ""
    return re.sub(r"^[^\w./]+", "", m.group(0)).strip()


# Prefixes that wrap the real program without being it — dropped from the run key so
# `time npm test` and `npm test` are the same check.
_KEY_WRAPPERS = frozenset(("nohup", "time", "env", "command", "exec", "stdbuf",
                           "nice", "caffeinate"))


def _run_key(bare: str) -> str:
    """A stable identity for "the same check, run again": the command's leading one or
    two non-flag words (`npm test`, `npx ava`, `pytest`, `cargo test`). Coarser than the
    full command on purpose — `npx ava test/main.ts` and `npx ava test/retry.ts` are the
    same suite at different scopes — but never so coarse that `npx tsc` and `npx ava`
    collide, which would let a baseline be recalled against a different tool's failure.
    "" for a command with no word-shaped head, which then records no baseline."""
    words = []
    for tok in _shell_tokens(bare)[:6]:
        if os.path.basename(tok) in _KEY_WRAPPERS:
            continue
        if tok.startswith("-"):
            break
        words.append(os.path.basename(tok))
        if len(words) == 2:
            break
    if not words or not re.match(r"^[\w.+-]+$", words[0]):
        return ""
    return " ".join(words)


def _landed(name: str, result: str) -> bool:
    if name == "write":
        return result.startswith("[wrote")
    return name in _EDIT_TOOLS and result.startswith("[edited")


def note_call(name: str, args: dict, result: str) -> None:
    """Fold one tool result into the session state. Facts only — no lever gates
    here, so state is true whenever a lever consults it."""
    global _calls, _last_run
    _calls += 1
    if _landed(name, result):
        path = str(args.get("path", "") or "")
        if name == "write":
            rel = _rel(path)
            if rel and rel not in _wrote:
                _wrote.append(rel)
            _edited.pop(rel, None)  # a whole-file write supersedes prior edit facts
        else:
            rel = _rel(path) if path else ""
            if rel:
                _edited.setdefault(rel, set())
        return
    if name == "bash":
        from . import guardrails
        cmd = str(args.get("command", "") or "")
        if not cmd:
            return
        if guardrails.reverts_working_tree(cmd) and not result.startswith(_ERROR_SENTINELS):
            # The tree was discarded: earlier edit facts are no longer true state.
            _edited.clear()
            _wrote.clear()
            return
        if result.startswith(("[timed out", "[interrupted", "[failed to launch")):
            return
        # The verify gate's own normalization and predicates (`bash_result_verifies`), so
        # `uv run pytest` records a pytest run and the ledger records exactly the
        # commands that gate counts as runs — including ones that exited non-zero.
        bare = _RUNNER_WRAPPER_RE.sub("", cmd)
        if guardrails._is_trivial_check(bare) or not guardrails._is_executing_command(bare):
            return
        m = _EXIT_RE.match(result)
        key = _run_key(bare)
        _last_run = {"call": _calls, "head": _cmd_head(bare) or "bash",
                     "exit": int(m.group(1)) if m else 0, "key": key}
        if key and key not in _baselines and not _edited and not _wrote:
            # This runner's state BEFORE the session touched anything. Captured only
            # here: once an edit lands, no observation can separate a failure that was
            # already there from one the model just caused. Keyed per runner so a `tsc`
            # baseline is never recalled against an `ava` failure.
            _baselines[key] = {"call": _calls, "exit": _last_run["exit"],
                               "summary": _summary_lines(result)}


# ---------------------------------------------------------------------------
# E3 — skeleton / definition pointer on the bash route
# ---------------------------------------------------------------------------

_BASH_OPEN_RE = re.compile(
    r"(?:^|[|;&]|\$\()\s*(?:sudo\s+)?(?:cat|head|tail|nl|less|more|sed)\b")
_BASH_GREP_RE = re.compile(r"(?:^|[|;&]|\$\()\s*(?:sudo\s+)?(?:grep|rg|ag)\b")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{2,}$")
_PATHISH_RE = re.compile(r"^[\w./+-]+\.[A-Za-z0-9]{1,6}$")
_GREP_SKIP_ARG = frozenset(("-e", "-A", "-B", "-C", "-m", "-g", "-t",
                            "--include", "--exclude", "--exclude-dir", "--glob",
                            "--type"))


def _shell_tokens(command: str) -> list:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _opened_paths(command: str) -> list:
    """Existing file paths a display command (cat/head/sed …) put on screen."""
    if not _BASH_OPEN_RE.search(command):
        return []
    return [t for t in _shell_tokens(command)
            if _PATHISH_RE.match(t) and os.path.isfile(t)]


def _grep_ident(command: str) -> str:
    """The identifier a bash grep searched for, or "". First non-flag argument
    of the first grep/rg/ag segment, identifier-shaped only."""
    m = _BASH_GREP_RE.search(command)
    if not m:
        return ""
    tail = command[m.start():]
    for cut in ("|", ";", "&&"):
        i = tail.find(cut, 1)
        if i > 0:
            tail = tail[:i]
    toks = _shell_tokens(tail)
    skip = False
    for t in toks[1:]:
        if skip:
            skip = False
            continue
        if t.startswith("-"):
            skip = t in _GREP_SKIP_ARG and "=" not in t
            continue
        t = t.strip("'\"")
        return t if _IDENT_RE.match(t) else ""
    return ""


def _skeleton_line(path: str) -> str:
    """`[file] rel: sym 12-88 · …` from the cached tags, top-level symbols only,
    or "". Never parses fresh beyond what the mtime cache already does."""
    from . import repomap
    svc = repomap.service()
    if not svc.lang_for(path):
        return ""
    try:
        defs, _ = svc._extract(os.path.abspath(path))
    except Exception:  # noqa: BLE001 - structure is best-effort decoration
        return ""
    top = [d for d in defs if not d.scope]
    if len(top) < _SKELETON_MIN_DEFS:
        return ""
    ent = []
    for d in top:
        mark = "()" if d.kind in ("function", "method") else ""
        ent.append(f"{d.name}{mark} {d.line}-{d.end_line}")
    line = f"[file] {_rel(path)}: " + " · ".join(ent)
    if len(line) > _SKELETON_MAX:
        line = line[: _SKELETON_MAX - 1] + "…"
    return line


def _tags_lookup(ident: str, should_stop: Callable[[], bool]) -> list[Tag]:
    """Definitions of `ident` from the cwd's tags service. A miss is reported as-is:
    it never re-walks the tree for files created since the service's walk."""
    from . import repomap
    return repomap.service()._find_defs(ident, should_stop=should_stop, refresh=False)


def _def_pointer(ident: str,
                 lookup: Callable[[str, Callable[[], bool]], list[Tag]] = _tags_lookup,
                 budget_s: float = _DEF_POINTER_BUDGET_S) -> str:
    """`[file] this came back empty; 'x' is defined at rel:line` when a bash search that
    returned nothing named a symbol the tags cache knows — the definition answer
    delivered on the bash route. The wording is about the RESULT, not the grep: in
    `rg X src/ | grep -v y` the grep matched fine and a later stage emptied it.

    Memoized per identifier for the session, and bounded by `budget_s`: `lookup` is
    asked to stop once it is spent, and a lookup cut short may have missed same-named
    defs, so it answers nothing rather than a partial list."""
    if ident in _def_pointer_seen:
        return _def_pointer_seen[ident]
    deadline = time.monotonic() + budget_s
    try:
        hits = lookup(ident, lambda: time.monotonic() > deadline)
    except Exception:  # noqa: BLE001
        hits = []
    if time.monotonic() > deadline:
        hits = []
    line = ""
    if hits and len(hits) <= 3:  # a pile of same-named defs is not an answer
        where = " · ".join(f"{d.rel}:{d.line}" for d in hits)
        line = f"[file] this came back empty; `{ident}` is defined at {where}"
    _def_pointer_seen[ident] = line
    return line


def _skeleton_suffix(name: str, args: dict, result: str, step=None) -> str:
    if not levers.enabled("bash_read_skeleton"):
        return ""
    candidates: list = []
    zero_hit_ident = ""
    if name == "bash":
        cmd = str(args.get("command", "") or "")
        if not result.startswith(("[timed out", "[interrupted", "[failed to launch")):
            candidates.extend(_opened_paths(cmd)[:2])
        if result == "[no output]" or result.startswith("[exit 1]"):
            zero_hit_ident = _grep_ident(cmd)
    out = []
    for p in candidates[:2]:
        ap = os.path.abspath(p)
        if ap in _skeleton_shown:
            continue
        line = _skeleton_line(p)
        if line:
            _skeleton_shown.add(ap)
            levers.fired("bash_read_skeleton", step=step, kind="skeleton")
            out.append(line)
    if zero_hit_ident:
        line = _def_pointer(zero_hit_ident)
        if line:
            levers.fired("bash_read_skeleton", step=step, kind="defsite")
            out.append(line)
    return ("\n" + "\n".join(out)) if out else ""


# ---------------------------------------------------------------------------
# E4 — verify baseline
# ---------------------------------------------------------------------------

# Lines a test runner uses to state its own bottom line. Matched against the tail of a
# run's output so the baseline carries WHAT the suite did, not just its exit code:
# "2 known failures" is the fact that answers "was this failing before I started?".
_SUMMARY_RE = re.compile(
    r"^\s*[\-=✔✘✗•]*\s*\d+\s+(?:tests?\s+)?"
    r"(?:passed|failed|failing|passing|errors?|known failures?|skipped|pending)\b"
    r"|^\s*(?:OK|FAILED|PASSED)\b|^=+\s*\d+\s+\w+.*=+$",
    re.I | re.M)
_SUMMARY_MAX = 120


def _summary_lines(result: str) -> str:
    """The runner's own bottom-line rows from a result, joined and clipped — "" when the
    output states no such summary. Verbatim slices only: a paraphrase of a test result
    is exactly the confabulation the ledger rule forbids."""
    hits = [m.group(0).strip() for m in _SUMMARY_RE.finditer(result)]
    if not hits:
        return ""
    out = " · ".join(dict.fromkeys(hits[-3:]))
    return out[:_SUMMARY_MAX - 1] + "…" if len(out) > _SUMMARY_MAX else out


def _baseline_suffix(name: str, args: dict, result: str, step: int | None = None) -> str:
    """The `[baseline]` fact line for a FAILING verifying run made after the session's
    first edit, or "".

    The measured cost of not having it: a run whose post-edit suite came back with two
    failures spent nine turns re-running subsets and grepping the tests for
    `test.failing` markers to work out which failures pre-dated it. The harness watched
    the pre-edit run go by and said nothing. Emitted only on a non-zero exit, because
    that is the only moment the question is live; only for the same runner that was
    baselined; and never as a verdict about the current failure — the line states what
    that command did before the first edit and stops."""
    if not levers.enabled("verify_baseline"):
        return ""
    if name != "bash" or _last_run is None or _last_run["call"] != _calls:
        return ""
    if _last_run["exit"] == 0 or not (_edited or _wrote):
        return ""            # it passed, or nothing has been changed to blame
    base = _baselines.get(_last_run["key"])
    if base is None or base["call"] == _calls:
        return ""            # never baselined, or this IS the baseline run
    status = "exit 0" if base["exit"] == 0 else f"exit {base['exit']}"
    summary = f' · "{base["summary"]}"' if base["summary"] else ""
    levers.fired("verify_baseline", step=step)
    return (f"\n[baseline] before your first edit, `{_last_run['key']}` → "
            f"{status}{summary}")


# ---------------------------------------------------------------------------
# E5 — empty-result diagnosis on the bash route
# ---------------------------------------------------------------------------

# A bash result with no output at all. `[exit N]` with nothing after it is the same
# event as `[no output]` — the command ran, printed nothing, and the model is left to
# guess whether that means "no matches", "wrong path", or "the tool is broken".
_EMPTY_RE = re.compile(r"\A\[exit -?\d+\]\s*\Z")
# `sed -n '120,180p' file` / `sed -n 120,180p file` — the ranged read the lean prompt
# teaches, and the one whose silent failure mode (range past EOF) is indistinguishable
# from "this file is empty here".
_SED_RANGE_RE = re.compile(r"""sed\s+(?:-[a-zA-Z]+\s+)*-n\s+['"]?(\d+)\s*,\s*(\d+)p['"]?""")
_GREP_TAIL_RE = re.compile(r"(?:^|[|;&])\s*(?:sudo\s+)?(grep|rg|ag)\b([^|;&]*)")


def _inverts(tok: str) -> bool:
    """Whether a grep argument turns the match inside out (`-v`, `--invert-match`, or
    `v` bundled into a short cluster like `-rnv`) — but never a long option that merely
    contains a v, and never a `-v` that is some other tool's version flag, since only
    the grep segment's own tokens are scanned."""
    if tok in ("--invert-match", "-v"):
        return True
    return (tok.startswith("-") and not tok.startswith("--")
            and "v" in tok[1:] and tok[1:].isalpha())


def _is_empty_result(result: str) -> bool:
    return result == "[no output]" or bool(_EMPTY_RE.match(result))


def _past_eof_note(command: str) -> str:
    """`file has N lines` when the command asked sed for a range that starts past the
    end of the file, else "". The measured loss is one whole turn: a `sed -n '330,420p'`
    on a 324-line file returned `[no output]`, which the model could only resolve by
    spending its next call on `wc -l`."""
    m = _SED_RANGE_RE.search(command)
    if not m:
        return ""
    start, end = int(m.group(1)), int(m.group(2))
    for path in _shell_tokens(command[m.end():]):
        if not _PATHISH_RE.match(path) or not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as f:
                total = sum(1 for _ in f)
        except OSError:
            return ""
        if start > total:
            return (f"[file] {_rel(path)} has {total} lines — the requested "
                    f"{start}-{end} is past the end")
        return ""
    return ""


def _no_match_note(command: str) -> str:
    """What the pipeline's LAST filter did with its input, or "". An empty result from
    `rg X src/ | grep -v y` is a fact about that final stage, not about the tree; naming
    the stage that came up empty is what stops the model re-running the same pipeline
    with a different tail. An inverting filter gets its own wording — "nothing matched"
    would be exactly backwards for `-v`, which comes up empty when everything did."""
    hits = list(_GREP_TAIL_RE.finditer(command))
    if not hits:
        return ""
    tool, rest = hits[-1].group(1), hits[-1].group(2)
    toks = _shell_tokens(rest)
    staged = " (the last stage of the pipeline)" if "|" in command else ""
    if any(_inverts(t) for t in toks):
        # Not "nothing matched": for `-v` an empty result means everything did — or
        # that an earlier stage sent it nothing. Both are true of "let no rows
        # through", and neither is a claim about the tree the first stage searched.
        return (f"[grep] the `{tool} -v` filter{staged} let no rows through — it "
                f"either received none or excluded them all")
    pattern, scope, skip = "", [], False
    for t in toks:
        if skip:
            skip = False
            continue
        if t.startswith("-"):
            skip = t in _GREP_SKIP_ARG and "=" not in t
            continue
        if not pattern:
            pattern = t
        else:
            scope.append(t)
    if not pattern:
        return ""
    where = " in " + " ".join(scope[:3]) if scope else ""
    return f"[grep] nothing matched {pattern!r}{where}{staged} — {tool} found no rows"


# `rg -r X` / `rg --replace X`: ripgrep prints matches with the matched text REWRITTEN.
# A short-flag cluster is the trap — `-rn` is `--replace n`, not grep's recursive `-n`
# pair — so the letters after `r` become the replacement and the rows on screen stop
# being the file's text.
_RG_SEGMENT_RE = re.compile(r"(?:^|[|;&(])\s*(?:sudo\s+)?rg\b([^|;&]*)")


def _replace_flag(tok: str) -> str:
    """The `--replace` form this rg argument is, or "". Short clusters count (`-rn` is
    `--replace n`); long options that merely start with the letters do not (`--regexp`
    is not `-r`), which is why this inspects tokens instead of pattern-matching the
    command string."""
    if tok == "--replace" or tok.startswith("--replace="):
        return "--replace"
    if (tok.startswith("-") and not tok.startswith("--")
            and tok[1:].isalpha() and "r" in tok[1:]):
        return tok
    return ""


def _rg_replace_note(command: str) -> str:
    """One line naming what an `rg -r` result actually is, or "". Fires on the command,
    not on the output: the whole failure mode is that the output looks plausible. A run
    that hit this twice read `test('n option', …)` back from its own test file and
    concluded the file had been corrupted."""
    for seg in _RG_SEGMENT_RE.finditer(command):
        for tok in _shell_tokens(seg.group(1)):
            flag = _replace_flag(tok)
            if not flag:
                continue
            how = (f"`{flag}` is `--replace` with `{flag[flag.index('r') + 1:]}` as the "
                   f"replacement text" if len(flag) > 2 and not flag.startswith("--")
                   else "`-r` is `--replace`, not grep's recursive flag")
            return ("[grep] these rows are rg's REWRITTEN output, not the file's text — "
                    f"{how} (rg recurses by default). Re-run without it to see the "
                    f"real lines.")
    return ""


def _rg_replace_suffix(name: str, args: dict, result: str, step: int | None = None) -> str:
    if not levers.enabled("rg_replace_flag_note"):
        return ""
    if name != "bash" or not result.strip():
        return ""
    note = _rg_replace_note(str(args.get("command", "") or ""))
    if not note:
        return ""
    levers.fired("rg_replace_flag_note", step=step)
    return "\n" + note


def _empty_suffix(name: str, args: dict, result: str, step: int | None = None) -> str:
    """Why a bash command produced nothing, when the harness can say so from facts it
    already has. Bare `[no output]` / `[exit 1]` results were 5% of the tool calls in a
    measured lean-mode run and every one of them cost a full round trip."""
    if not levers.enabled("bash_empty_diagnose"):
        return ""
    if name != "bash" or not _is_empty_result(result):
        return ""
    command = str(args.get("command", "") or "")
    note = _past_eof_note(command) or _no_match_note(command)
    if not note:
        return ""
    levers.fired("bash_empty_diagnose", step=step,
                 kind="past_eof" if note.startswith("[file]") else "no_match")
    return "\n" + note


# ---------------------------------------------------------------------------
# the one agent-facing hook
# ---------------------------------------------------------------------------

def annotate(name: str, args: dict, result: str, step: int | None = None) -> str:
    """Fold this result into session state, then append whatever ambient lines
    the enabled levers owe it. With every lever OFF, returns `result` unchanged
    (byte-identical) — the bookkeeping still runs so a lever enabled mid-session
    sees true state."""
    note_call(name, args, result)
    suffix = ""
    suffix += _skeleton_suffix(name, args, result, step=step)
    suffix += _empty_suffix(name, args, result, step=step)
    suffix += _rg_replace_suffix(name, args, result, step=step)
    suffix += _baseline_suffix(name, args, result, step=step)
    return result + suffix


# ---------------------------------------------------------------------------
# E1 — environment manifest
# ---------------------------------------------------------------------------

# What the traces show being probed. Version flags where a tool deviates from
# `--version`; absences reported only for the commonly-probed set (an exhaustive
# absence list would be noise). Alias families (python3/python, pip/pip3) count
# as absent only when the WHOLE family is missing — "NOT installed: pip" next to
# a present pip3 would be a lie.
_MANIFEST_TOOLS = (
    "python3", "python", "pip", "pip3", "gcc", "g++", "clang", "make", "cmake",
    "git", "node", "npm", "cargo", "rustc", "go", "java", "javac", "docker",
    "ruby", "perl", "pytest", "uv",
)
_VERSION_ARGS = {"go": ("version",), "java": ("-version",), "javac": ("-version",)}
_NOTABLE_ABSENT = ("python3", "pip", "gcc", "g++", "make", "cmake", "node",
                   "npm", "cargo", "go", "java", "docker")
_ALIAS_FAMILY = {"python3": ("python3", "python"), "pip": ("pip", "pip3")}
_PKG_MANAGERS = ("apt-get", "dnf", "yum", "apk", "brew", "pacman")
# The search/read toolbox itself — the verbs a shell-first arm does all of its looking
# with. Reported presence-only (no version probe): what costs a turn is `rg` not being
# installed on the host, not which ripgrep it is. This matters most where bash is the
# only route: the prompt there teaches `rg -n` as the first move, and chad's own grep is
# pure Python, so ripgrep is a dependency the SESSION acquires and the harness never
# had. An exit-127 answers it in one wasted round trip; this answers it in zero.
_SEARCH_TOOLS = ("rg", "grep", "sed", "awk", "find", "jq")
_VERSION_NUM_RE = re.compile(r"\d+\.\d+[\w.\-]*")


def _run_for_output(argv: tuple[str, ...]) -> str:
    """Everything a version probe printed: stdout, then stderr."""
    p = subprocess.run(argv, capture_output=True, text=True,
                       errors="replace", timeout=_MANIFEST_PROBE_TIMEOUT)
    return (p.stdout or "") + (p.stderr or "")


def _probe_version(tool: str,
                   output: Callable[[tuple[str, ...]], str] = _run_for_output) -> str:
    args = _VERSION_ARGS.get(tool, ("--version",))
    try:
        text = output((tool,) + args)
    except (OSError, subprocess.SubprocessError):
        return ""
    first = text.strip().splitlines()
    if not first:
        return ""
    m = _VERSION_NUM_RE.search(first[0])
    return m.group(0) if m else first[0][:24]


def _build_manifest(which: Callable[[str], str | None] = shutil.which,
                    output: Callable[[tuple[str, ...]], str] = _run_for_output) -> str:
    """The manifest body for this host: `which` locates a tool on PATH, and `output`
    runs a version probe and returns what it printed."""
    have = {t for t in _MANIFEST_TOOLS if which(t)}
    present = [t for t in _MANIFEST_TOOLS if t in have]
    absent = [t for t in _NOTABLE_ABSENT
              if not any(a in have for a in _ALIAS_FAMILY.get(t, (t,)))]
    lines = []
    if present:
        # Probes run concurrently — they are subprocess-spawn bound, and serially
        # a full toolchain costs ~4s of session startup (measured); parallel is
        # ~the slowest single probe.
        from concurrent.futures import ThreadPoolExecutor
        to_probe = present[:_MANIFEST_MAX_VERSIONS]
        with ThreadPoolExecutor(max_workers=8) as ex:
            versions = list(ex.map(functools.partial(_probe_version, output=output),
                                   to_probe))
        versioned = [f"{t} {v}" if v else t for t, v in zip(to_probe, versions)]
        versioned.extend(present[_MANIFEST_MAX_VERSIONS:])
        lines.append("- present: " + " · ".join(versioned))
    if absent:
        lines.append("- NOT installed: " + ", ".join(absent))
    pkgs = [p for p in _PKG_MANAGERS if which(p)]
    if pkgs or "pip" in have or "pip3" in have:
        pip = ["pip"] if ("pip" in have or "pip3" in have) else []
        lines.append("- package managers: " + ", ".join(pkgs + pip))
    search = [t for t in _SEARCH_TOOLS if which(t)]
    if search:
        line = "- search/text: " + " · ".join(search)
        gone = [t for t in _SEARCH_TOOLS if t not in search]
        lines.append(line + (" — NOT present: " + ", ".join(gone) if gone else ""))
    return "\n".join(lines)


def env_manifest(build: Callable[[], str] | None = None) -> str:
    """The manifest block body, built once per session (subprocess probes are not
    free) and never rebuilt — installs after session start are deliberately not
    reflected; the header text says so. `build` makes the body when the session has
    none yet, the host probe (`_build_manifest`) unless given. "" when the lever is off
    or nothing was detected."""
    global _manifest_cache
    if not levers.enabled("env_manifest"):
        return ""
    if _manifest_cache is None:
        try:
            _manifest_cache = (build or _build_manifest)()
        except Exception:  # noqa: BLE001 - orientation is best-effort, never fatal
            _manifest_cache = ""
    return _manifest_cache
