"""Deterministic post-edit syntax gate (plan 038; hardened by plans 067/073).

After a successful write/edit/replace_symbol/insert_symbol, re-check the mutated
file; if the edit *introduced* a syntax error, ride a warning along in the SAME tool
result (`check_syntax`). For the targeted edit tools there is a stronger contract
(`edit_reject`): an edit that turns a cleanly-parsing Python file into one that no
longer parses is REVERTED, not merely warned about. The 073 dogfood showed why the
warning alone is insufficient for a small model: it ignored ~10 consecutive "no longer
parses" warnings while line-addressed edits severed a multi-line `def` signature, and
every later edit was surgery on garbage. A file that was ALREADY broken stays editable
(a real fix passes through still-broken states), and whole-file `write` stays warn-only
as the escape hatch for deliberate multi-step rewrites.

Python is checked exactly with `ast.parse` (line-accurate). Other languages use a
tree-sitter ERROR/MISSING-node delta: warn only when the edit ADDED nodes, since many
real files carry baseline parse errors tree-sitter can't fully recover (and we must
never warn on a pre-existing one). Gated by CHAD_NO_SYNTAX_GATE for run_evals --ab.
"""

import ast
import os

from . import config, levers, repomap

_MAX_BYTES = 1_000_000  # skip pathologically large files — the parse cost isn't worth it
_PARSERS: dict = {}     # lang -> tree_sitter.Parser | None (grammar download cached by tlp)


def _parser(lang):
    """A raw tree-sitter Parser for `lang`, or None if the grammar won't build. Reuses
    repomap's language name; kept separate from repomap._lang_tools because that one is
    keyed on the *tags* query (absent for many grammars we can still parse)."""
    if lang not in _PARSERS:
        try:
            import tree_sitter_language_pack as tlp
            from tree_sitter import Parser
            _PARSERS[lang] = Parser(tlp.get_language(lang))
        except Exception:
            _PARSERS[lang] = None
    return _PARSERS[lang]


def _error_nodes(root) -> int:
    """Count ERROR and MISSING nodes in the tree — tree-sitter's two ways of flagging a
    fragment it couldn't parse."""
    n, stack = 0, [root]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            n += 1
        stack.extend(node.children)
    return n


def _ts_error_count(lang, text: str):
    """ERROR/MISSING count for `text` parsed as `lang`, or None if we can't tell (no
    grammar / parse blew up) — a None means 'don't warn', never 'clean'."""
    parser = _parser(lang)
    if parser is None:
        return None
    try:
        tree = parser.parse(text.encode("utf-8", "replace"))
    except Exception:
        return None
    return _error_nodes(tree.root_node)


def check_syntax(path: str, before: str | None) -> str | None:
    """A warning string when the current on-disk content of `path` has a *newly
    introduced* syntax error, else None.

    `before` is the pre-edit text (None for a freshly created file). The post-edit
    content is read here, so a tool that left the file unchanged (a failed or no-op
    edit → before == after) never warns, and callers don't have to detect success.
    """
    if config.flag("CHAD_NO_SYNTAX_GATE"):
        return None
    try:
        if os.path.getsize(path) > _MAX_BYTES:
            return None
        with open(path, errors="replace") as f:
            after = f.read()
    except OSError:
        return None
    if after == before:      # the tool didn't actually change the file — nothing to flag
        return None

    lang = repomap.service().lang_for(path)

    if lang == "python":
        try:
            ast.parse(after)
        except SyntaxError as e:
            lines = after.splitlines()
            line = lines[e.lineno - 1] if e.lineno and e.lineno <= len(lines) else ""
            return (f"\n[warning: the file no longer parses — {e.msg} at line "
                    f"{e.lineno}: {line.strip()!r}. Fix this before moving on.]")
        return None

    if lang:
        after_errs = _ts_error_count(lang, after)
        if not after_errs:               # None (can't tell) or 0 (clean) -> no warning
            return None
        # Only warn if the edit ADDED errors. A new file has a baseline of 0.
        before_errs = _ts_error_count(lang, before) if before is not None else 0
        if before_errs is None or after_errs <= before_errs:
            return None
        return ("\n[warning: this edit introduced a syntax error — the file no longer "
                "parses cleanly. Re-check the change before moving on.]")
    return None


def _enclosing_symbol(tree, line: int) -> str | None:
    """The `Class/method` (or bare function/class) name whose body encloses `line` in a
    parsed tree — the unit the model should hand to replace_symbol. Deepest match wins;
    None when the line is at module level (nothing to rewrite whole)."""
    chain: list[str] = []

    def visit(node, path):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = getattr(child, "lineno", None)
                end = getattr(child, "end_lineno", start)
                if start and end and start <= line <= end:
                    newpath = path + [child.name]
                    chain[:] = newpath          # deepest containing chain wins (assigned last)
                    visit(child, newpath)
            else:
                visit(child, path)

    visit(tree, [])
    return "/".join(chain[-2:]) if chain else None


def _severed_span(tree, start: int, end: int) -> tuple[int, int] | None:
    """The innermost multi-line region that the edit range [start, end] cuts INTO
    without covering — the statement (or compound-statement header, e.g. a multi-line
    `def` signature) whose severing produced the parse failure. `start = end + 1`
    encodes an insertion boundary between `end` and `start`. Message-quality only:
    the reject has already been decided by ast.parse, so an approximate span is fine."""
    best: tuple[int, int] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        a: int | None = getattr(node, "lineno", None)
        if a is None:
            continue
        b: int = getattr(node, "end_lineno", None) or a
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and hasattr(body[0], "lineno"):
            # Compound statement: only its HEADER (def/class/if... line through the
            # colon) is unsplittable; the body is made of its own statements.
            b = max(a, body[0].lineno - 1)
        if b <= a:
            continue                       # single physical line — can't be severed
        cut_top = a < start <= b           # range (or insertion point) starts inside
        cut_bot = a <= end < b             # range ends inside
        if (cut_top or cut_bot) and (best is None or (b - a) < (best[1] - best[0])):
            best = (a, b)
    return best


def _numbered(text: str, first: int, last: int, cap: int = 10) -> str:
    """Lines [first, last] of `text` with 1-based line numbers, capped at `cap` lines —
    the re-anchoring echo a reject carries so the model's next call uses numbers it has
    actually seen (post-revert, `text` is still what is on disk)."""
    lines = text.splitlines()
    first, last = max(1, first), min(len(lines), last)
    shown = lines[first - 1:last]
    clipped = ""
    if len(shown) > cap:
        shown, clipped = shown[:cap], f"\n  … (+{last - first + 1 - cap} more lines to {last})"
    width = len(str(first + len(shown)))
    body = "\n".join(f"{first + i:>{width}}  {ln.rstrip()}" for i, ln in enumerate(shown))
    return body + clipped


def edit_reject(path: str, before: str, after: str,
                edit_range: tuple[int, int] | None = None) -> str | None:
    """A rejection message when an *edit* would take a cleanly-parsing Python file to
    one that no longer parses, else None — the edit path uses this to REVERT rather
    than let the break land.

    check_syntax only warns and lets the edit stand. But the 073 dogfood measured what
    a landed break costs a small model: it can't reliably repair a file it broke — it
    ignored ten consecutive parse warnings while stale line-range edits severed a
    multi-line `def` signature, then LOOP-ABORTed with the file broken. So for the
    targeted edit tools the contract is: every landed edit leaves the file parsing.
    A multi-step change that must pass through a broken state has two sanctioned paths:
    replace_symbol (whole-definition rewrite) or write (whole-file, still warn-only).
    We reject ONLY when `before` parses cleanly — a file that was ALREADY broken is left
    editable so a real fix (which passes through a transiently-still-broken state) is
    never stranded. IndentationError keeps its own lever (`syntaxgate_revert`, plan 067)
    and message; the generalization to any SyntaxError is levered as `syntax_revert`
    (plan 073). `edit_range` = the 1-based [start, end] the edit replaced (start = end+1
    for an insertion boundary), used to name the severed statement and echo the region.
    """
    if config.flag("CHAD_NO_SYNTAX_GATE"):
        return None
    if len(after) > _MAX_BYTES:
        return None
    if repomap.service().lang_for(path) != "python":
        return None
    try:
        tree = ast.parse(before)
    except SyntaxError:
        return None            # before wasn't clean — don't strand a fix-in-progress
    try:
        ast.parse(after)
        return None
    except IndentationError as e:  # catch before SyntaxError — it is a subclass
        # Ablating this reverts to warn-only: the indent break LANDS, which is the
        # precondition for the whitespace-surgery death loop this fix exists to stop.
        if not levers.enabled("syntaxgate_revert"):
            return None
        lines = after.splitlines()
        line = lines[e.lineno - 1] if e.lineno and e.lineno <= len(lines) else ""
        # B: name the enclosing function so the model can take the STABLE path — rewrite
        # the whole symbol — instead of re-hand-indenting lines it can't get right. The
        # error line is in `after`; against `before`'s tree it lands in the same function.
        before_lines = before.count("\n") + 1
        sym = _enclosing_symbol(tree, min(e.lineno or 1, before_lines))
        steer = (f" You're editing inside `{sym}`: the reliable fix is replace_symbol to "
                 f"rewrite that whole function (you send the complete function and its "
                 f"indentation is handled)." if sym else "")
        return (f"[edit rejected: it would break {os.path.basename(path)} — {e.msg} at "
                f"line {e.lineno}: {line.strip()!r}. The file was left unchanged.{steer} "
                f"Or use replace_lines / insert_lines with the line numbers from read — they "
                f"fit indentation for you — instead of hand-quoting whitespace.]")
    except SyntaxError as e:
        # Ablating this restores warn-and-land for non-indent breaks — the corruption
        # engine of the 073 dogfood (severed signature landed, then compounded).
        if not levers.enabled("syntax_revert"):
            return None
        lines = after.splitlines()
        line = lines[e.lineno - 1] if e.lineno and e.lineno <= len(lines) else ""
        before_lines = before.count("\n") + 1
        anchor = edit_range[0] if edit_range else (e.lineno or 1)
        sym = _enclosing_symbol(tree, min(anchor, before_lines))
        span = _severed_span(tree, *edit_range) if edit_range else None
        parts = [f"[edit rejected: it would leave {os.path.basename(path)} unparseable — "
                 f"{e.msg} at line {e.lineno}: {line.strip()!r}. The file was left "
                 f"unchanged."]
        if span:
            parts.append(f" Your range cut into a multi-line statement spanning lines "
                         f"{span[0]}-{span[1]} — never replace a fragment of it: send "
                         f"the COMPLETE statement, replacing lines {span[0]}-{span[1]} "
                         f"in one call.")
        if sym:
            parts.append(f" Most reliable: replace_symbol('{sym}') with the complete "
                         f"new definition.")
        echo_a, echo_b = (span or edit_range or (e.lineno or 1, e.lineno or 1))
        parts.append("\n Current lines (unchanged, use THESE numbers):\n"
                     + _numbered(before, echo_a - 2, echo_b + 2) + "\n]")
        return "".join(parts)


def indent_reject(path: str, before: str, after: str) -> str | None:
    """Back-compat name for `edit_reject` (plan 067 shipped it as indent-only; plan 073
    generalized it). Prefer `edit_reject`, which also takes the edit range."""
    return edit_reject(path, before, after)
