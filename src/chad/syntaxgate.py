"""Deterministic post-edit syntax gate (plan 038).

After a successful write/edit/replace_symbol/insert_symbol, re-check the mutated
file; if the edit *introduced* a syntax error, ride a warning along in the SAME tool
result. It never blocks or rolls the edit back — the edit stands and the warning
costs no extra round-trip, so the model reacts with the context still hot instead of
paying a run-tests → parse-failure → re-locate cycle to notice a broken file.

Python is checked exactly with `ast.parse` (line-accurate). Other languages use a
tree-sitter ERROR/MISSING-node delta: warn only when the edit ADDED nodes, since many
real files carry baseline parse errors tree-sitter can't fully recover (and we must
never warn on a pre-existing one). Gated by CHAD_NO_SYNTAX_GATE for run_evals --ab.
"""

import ast
import os

from . import repomap

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
    if os.environ.get("CHAD_NO_SYNTAX_GATE"):
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
