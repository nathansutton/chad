"""Post-mutation syntax warning.

When a landed `edit`/`write` (or a bash command that rewrote a watched file)
leaves the file with a *newly introduced* parse error, the tool result carries a
warning naming the error and its line — in the same message the model is about
to read, which is the only placement that reliably gets acted on. Warn-only by
design: the file the model wrote is the file on disk, and an already-broken file
stays freely editable (a real fix passes through still-broken states).

Python is checked exactly with `ast.parse` (line-accurate). Other languages use
a tree-sitter ERROR/MISSING-node delta: warn only when the mutation ADDED nodes,
since many real files carry baseline parse errors tree-sitter can't fully
recover (and we must never flag a pre-existing one). Gated by
CHAD_NO_SYNTAX_GATE for A/B evaluation.
"""

import ast
import os
import sys

from . import config, repomap

_MAX_BYTES = 1_000_000  # skip pathologically large files — the parse cost isn't worth it
_PARSERS: dict = {}     # lang -> tree_sitter.Parser | None (grammar download cached by tlp)


def _parser(lang):
    """A raw tree-sitter Parser for `lang`, or None if the grammar won't build."""
    if lang not in _PARSERS:
        try:
            import tree_sitter_language_pack as tlp
            from tree_sitter import Parser
            _PARSERS[lang] = Parser(tlp.get_language(lang))
        except Exception:
            _PARSERS[lang] = None
    return _PARSERS[lang]


def _ts_error_count(lang, text: str):
    """Count of ERROR/MISSING nodes — tree-sitter's two ways of flagging a fragment
    it couldn't parse — for `text` parsed as `lang`. None if we can't tell (no
    grammar / parse blew up) — a None means 'don't act', never 'clean'."""
    parser = _parser(lang)
    if parser is None:
        return None
    try:
        tree = parser.parse(text.encode("utf-8", "replace"))
    except Exception:
        return None
    count, stack = 0, [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            count += 1
        stack.extend(node.children)
    return count


# Prose/data formats whose tree-sitter grammars exist but whose "syntax errors" are
# noise for this gate: the language pack maps `.txt` to VIMDOC, so plain-text
# deliverable writes (a task's answer.txt or requirements.txt output) got
# grammar-checked and warned on exactly the deliverable-landing write. A missing entry
# here costs a spurious warning, never a missed real one — code languages are not
# listed.
_NON_CODE_LANGS = frozenset({
    "vimdoc", "markdown", "markdown_inline", "csv", "tsv", "psv", "text", "rst",
    "org", "diff", "gitcommit", "git_rebase", "gitattributes", "gitignore",
    "properties", "requirements",
})


# Grammars that reject idiomatic, valid code, so a warning from them is noise: each
# entry was caught by a valid file in tests/fixtures/syntaxgate_corpus (CSS nesting and
# `@container` range queries; SCSS `@each … in 1, 2` and `@extend %placeholder`; a
# Groovy GString times an int; a PowerShell `switch` condition block; Tcl's `;`
# command separator; V's `!` error return; NASM and GAS operands). A tree-sitter
# grammar lags its language, and this gate has no way to tell "the model broke the
# file" from "the grammar never learned this form", so these are not policed at all.
_UNRELIABLE_LANGS = frozenset({
    "css", "scss", "groovy", "powershell", "tcl", "v", "asm",
})

# JSON files that are JSONC by convention: their consumers accept comments and
# trailing commas, the strict grammar does not.
_JSONC_NAMES = ("tsconfig", "jsconfig", ".eslintrc", ".babelrc", ".swcrc", "devcontainer")
_JSONC_DIRS = ("/.vscode/", "/.devcontainer/")


def _code_lang(path: str) -> str | None:
    """`lang_for`, filtered to languages this gate should police: None for unknown
    extensions, the prose/data formats above, the grammars that reject valid code,
    and JSON files that are JSONC by convention."""
    detected = repomap.lang_for(path)
    if detected in _NON_CODE_LANGS or detected in _UNRELIABLE_LANGS:
        return None
    if detected == "json":
        name = os.path.basename(path)
        norm = "/" + path.replace(os.sep, "/")
        if (name.startswith(_JSONC_NAMES) or name.endswith(".code-workspace")
                or any(d in norm for d in _JSONC_DIRS)):
            return None
    return detected


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
    # A byte-order mark is the file's encoding signature, not source text: every
    # interpreter skips it, and `ast.parse` on the decoded string does not.
    after = after.removeprefix("\ufeff")
    if before is not None:
        before = before.removeprefix("\ufeff")

    lang = _code_lang(path)

    if lang == "python":
        try:
            ast.parse(after)
        except SyntaxError as e:
            lines = after.splitlines()
            line = lines[e.lineno - 1] if e.lineno and e.lineno <= len(lines) else ""
            # The check runs on chad's own interpreter, which may be older than the
            # project's; the version is the provenance the model needs to discount a
            # `type` statement or a newer form that is fine where the code will run.
            py = "Python {}.{}".format(*sys.version_info[:2])
            return (f"\n[warning: the file no longer parses under {py} — {e.msg} at "
                    f"line {e.lineno}: {line.strip()!r}. Fix this before moving on.]")
        return None

    if lang:
        # The language pack calls every `.h` C, and most of them are C++ (namespaces,
        # classes, templates), which the C grammar rejects: 18 of the 19 warnings in one
        # 90-trial eval run were valid C++ headers. An ambiguous header is judged by
        # whichever grammar fits the file as it stood (the new content, for a new file).
        if lang == "c" and path.endswith(".h"):
            basis = after if before is None else before
            counts = {lg: _ts_error_count(lg, basis) for lg in ("c", "cpp")}
            lang = min(counts, key=lambda lg: (counts[lg] is None, counts[lg] or 0))
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
