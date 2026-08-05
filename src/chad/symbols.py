"""Symbol EDITOR for chad — `replace_symbol` / `insert_symbol`, all languages.

This module edits code by symbol rather than by text: locate a function / class /
method by name and replace or insert around it, returning a unified diff. ONE
backend for every language: the tree-sitter repo map's definition tags
(`repomap._find_defs`), whose `line`/`end_line` spans are the same ones
`view_symbol` displays — so what the model just viewed is exactly what an edit
replaces. The splice engine (`_apply` + syntaxgate) reverts any edit that takes a
parsing file to a syntax error.

A jedi backend used to own the Python half of this split (scope-aware, qualified
paths — Python-only). Qualified paths now resolve for every language via the
Tags' scope chains (span containment + receiver/impl context; see repomap.Tag),
so the split — and the jedi dependency — is gone.

Name paths: a bare name ("generate") or a qualified path ("Engine/generate" or
"Engine.generate") to disambiguate a method from a free function of the same
name, in any language the repo map parses. A name defined in several files
returns a disambiguation listing; pass path= to pick one.

Symbolic **reads / search / rename** (`overview`, `view_symbol`, `find_symbol`,
`find_refs`, `repo_map`, `rename_symbol`) live in `repomap.py`; `tools.py`
routes reads there and only edits here.
"""

import difflib
import os

from . import syntaxgate

_DISAMBIG_MAX_LINES = 20  # cap disambiguation listings (205 lines of 'main' is prefill)


class SymbolService:
    """Symbol editor rooted at a project directory: tree-sitter definition spans,
    every language the repo map parses, qualified paths included."""

    def __init__(self, root: str = "."):
        self.root = os.path.abspath(root)
        self._rm = None   # private RepoMap, only if roots diverge (tests)

    def _repomap(self):
        """The repo map rooted at OUR root — normally the cwd-keyed singleton (shared
        warm cache); a private instance only when the roots diverge (tests)."""
        from . import repomap
        svc = repomap.service()
        if svc.root == self.root:
            return svc
        if self._rm is None:
            self._rm = repomap.RepoMap(self.root)
        return self._rm

    def _rel(self, p):
        try:
            return os.path.relpath(p, self.root)
        except ValueError:
            return p

    @staticmethod
    def _label(tag):
        return ".".join(tag.scope + (tag.name,))

    def _disambig(self, name, tags):
        opts = "\n".join(f"  {t.rel}:{t.line}  {t.kind} {self._label(t)}"
                         for t in tags[:_DISAMBIG_MAX_LINES])
        if len(tags) > _DISAMBIG_MAX_LINES:
            opts += f"\n  … (+{len(tags) - _DISAMBIG_MAX_LINES} more)"
        return f"[{len(tags)} symbols named '{name}'; pass path= to disambiguate:]\n{opts}"

    def _locate_one(self, name, path, should_stop=None):
        """Resolve `name` (optionally within `path`) to one (start, end, file, code,
        label) splice spec, or (None, error-message)."""
        try:
            tags = self._repomap()._find_defs(name, path, should_stop)
        except Exception:   # a broken map must degrade to "not found", not a crash
            tags = []
        if len(tags) > 1 and path is None:
            return None, self._disambig(name, tags)
        if not tags:
            return None, f"[symbol not found: {name}]"
        t = tags[0]
        try:
            with open(t.path, errors="replace") as f:
                code = f.read()
        except OSError:
            return None, f"[could not read {t.rel}]"
        return (t.line, t.end_line, t.path, code, self._label(t)), None

    # -- edit -------------------------------------------------------------

    def _apply(self, fp, code, a, b, new_lines, label):
        lines = code.splitlines()
        old_block = lines[a - 1:b]
        updated = lines[:a - 1] + new_lines + lines[b:]
        text = "\n".join(updated)
        if code.endswith("\n"):
            text += "\n"
        # Same contract as the line/string edit tools: a symbol edit that
        # takes a parsing file to a SyntaxError is reverted, not landed-with-warning.
        # Here the placement is ours and correct by construction, so the break can only
        # come from the model's own `new` code — say so instead of the generic steer.
        reject = syntaxgate.edit_reject(fp, code, text, (a, b))
        if reject:
            return (reject.rstrip("]").rstrip()
                    + "\n (the syntax error is inside the code you sent — fix it and "
                    "re-send the complete definition.)]")
        with open(fp, "w") as f:
            f.write(text)
        diff = [d for d in difflib.unified_diff(old_block, new_lines, lineterm="", n=1)
                if not d.startswith(("---", "+++", "@@"))]
        adds = sum(d.startswith("+") for d in diff)
        dels = sum(d.startswith("-") for d in diff)
        result = f"[{label} in {self._rel(fp)}: +{adds} -{dels}]\n" + "\n".join(diff)
        warn = syntaxgate.check_syntax(fp, code)  # `code` is the pre-edit content
        if warn:
            result += warn
        drift = syntaxgate.drift_warn(fp, code, text)
        if drift:
            result += drift
        from . import levers, lsp
        if levers.enabled("post_edit_diagnostics"):
            # The semantic tier above syntaxgate: the language server's typecheck of
            # the landed edit ("" when no server is warm for this language).
            note = lsp.diagnostics_note(fp)
            if note:
                levers.fired("post_edit_diagnostics")
            result += note
        return result

    def replace_symbol(self, name: str, new: str, path=None, should_stop=None) -> str:
        hit, err = self._locate_one(name, path, should_stop)
        if err:
            return err
        a, b, fp, code, label = hit
        if not new.strip():
            return "[refusing to replace a symbol with empty content]"
        return self._apply(fp, code, a, b, new.rstrip("\n").split("\n"),
                           f"replaced {label}")

    def insert_symbol(self, name: str, code_text: str, where: str = "after",
                      path=None, should_stop=None) -> str:
        hit, err = self._locate_one(name, path, should_stop)
        if err:
            return err
        a, b, fp, code, label = hit
        if not code_text.strip():
            return "[nothing to insert]"
        new_lines = code_text.rstrip("\n").split("\n")
        if where == "before":  # insert just above the symbol's first line
            return self._apply(fp, code, a, a - 1, new_lines + [""],
                               f"inserted before {label}")
        # after (default): splice in just past the symbol's last line
        return self._apply(fp, code, b + 1, b, [""] + new_lines,
                           f"inserted after {label}")


_SERVICE = None


def service() -> SymbolService:
    """Lazily-bound, cwd-rooted service (the project the agent is operating in)."""
    global _SERVICE
    if _SERVICE is None or _SERVICE.root != os.path.abspath(os.getcwd()):
        _SERVICE = SymbolService(os.getcwd())
    return _SERVICE
