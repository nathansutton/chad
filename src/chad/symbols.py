"""jedi-backed symbol EDITOR for chad — `replace_symbol` / `insert_symbol`.

This module edits Python code by symbol rather than by text: locate a function /
class / method by name and replace or insert around it, returning a unified diff.
jedi is a pure-Python static-analysis engine (already a dependency) that runs
in-process — no language-server subprocess, no JSON-RPC, no MCP.

Symbolic **reads / search / rename** (`overview`, `view_symbol`, `find_symbol`,
`find_refs`, `repo_map`, `rename_symbol`) live in `repomap.py`, the tree-sitter
backend — it's language-agnostic and adds LSP "used by" annotations. `tools.py`
routes reads there and only edits here. A jedi read backend used to live in this
module too; it was dead (nothing routed to it) and has been removed.

Name paths: a bare name ("generate") or a qualified path ("Engine/generate" or
"Engine.generate") to disambiguate a method from a free function of the same name.
"""

import difflib
import glob as _glob
import os

import jedi

from .ignore import IGNORE_DIRS, slash_wrapped

_SKIP = slash_wrapped(IGNORE_DIRS)


def _norm_path(parts):
    return [p for p in parts.replace(".", "/").split("/") if p]


class SymbolService:
    """jedi-backed symbol editor, rooted at a project directory."""

    def __init__(self, root: str = "."):
        self.root = os.path.abspath(root)
        self._project = jedi.Project(self.root)

    # -- helpers ----------------------------------------------------------

    def _py_files(self):
        files = _glob.glob(os.path.join(self.root, "**", "*.py"), recursive=True)
        return sorted(f for f in files
                      if not any(d in "/" + f.replace(os.sep, "/") for d in _SKIP))[:20000]

    def _candidate_files(self, needle, path=None, should_stop=None):
        """Files that textually contain `needle` — a cheap prefilter so we only run
        (expensive) jedi analysis on the handful of files that could possibly match,
        instead of every .py in the project. Interruptible between files."""
        files = [path] if path else self._py_files()
        out = []
        for fp in files:
            if should_stop and should_stop():
                break
            if not (os.path.isfile(fp) and fp.endswith(".py")):
                continue
            try:
                with open(fp, errors="ignore") as f:
                    if needle in f.read():
                        out.append(fp)
            except OSError:
                continue
        return out

    def _script(self, path):
        with open(path, errors="replace") as f:
            code = f.read()
        return jedi.Script(code=code, path=path, project=self._project), code

    @staticmethod
    def _is_def(code, name_obj):
        """True only for functions/classes DEFINED in this file. Keyed off the name's
        own line (cheap, in-file) — NOT get_definition_start_position(), which resolves
        imports and can make jedi follow a name into site-packages and crash there."""
        if name_obj.type not in ("function", "class"):
            return False
        try:
            line = code.splitlines()[name_obj.line - 1]
        except (IndexError, TypeError):
            return False
        return line.lstrip().startswith(("def ", "class ", "async def "))

    @staticmethod
    def _matches(name_obj, parts):
        """True if a jedi Name matches the (possibly qualified) name-path parts."""
        if name_obj.name != parts[-1]:
            return False
        if len(parts) == 1:
            return True
        try:
            parent = name_obj.parent()
        except Exception:
            return False
        return parent is not None and parent.name == parts[-2]

    @staticmethod
    def _span(name_obj):
        """(start_line, end_line), 1-based, of a LOCAL definition; falls back to the
        name's own line if jedi can't produce a range."""
        try:
            a = name_obj.get_definition_start_position()[0]
            b = name_obj.get_definition_end_position()[0]
            if a and b and b >= a:
                return a, b
        except Exception:
            pass
        return name_obj.line, name_obj.line

    def _find_defs(self, parts, path=None, should_stop=None):
        """Definitions (function/class) matching the name path. Only files that
        textually contain the name are jedi-parsed (fast prefilter); errors on any
        single file or symbol are swallowed so one bad file can't break the search."""
        hits = []
        for fp in self._candidate_files(parts[-1], path, should_stop):
            if should_stop and should_stop():
                break
            try:
                s, code = self._script(fp)
                names = s.get_names(all_scopes=True, definitions=True, references=False)
            except Exception:   # jedi can raise KeyError/RecursionError/etc.
                continue
            for n in names:
                try:
                    if self._is_def(code, n) and self._matches(n, parts):
                        hits.append((n, fp, code))
                except Exception:
                    continue
        return hits

    def _rel(self, p):
        try:
            return os.path.relpath(p, self.root)
        except ValueError:
            return p

    def _disambig(self, name, hits):
        opts = "\n".join(f"  {self._rel(fp)}:{n.line}  {n.full_name}" for n, fp, _ in hits)
        return f"[{len(hits)} symbols named '{name}'; pass path= to disambiguate:]\n{opts}"

    # -- edit -------------------------------------------------------------

    def _apply(self, fp, code, a, b, new_lines, label):
        lines = code.splitlines()
        old_block = lines[a - 1:b]
        updated = lines[:a - 1] + new_lines + lines[b:]
        text = "\n".join(updated)
        if code.endswith("\n"):
            text += "\n"
        with open(fp, "w") as f:
            f.write(text)
        diff = [d for d in difflib.unified_diff(old_block, new_lines, lineterm="", n=1)
                if not d.startswith(("---", "+++", "@@"))]
        adds = sum(d.startswith("+") for d in diff)
        dels = sum(d.startswith("-") for d in diff)
        return f"[{label} in {self._rel(fp)}: +{adds} -{dels}]\n" + "\n".join(diff)

    def _locate_one(self, name, path, should_stop=None):
        parts = _norm_path(name)
        hits = self._find_defs(parts, path, should_stop)
        if not hits:
            return None, f"[symbol not found: {name}]"
        if len(hits) > 1 and path is None:
            return None, self._disambig(name, hits)
        return hits[0], None

    def replace_symbol(self, name: str, new: str, path=None, should_stop=None) -> str:
        hit, err = self._locate_one(name, path, should_stop)
        if err:
            return err
        n, fp, code = hit
        if not new.strip():
            return "[refusing to replace a symbol with empty content]"
        a, b = self._span(n)
        return self._apply(fp, code, a, b, new.rstrip("\n").split("\n"),
                           f"replaced {n.full_name or name}")

    def insert_symbol(self, name: str, code_text: str, where: str = "after",
                      path=None, should_stop=None) -> str:
        hit, err = self._locate_one(name, path, should_stop)
        if err:
            return err
        n, fp, code = hit
        if not code_text.strip():
            return "[nothing to insert]"
        a, b = self._span(n)
        new_lines = code_text.rstrip("\n").split("\n")
        if where == "before":  # insert just above the symbol's first line
            return self._apply(fp, code, a, a - 1, new_lines + [""],
                               f"inserted before {n.full_name or name}")
        # after (default): splice in just past the symbol's last line
        return self._apply(fp, code, b + 1, b, [""] + new_lines,
                           f"inserted after {n.full_name or name}")


_SERVICE = None


def service() -> SymbolService:
    """Lazily-bound, cwd-rooted service (the project the agent is operating in)."""
    global _SERVICE
    if _SERVICE is None or _SERVICE.root != os.path.abspath(os.getcwd()):
        _SERVICE = SymbolService(os.getcwd())
    return _SERVICE
