"""Symbol EDITOR for chad — `replace_symbol` / `insert_symbol`, all languages.

This module edits code by symbol rather than by text: locate a function / class /
method by name and replace or insert around it, returning a unified diff. Two
backends share one splice engine (`_apply` + syntaxgate):

* **Python** goes through jedi (already a dependency, in-process, scope-aware,
  understands qualified `Class/method` paths).
* **Every other language** goes through the tree-sitter repo map's definition tags
  (`repomap._find_defs`), whose `line`/`end_line` spans are the same ones
  `view_symbol` displays — so what the model just viewed is exactly what an edit
  replaces. No language-server subprocess on either path.

Symbolic **reads / search / rename** (`overview`, `view_symbol`, `find_symbol`,
`find_refs`, `repo_map`, `rename_symbol`) live in `repomap.py`, the tree-sitter
backend — it's language-agnostic and adds LSP "used by" annotations. `tools.py`
routes reads there and only edits here. A jedi read backend used to live in this
module too; it was dead (nothing routed to it) and has been removed.

Name paths: a bare name ("generate") or a qualified path ("Engine/generate" or
"Engine.generate") to disambiguate a method from a free function of the same name
(qualified paths are jedi-only; for other languages pass path=).
"""

import difflib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import jedi

from . import syntaxgate
from .ignore import IGNORE_DIRS

# How long a walked .py file list stays fresh. Agent edits come in bursts (locate,
# replace, re-replace after a failed test); re-walking an 11k-file tree for each one
# measured ~200ms of pure repetition. 30s is short enough that a file created by a
# `write`/`bash` step is visible by the time the model turns back to symbol edits.
_PY_FILES_TTL_S = 30.0
_NAMES_CACHE_MAX = 16   # (path, mtime)-keyed jedi parses; re-edits of one file are common
_MAX_DEF_FILES = 16     # more definers than this -> disambiguate by file, skip jedi
_DISAMBIG_MAX_LINES = 20  # cap disambiguation listings (205 lines of 'main' is prefill)


def _norm_path(parts):
    return [p for p in parts.replace(".", "/").split("/") if p]


class SymbolService:
    """Symbol editor rooted at a project directory: jedi for Python, tree-sitter
    definition spans for every other language the repo map parses."""

    def __init__(self, root: str = "."):
        self.root = os.path.abspath(root)
        self._project = jedi.Project(self.root)
        self._py_files_cache: list = []
        self._py_files_at = 0.0
        self._names_cache: dict = {}  # path -> (mtime, names, code)
        self._rm = None               # private RepoMap, only if roots diverge (tests)

    # -- helpers ----------------------------------------------------------

    def _py_files(self):
        now = time.monotonic()
        if self._py_files_cache and now - self._py_files_at < _PY_FILES_TTL_S:
            return self._py_files_cache
        out = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            # prune ignored + hidden dirs during the walk (never descend into a
            # node_modules/ or .venv/ at all) instead of filtering the full list after
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(".") and d not in IGNORE_DIRS]
            out.extend(os.path.join(dirpath, fn) for fn in filenames
                       if fn.endswith(".py") and not fn.startswith("."))
        self._py_files_cache = sorted(out)[:20000]
        self._py_files_at = now
        return self._py_files_cache

    def _candidate_files(self, needle, path=None, should_stop=None):
        """Files that could DEFINE `needle` — a cheap prefilter so we only run
        (expensive) jedi analysis on the handful of files that could possibly match,
        instead of every .py in the project. A hit must put the name on a def/class
        line (`_is_def`'s contract), so the probe is that textual shape, not a bare
        substring — `"main" in text` matched thousands of files ("domain", comments,
        `__main__` guards) and each false candidate cost a full jedi parse: measured
        130s for _find_defs("main") on an 11k-file repo. Reads are farmed to a thread
        pool (I/O releases the GIL); order stays deterministic."""
        probe = re.compile(rf"\b(?:def|class)\s+{re.escape(needle)}\b")
        files = [path] if path else self._py_files()
        files = [fp for fp in files if os.path.isfile(fp) and fp.endswith(".py")]

        def scan(chunk):
            hits = []
            for fp in chunk:
                if should_stop and should_stop():
                    break
                try:
                    with open(fp, errors="ignore") as f:
                        text = f.read()
                    # substring first: C-speed rejection of the ~all files that don't
                    # even contain the name; only those pay the regex probe
                    if needle in text and probe.search(text):
                        hits.append(fp)
                except OSError:
                    continue
            return hits

        # 8 contiguous chunks, one future each: per-file futures measured ~140ms of
        # pure executor overhead on 4.5k files. Concatenating in order keeps the
        # deterministic sorted result.
        step = max(1, -(-len(files) // 8))
        chunks = [files[i:i + step] for i in range(0, len(files), step)]
        out = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for hits in pool.map(scan, chunks):
                out.extend(hits)
        return out

    def _script(self, path):
        with open(path, errors="replace") as f:
            code = f.read()
        return jedi.Script(code=code, path=path, project=self._project), code

    @staticmethod
    def _is_def(code_lines, name_obj):
        """True only for functions/classes DEFINED in this file. Keyed off the name's
        own line (cheap, in-file) — NOT get_definition_start_position(), which resolves
        imports and can make jedi follow a name into site-packages and crash there.
        Takes pre-split lines: jedi returns tens of thousands of names for a big
        module, and splitting the source once per name measured ~850ms/file."""
        if name_obj.type not in ("function", "class"):
            return False
        try:
            line = code_lines[name_obj.line - 1]
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

    def _names(self, fp):
        """jedi names + source for one file, cached by (path, mtime). An agent edits
        the same file repeatedly (replace, test, re-replace); each jedi parse of a big
        module costs 100ms+, so a re-locate in an unchanged file must not re-pay it.
        Our own _apply rewrites the file, which changes its mtime and evicts."""
        try:
            mt = os.path.getmtime(fp)
        except OSError:
            mt = None
        hit = self._names_cache.get(fp)
        if hit is not None and hit[0] == mt:
            return hit[1], hit[2]
        s, code = self._script(fp)
        names = s.get_names(all_scopes=True, definitions=True, references=False)
        if len(self._names_cache) >= _NAMES_CACHE_MAX:
            self._names_cache.pop(next(iter(self._names_cache)))
        self._names_cache[fp] = (mt, names, code)
        return names, code

    def _find_defs(self, parts, path=None, should_stop=None, files=None):
        """Definitions (function/class) matching the name path. Only files that
        could define the name are jedi-parsed (fast prefilter); errors on any
        single file or symbol are swallowed so one bad file can't break the search.
        `files` short-circuits the prefilter when the caller already ran it."""
        hits = []
        if files is None:
            files = self._candidate_files(parts[-1], path, should_stop)
        for fp in files:
            if should_stop and should_stop():
                break
            try:
                names, code = self._names(fp)
            except Exception:   # jedi can raise KeyError/RecursionError/etc.
                continue
            code_lines = code.splitlines()
            for n in names:
                try:
                    # name equality first: it filters ~all of a big module's names
                    # before the (comparatively pricey) definition-line check runs
                    if self._matches(n, parts) and self._is_def(code_lines, n):
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
        opts = "\n".join(f"  {self._rel(fp)}:{n.line}  {n.full_name}"
                         for n, fp, _ in hits[:_DISAMBIG_MAX_LINES])
        if len(hits) > _DISAMBIG_MAX_LINES:
            opts += f"\n  … (+{len(hits) - _DISAMBIG_MAX_LINES} more)"
        return f"[{len(hits)} symbols named '{name}'; pass path= to disambiguate:]\n{opts}"

    # -- edit -------------------------------------------------------------

    def _apply(self, fp, code, a, b, new_lines, label):
        lines = code.splitlines()
        old_block = lines[a - 1:b]
        updated = lines[:a - 1] + new_lines + lines[b:]
        text = "\n".join(updated)
        if code.endswith("\n"):
            text += "\n"
        # Same contract as the line/string edit tools (plan 073): a symbol edit that
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
        self._names_cache.pop(fp, None)  # mtime would evict anyway; don't rely on it
        diff = [d for d in difflib.unified_diff(old_block, new_lines, lineterm="", n=1)
                if not d.startswith(("---", "+++", "@@"))]
        adds = sum(d.startswith("+") for d in diff)
        dels = sum(d.startswith("-") for d in diff)
        result = f"[{label} in {self._rel(fp)}: +{adds} -{dels}]\n" + "\n".join(diff)
        warn = syntaxgate.check_syntax(fp, code)  # `code` is the pre-edit content
        return result + warn if warn else result

    # -- locate: jedi for .py, tree-sitter tags for everything else --------

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

    def _ts_locate(self, name, path, should_stop=None):
        """Non-Python definitions of `name` from the tree-sitter repo map, as
        (start, end, path, code, label) splice specs. The map's `line`/`end_line`
        spans are what view_symbol shows, so an edit replaces what was viewed."""
        try:
            tags = self._repomap()._find_defs(name, path, should_stop)
        except Exception:   # a broken map must degrade to "not found", not a crash
            return []
        out = []
        for t in tags:
            if t.path.endswith(".py"):
                continue    # jedi owns Python — scope-aware and qualified-path capable
            try:
                with open(t.path, errors="replace") as f:
                    code = f.read()
            except OSError:
                continue
            out.append((t.line, t.end_line, t.path, code, f"{t.kind} {t.name}"))
        return out

    def _locate_one(self, name, path, should_stop=None):
        """Resolve `name` (optionally within `path`) to one (start, end, file, code,
        label) splice spec, or (None, error-message). Python resolves via jedi;
        misses fall through to the tree-sitter map so every language the map parses
        is editable. A name found in BOTH resolves to Python (the pre-existing
        behavior when other languages were invisible)."""
        parts = _norm_path(name)
        py_target = path is None or path.endswith(".py")
        hits: list = []
        if py_target:
            for _attempt in (0, 1):
                cands = self._candidate_files(parts[-1], path, should_stop)
                if path is None and len(cands) > _MAX_DEF_FILES:
                    # A name defined in this many files can't be located without
                    # path= anyway; jedi-parsing every definer just to say
                    # "ambiguous" measured 60s for 'forward' on the pytorch clone.
                    listing = "\n".join(f"  {self._rel(fp)}"
                                        for fp in cands[:_DISAMBIG_MAX_LINES])
                    if len(cands) > _DISAMBIG_MAX_LINES:
                        listing += f"\n  … (+{len(cands) - _DISAMBIG_MAX_LINES} more)"
                    return None, (f"['{parts[-1]}' is defined in {len(cands)} files; "
                                  f"pass path= to pick one:]\n{listing}")
                hits = self._find_defs(parts, path, should_stop, files=cands)
                if hits or path is not None or time.monotonic() - self._py_files_at <= 1.0:
                    break
                # The symbol may live in a file created after the cached walk (the
                # TTL trades that staleness for burst speed). Before reporting
                # not-found, re-walk once — only genuine misses pay this.
                self._py_files_at = 0.0
        if hits:
            if len(hits) > 1 and path is None:
                return None, self._disambig(name, hits)
            n, fp, code = hits[0]
            a, b = self._span(n)
            return (a, b, fp, code, n.full_name or name), None
        if path is None or not py_target:
            ts = self._ts_locate(parts[-1], path, should_stop)
            if len(ts) > 1 and path is None:
                opts = "\n".join(f"  {self._rel(fp)}:{a}  {label}"
                                 for a, _b, fp, _c, label in ts[:_DISAMBIG_MAX_LINES])
                if len(ts) > _DISAMBIG_MAX_LINES:
                    opts += f"\n  … (+{len(ts) - _DISAMBIG_MAX_LINES} more)"
                return None, (f"[{len(ts)} symbols named '{parts[-1]}'; pass path= "
                              f"to disambiguate:]\n{opts}")
            if ts:
                return ts[0], None
        return None, f"[symbol not found: {name}]"

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
