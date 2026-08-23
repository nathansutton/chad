"""Tree-sitter tags extraction for the bash-route ambient levers.

chad's tool surface is bash-first, so nothing here is a model-facing tool. What
survives of the 1.x symbolic layer is exactly what the result channel needs:

* `lang_for` — extension→language detection (also used by the syntax warning);
* `RepoMap._extract` — per-file definition Tags (name, kind, line span, scope),
  mtime-cached in memory and on disk, feeding the one-line skeleton appended the
  first time a file's content comes back through bash;
* `RepoMap._find_defs` — cross-file definition lookup, feeding the
  "this came back empty; `x` is defined at rel:line" pointer when a bash grep
  for a known symbol returns nothing.

`tree-sitter-language-pack` ships ~300 grammars (downloaded + cached on first
use) AND the `tags.scm` queries that mark every definition/reference — so
extraction is language-agnostic with no language-server subprocess. Whole-repo
extraction is mtime-cached on disk per repo and sharded across subprocess
workers on a cold scan (see `_extract_all`).
"""

import hashlib
import logging
import os
import pickle
import subprocess
import sys
import threading
import time
from collections import namedtuple

# tree-sitter-language-pack ships native (maturin/pyo3) wheels. On a platform with no
# matching wheel — e.g. a container running emulated amd64 — uv falls back to a Rust
# source build that can fail, and this module-level import used to take ALL of chad down
# with it: `tools.py` imports `repomap`, so a missing parser killed bash/read/edit too,
# and the run errored out before the agent took a single step. Symbol
# ranking is the only thing that actually needs it; `lang_for` and `_lang_tools` already
# return None on any failure, so the rest of the toolset degrades cleanly instead.
try:
    import tree_sitter_language_pack as tlp
    from tree_sitter import Parser, Query, QueryCursor
except ImportError:  # pragma: no cover — exercised only on wheel-less platforms
    tlp = None                              # type: ignore[assignment]
    Parser = Query = QueryCursor = None     # type: ignore[assignment,misc]

from . import config
from .ignore import IGNORE_DIRS, REPOMAP_EXTRA

log = logging.getLogger("chad")

# repomap indexes the whole repo, so it skips the base set PLUS model weights, installed
# packages, and caches (REPOMAP_EXTRA) — a symbol editor doesn't need those exclusions.
_SKIP_NAMES = frozenset(IGNORE_DIRS + REPOMAP_EXTRA)

_MAX_FILE_BYTES = 1_000_000   # skip anything bigger than ~1MB (generated/minified)
_MAX_FILES = 20000
# -- whole-repo extraction scaling -----------------------------------------------
# The tree-sitter parse loop is the dominant cost of a cold scan (measured 17.5s of a
# 19s repo_map on an 11k-file repo) and py-tree-sitter never releases the GIL, so
# threads don't help (measured 1.0x). Two levers instead:
#   1. a per-repo on-disk tags cache keyed by mtime, so a warm scan parses only what
#      changed (measured 0.34s to load 11k files' tags vs 17.5s to re-parse), and
#   2. subprocess workers for cold scans (measured 5.5x on 8 workers). Workers run
#      `python -c` importing ONLY chad.repomap (0.02s, no mlx) — never fork, and never
#      re-import chad's entry point, which would drag the whole MLX engine into each.
_PARALLEL_MIN_FILES = 200   # below this, worker startup costs more than it saves
_CACHE_SAVE_MIN = 32        # don't persist a cache for tiny repos (or tiny test fixtures)
_CACHE_VERSION = 3          # bump when the entry shape or tags queries change
_CACHE_DIR = os.path.expanduser("~/.chad/cache/repomap")

_WORKER_SRC = """\
import pickle, sys
from chad.repomap import RepoMap
root, paths = pickle.load(sys.stdin.buffer)
rm = RepoMap(root)
for p in paths:
    rm._extract(p)
sys.stdout.buffer.write(pickle.dumps(rm._cache, protocol=pickle.HIGHEST_PROTOCOL))
"""


def _worker_count() -> int:
    n = config.env_int("CHAD_REPOMAP_WORKERS", 0) or 0
    if n > 0:
        return n
    return max(1, min(8, (os.cpu_count() or 4) - 2))


# Chad-owned tags queries for languages where the pack's are missing or broken
# (measured on tests/fixtures/polyglot, 2026-07-29): typescript/tsx ship NO query
# (0 chars); php's names `method_call_expression`, which current tree-sitter-php
# renamed `member_call_expression`, so Query() refuses to compile the whole thing;
# c has no @reference captures and tags header prototypes as definitions; cpp
# misses out-of-class (`Engine::process`) and in-class method definitions.
_TS_TAGS = """\
(class_declaration name: (type_identifier) @name) @definition.class
(abstract_class_declaration name: (type_identifier) @name) @definition.class
(interface_declaration name: (type_identifier) @name) @definition.interface
(type_alias_declaration name: (type_identifier) @name) @definition.type
(enum_declaration name: (identifier) @name) @definition.type
(internal_module name: (identifier) @name) @definition.module
(function_declaration name: (identifier) @name) @definition.function
(generator_function_declaration name: (identifier) @name) @definition.function
((method_definition name: (property_identifier) @name) @definition.method
 (#not-eq? @name "constructor"))
(method_signature name: (property_identifier) @name) @definition.method
(lexical_declaration (variable_declarator
  name: (identifier) @name
  value: [(arrow_function) (function_expression)]) @definition.function)
(variable_declaration (variable_declarator
  name: (identifier) @name
  value: [(arrow_function) (function_expression)]) @definition.function)
((call_expression function: (identifier) @name) @reference.call
 (#not-match? @name "^(require)$"))
(call_expression function: (member_expression
  property: (property_identifier) @name)) @reference.call
(new_expression constructor: (identifier) @name) @reference.class
"""

# go and rust attach methods without lexical nesting (receivers / impl blocks), so
# span containment alone can't derive `Engine/Process` there. Their overrides add
# @context.receiver (the method's receiver type) and @context.scope (the impl
# block's span) on top of the pack's own captures.
_TAGS_OVERRIDE = {
    "typescript": _TS_TAGS,
    "tsx": _TS_TAGS,
    # The pack ships NO bash tags query, yet shell is everywhere real.  Without
    # tags, every .sh function is invisible to repo_map/overview/view_symbol and the
    # symbol editor. Functions (both `f() {}` and `function f {}` parse to the same
    # node), TOP-LEVEL variable assignments only (anchored to program: config
    # constants surface, function-local temps don't), and command invocations as
    # references (builtins like echo/set resolve to no definer, so the rank graph
    # ignores them; invocations of repo-defined functions are exactly find_refs).
    "bash": """\
(function_definition name: (word) @name) @definition.function
(program (variable_assignment name: (variable_name) @name) @definition.constant)
(command name: (command_name (word) @name)) @reference.call
""",
    "go": """\
(function_declaration name: (identifier) @name) @definition.function
(method_declaration
  receiver: (parameter_list (parameter_declaration
    type: [(type_identifier) @context.receiver
           (pointer_type (type_identifier) @context.receiver)]))
  name: (field_identifier) @name) @definition.method
(type_spec name: (type_identifier) @name) @definition.type
(call_expression function: [
  (identifier) @name
  (parenthesized_expression (identifier) @name)
  (selector_expression field: (field_identifier) @name)
  (parenthesized_expression (selector_expression field: (field_identifier) @name))
]) @reference.call
(type_identifier) @name @reference.type
""",
    "rust": """\
(struct_item name: (type_identifier) @name) @definition.class
(enum_item name: (type_identifier) @name) @definition.class
(union_item name: (type_identifier) @name) @definition.class
(type_item name: (type_identifier) @name) @definition.class
(declaration_list (function_item name: (identifier) @name) @definition.method)
(function_item name: (identifier) @name) @definition.function
(trait_item name: (type_identifier) @name) @definition.interface
(mod_item name: (identifier) @name) @definition.module
(macro_definition name: (identifier) @name) @definition.macro
(call_expression function: (identifier) @name) @reference.call
(call_expression function: (field_expression
  field: (field_identifier) @name)) @reference.call
(macro_invocation macro: (identifier) @name) @reference.call
(impl_item trait: (type_identifier) @name) @reference.implementation
(impl_item type: (type_identifier) @name !trait) @reference.implementation
(impl_item type: (type_identifier) @name) @context.scope
(impl_item type: (generic_type type: (type_identifier) @name)) @context.scope
""",
    "php": """\
(class_declaration name: (name) @name) @definition.class
(interface_declaration name: (name) @name) @definition.interface
(method_declaration name: (name) @name) @definition.method
(function_definition name: (name) @name) @definition.function
(class_declaration (base_clause (name) @name)) @reference.implementation
(class_declaration (class_interface_clause (name) @name)) @reference.implementation
(interface_declaration (base_clause (name) @name)) @reference.implementation
(function_call_expression function: (name) @name) @reference.call
(member_call_expression name: (name) @name) @reference.call
(scoped_call_expression name: (name) @name) @reference.call
""",
    "c": """\
(function_definition declarator: (function_declarator
  declarator: (identifier) @name)) @definition.function
(function_definition declarator: (pointer_declarator
  declarator: (function_declarator
    declarator: (identifier) @name))) @definition.function
(struct_specifier name: (type_identifier) @name body: (_)) @definition.class
(declaration type: (union_specifier name: (type_identifier) @name)) @definition.class
(type_definition declarator: (type_identifier) @name) @definition.type
(enum_specifier name: (type_identifier) @name) @definition.type
(call_expression function: (identifier) @name) @reference.call
""",
    "cpp": """\
(class_specifier name: (type_identifier) @name) @definition.class
(struct_specifier name: (type_identifier) @name body: (_)) @definition.class
(function_definition declarator: (function_declarator
  declarator: (identifier) @name)) @definition.function
(function_definition declarator: (function_declarator
  declarator: (qualified_identifier scope: (_) @context.receiver
               name: (identifier) @name))) @definition.method
(function_definition declarator: (function_declarator
  declarator: (field_identifier) @name)) @definition.method
(class_specifier (base_class_clause (type_identifier) @name)) @reference.implementation
(call_expression function: (identifier) @name) @reference.call
(call_expression function: (field_expression
  field: (field_identifier) @name)) @reference.call
""",
}

# A definition discovered by tree-sitter. `kind` is the tag suffix (function,
# class, method, constant, ...); `sig` is the collapsed header line(s). `name_row`
# and `name_col` are the 0-based position of the identifier itself (what an LSP
# wants for go-to-def / find-references). `scope` is the enclosing-symbol chain,
# outermost first (("Engine",) for a method, ("Outer", "Inner") when nested) — how
# a qualified path like `Engine/generate` resolves in any language: by span
# containment where methods nest lexically, and by receiver/impl context captures
# (@context.receiver / @context.scope) where they don't (Go, Rust, C++ Engine::).
Tag = namedtuple("Tag", "rel path name kind line end_line sig name_row name_col scope")


def _scope_chains(spans):
    """Enclosing-name chains by span containment: spans is [(start, end, name)],
    returns for each input its chain of strictly-containing names, outermost first.
    Tree-sitter spans nest or are disjoint (never partially overlap), so a single
    (start, -end)-ordered pass with a stack is exact and O(n log n)."""
    order = sorted(range(len(spans)), key=lambda i: (spans[i][0], -spans[i][1]))
    stack = []  # indices of open (containing) spans
    chains = [()] * len(spans)
    for i in order:
        start, end, _name = spans[i]
        while stack and spans[stack[-1]][1] < start:
            stack.pop()
        chains[i] = tuple(spans[j][2] for j in stack
                          if (spans[j][0], spans[j][1]) != (start, end))
        stack.append(i)
    return chains


def _qual_parts(name: str):
    """A symbol path ("Engine/generate" or "Engine.generate") as its segments."""
    return [p for p in name.replace(".", "/").split("/") if p]


class RepoMap:
    """Tree-sitter symbol intelligence rooted at a project directory."""

    def __init__(self, root: str = "."):
        self.root = os.path.abspath(root)
        self._tooling = {}   # lang -> (Parser, Query) | None
        self._cache = {}     # path -> (mtime, [defs], [(refname, rel, line)])
        self._files = None   # memoized completed _code_files() result; None = uncomputed
        self._files_at = 0.0  # when that walk completed (staleness retry guard)
        self._disk_checked = False  # the on-disk tags cache is loaded at most once

    # -- persistent tags cache ---------------------------------------------
    # Parsing is the dominant cost of a whole-repo scan and the in-memory cache dies
    # with the process, so every new session used to re-pay it in full. Entries are
    # mtime-validated per file on use (`_extract`'s existing contract), so a stale
    # disk entry is re-parsed, never trusted. Pickle is fine here trust-wise: the
    # cache lives under ~/.chad (0700/0600) — the same trust domain as sessions.

    def _cache_file(self) -> str:
        return os.path.join(
            _CACHE_DIR, hashlib.sha256(self.root.encode()).hexdigest()[:16] + ".pkl")

    def _load_disk_cache(self):
        if self._disk_checked:
            return
        self._disk_checked = True
        try:
            with open(self._cache_file(), "rb") as f:
                data = pickle.load(f)
            if (data.get("v") == _CACHE_VERSION
                    and data.get("tag_fields") == list(Tag._fields)
                    and data.get("root") == self.root):
                # in-memory (this session, freshest) entries win over disk ones
                self._cache = {**data["files"], **self._cache}
        except Exception:  # noqa: BLE001 - absent/corrupt/foreign cache: parse fresh
            pass

    def _save_disk_cache(self, keep):
        keep = set(keep)
        try:
            os.makedirs(_CACHE_DIR, mode=0o700, exist_ok=True)
            blob = pickle.dumps(
                {"v": _CACHE_VERSION, "tag_fields": list(Tag._fields), "root": self.root,
                 "files": {p: e for p, e in self._cache.items() if p in keep}},
                protocol=pickle.HIGHEST_PROTOCOL)
            tmp = self._cache_file() + ".tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
            os.replace(tmp, self._cache_file())
        except Exception:  # noqa: BLE001 - cache is an optimization, never a failure
            pass

    # -- whole-repo extraction ---------------------------------------------

    def _extract_all(self, files, should_stop=None):
        """Ensure every file's tags are cached: disk cache first, then subprocess
        workers for a large cold miss set, else the plain serial loop. Callers keep
        their per-file `_extract` loops — after this they're cache hits, and any
        shard a failed/killed worker didn't return degrades to a serial parse there."""
        self._load_disk_cache()
        misses = []
        for f in files:
            c = self._cache.get(f)
            try:
                if not c or c[0] != os.path.getmtime(f):
                    misses.append(f)
            except OSError:
                continue
        if not misses:
            return
        if len(misses) >= _PARALLEL_MIN_FILES and _worker_count() > 1:
            self._extract_parallel(misses, should_stop)
        else:
            for f in misses:
                if should_stop and should_stop():
                    return
                self._extract(f)
        parsed = sum(1 for f in misses if f in self._cache)
        if parsed >= _CACHE_SAVE_MIN and not (should_stop and should_stop()):
            self._save_disk_cache(files)

    def _extract_parallel(self, misses, should_stop=None):
        """Shard `misses` across `python -c` workers (size-sorted round-robin for
        balance) and merge their tag caches. Workers import only chad.repomap —
        never chad's entry point, which would pull the MLX engine into each. One
        reader thread per worker (communicate) so a big result can't deadlock the
        pipe; should_stop kills the lot."""
        def size_of(f):
            try:
                return os.path.getsize(f)
            except OSError:
                return 0
        nw = min(_worker_count(), max(1, len(misses) // 50))
        by_size = sorted(misses, key=size_of, reverse=True)
        shards = [by_size[i::nw] for i in range(nw)]
        procs, results, threads = [], [b""] * nw, []
        try:
            for i, shard in enumerate(shards):
                p = subprocess.Popen([sys.executable, "-c", _WORKER_SRC],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL)
                procs.append(p)

                def pump(p=p, i=i, payload=pickle.dumps((self.root, shard))):
                    try:
                        results[i] = p.communicate(payload)[0]
                    except Exception:  # noqa: BLE001 - a dead worker just loses its shard
                        results[i] = b""
                t = threading.Thread(target=pump, daemon=True)
                t.start()
                threads.append(t)
            while any(t.is_alive() for t in threads):
                if should_stop and should_stop():
                    for p in procs:
                        p.kill()
                for t in threads:
                    t.join(timeout=0.05)
        finally:
            for p in procs:
                if p.poll() is None:
                    p.kill()
        allowed = set(misses)
        for blob in results:
            if not blob:
                continue
            try:
                part = pickle.loads(blob)
            except Exception:  # noqa: BLE001 - partial write from a killed worker
                continue
            self._cache.update({p: e for p, e in part.items() if p in allowed})

    # -- tree-sitter plumbing --------------------------------------------

    def lang_for(self, path):
        try:
            return tlp.detect_language_from_path(path)
        except Exception:
            return None

    def _lang_tools(self, lang):
        """(Parser, tags-Query) for a language, lazily built and cached. Grammars
        download on first use; a language without a tags query yields None."""
        if lang not in self._tooling:
            try:
                language = tlp.get_language(lang)
                qsrc = _TAGS_OVERRIDE.get(lang) or tlp.get_tags_query(lang)
                if not qsrc:
                    self._tooling[lang] = None
                else:
                    self._tooling[lang] = (Parser(language), Query(language, qsrc))
            except Exception:
                self._tooling[lang] = None
        return self._tooling[lang]

    def _rel(self, p):
        try:
            return os.path.relpath(p, self.root)
        except ValueError:
            return p

    def _code_files(self, should_stop=None):
        # Memoized on the instance: service() is a cwd-cached singleton living for the
        # whole session, so once the tree is walked every later symbol lookup reuses the
        # list instead of re-walking the repo mid-turn. A scan interrupted by should_stop
        # is returned but NOT cached — caching a partial walk would hide real files.
        if self._files is not None:
            return self._files
        out = []
        interrupted = False
        # os.walk with in-place pruning: an ignored or hidden directory is never even
        # entered, where the old glob("**") enumerated every path under node_modules/
        # models/ before filtering (and followed symlinked dirs, risking cycles).
        for dirpath, dirnames, filenames in os.walk(self.root):
            if should_stop and should_stop():
                interrupted = True
                break
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(".") and d not in _SKIP_NAMES]
            for name in filenames:
                if name.startswith("."):
                    continue
                f = os.path.join(dirpath, name)
                try:
                    if not os.path.isfile(f) or os.path.getsize(f) > _MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                if self.lang_for(f):
                    out.append(f)
        result = sorted(out)[:_MAX_FILES]
        if not interrupted:        # never cache a partial scan
            self._files = result
            self._files_at = time.monotonic()
        return result

    @staticmethod
    def _header(src: bytes, node) -> str:
        """The definition's header line(s) — up to the line that closes the
        signature (ends in ':' or '{'), collapsed to one line. Language-agnostic."""
        seg = src[node.start_byte:node.end_byte]
        out = []
        for raw in seg.split(b"\n")[:4]:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                out.append(line)
            if line.endswith((":", "{")):
                break
        return " ".join(out)[:200]

    def _extract(self, path):
        """(defs, refs) for one file, cached on mtime. defs: [Tag]; refs:
        [(name, rel, line)]. Pairs each `@name` with the `@definition.*`/
        `@reference.*` it was captured with via per-pattern matches()."""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            # gone/unreadable: drop the stale entry so _aggregates unfolds its old
            # contribution instead of keeping a deleted file's symbols in the map
            self._cache.pop(path, None)
            return [], []
        cached = self._cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1], cached[2]

        defs, refs = [], []
        tools = self._lang_tools(self.lang_for(path))
        if tools:
            parser, query = tools
            try:
                with open(path, "rb") as f:
                    src = f.read()
                tree = parser.parse(src)
                matches = QueryCursor(query).matches(tree.root_node)
            except Exception:
                matches = []
                src = b""
            rel = self._rel(path)
            seen_spans = set()  # rust tags an impl fn as method AND function: one Tag
            raw = []      # (name, kind, line0, end0, sig, nrow, ncol, receiver)
            scopes = []   # (line0, end0, name) container spans that aren't symbols
                          # themselves (rust `impl Engine { … }` blocks)
            for _pat, caps in matches:
                name_nodes = caps.get("name")
                name = (src[name_nodes[0].start_byte:name_nodes[0].end_byte]
                        .decode("utf-8", "replace")) if name_nodes else None
                nrow = name_nodes[0].start_point[0] if name_nodes else 0
                ncol = name_nodes[0].start_point[1] if name_nodes else 0
                rnodes = caps.get("context.receiver")
                receiver = (src[rnodes[0].start_byte:rnodes[0].end_byte]
                            .decode("utf-8", "replace")) if rnodes else None
                for cap, nodes in caps.items():
                    if cap.startswith("definition") and name:
                        kind = cap.split(".", 1)[1] if "." in cap else "def"
                        for dn in nodes:
                            span = (name, dn.start_point[0], dn.end_point[0])
                            if span in seen_spans:
                                continue
                            seen_spans.add(span)
                            raw.append((name, kind, dn.start_point[0], dn.end_point[0],
                                        self._header(src, dn), nrow, ncol, receiver))
                    elif cap.startswith("reference") and name:
                        for rn in nodes:
                            refs.append((name, rel, rn.start_point[0] + 1))
                    elif cap.startswith("context.scope") and name:
                        for sn in nodes:
                            scopes.append((sn.start_point[0], sn.end_point[0], name))
            # Scope chains: lexical containment over this file's defs + context
            # spans, then the receiver (Go `func (e Engine)`, C++ `Engine::`)
            # appended where the language attaches methods without nesting them.
            spans = [(r[2], r[3], r[0]) for r in raw] + scopes
            chains = _scope_chains(spans)
            for i, (name, kind, l0, e0, sig, nrow, ncol, receiver) in enumerate(raw):
                scope = chains[i]
                if receiver:
                    scope = scope + tuple(_qual_parts(
                        receiver.replace("::", "/").lstrip("*").strip()))
                defs.append(Tag(rel, path, name, kind, l0 + 1, e0 + 1,
                                sig, nrow, ncol, scope))
        self._cache[path] = (mtime, defs, refs)
        return defs, refs

    # -- definition lookup ------------------------------------------------

    def _find_defs(self, name, path=None, should_stop=None):
        """Definition Tags matching `name` — a bare identifier or a qualified path
        ("Engine/generate", "Engine.generate"), any language. Qualified segments
        must be a suffix of the Tag's scope chain, so `Engine/generate` never
        silently resolves to a free `generate` (strict, like the jedi backend it
        replaced); bare names behave exactly as before."""
        parts = _qual_parts(name)
        if not parts:
            return []
        target, quals = parts[-1], tuple(parts[:-1])
        files = [os.path.join(self.root, path)] if path and not os.path.isabs(path) \
            else ([path] if path else self._code_files(should_stop))
        if not path:
            self._extract_all(files, should_stop)
        hits = []
        for f in files:
            if should_stop and should_stop():
                break
            defs, _ = self._extract(f)
            for d in defs:
                if d.name == target and (
                        not quals or d.scope[max(0, len(d.scope) - len(quals)):] == quals):
                    hits.append(d)
        if not hits and path is None and not (should_stop and should_stop()):
            # The symbol may live in a file created after the memoized walk. Before
            # reporting not-found, re-walk once — only genuine misses pay this.
            fresh = self._refresh_files()
            if fresh:
                return self._find_defs(name, path, should_stop)
        return hits

    def _refresh_files(self):
        """Invalidate the memoized file walk if it could be stale; True if a
        re-walk actually found a different file set (callers then retry their
        lookup). A walk under a second old is trusted — a burst of misses must
        not re-walk an 11k-file tree per call."""
        if self._files is None or time.monotonic() - self._files_at < 1.0:
            return False
        before = self._files
        self._files = None
        return self._code_files() != before

_SERVICE = None


def service() -> RepoMap:
    """Lazily-bound, cwd-rooted tags service (the project the agent is operating in)."""
    global _SERVICE
    if _SERVICE is None or _SERVICE.root != os.path.abspath(os.getcwd()):
        _SERVICE = RepoMap(os.getcwd())
    return _SERVICE


def lang_for(path) -> str | None:
    """Module-level language detection (extension-based, root-independent)."""
    return service().lang_for(path)
