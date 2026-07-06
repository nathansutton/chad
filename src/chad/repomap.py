"""Tree-sitter repo map + multi-language symbol intelligence for chad.

The dominant cost of a local coding model is *prefill*: every token of context the
model has never seen must be encoded before it can answer, and on Ornith's
non-trimmable cache a bloated transcript is expensive forever. The cheapest way to
keep prefills small is to never put a whole file in the transcript — give the model
a ranked *skeleton* (signatures only) to navigate by, and let it pull the one
symbol it actually needs.

This is the aider "repo map" idea, built in-process on tree-sitter:

* `tree-sitter-language-pack` ships ~300 grammars (downloaded + cached on first use)
  AND the `tags.scm` queries that mark every definition/reference — so symbol
  extraction is **language-agnostic** with no language-server subprocess to install
  (that precision layer comes later, behind this same surface, via solidlsp).
* `repo_map()` ranks definitions with personalized PageRank (rustworkx) over the
  file→symbol reference graph and renders the most central ones as elided signatures
  within a token budget. Whole-repo tag extraction is mtime-cached on disk per repo
  and sharded across subprocess workers on a cold scan (see `_extract_all`).
* `overview` / `find_symbol` / `view_symbol` / `find_refs` are the per-symbol read
  tools, now multi-language (they replace the Python-only jedi backend in
  `symbols.py`; jedi remains the editor for `replace_symbol`/`insert_symbol`).

API mirrors `symbols.SymbolService` so `tools.py` can route to either backend.
"""

import hashlib
import logging
import os
import pickle
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict, namedtuple

import rustworkx as rx
import tree_sitter_language_pack as tlp
from tree_sitter import Parser, Query, QueryCursor

from . import config
from .ignore import IGNORE_DIRS, REPOMAP_EXTRA

log = logging.getLogger("chad")

# repomap indexes the whole repo, so it skips the base set PLUS model weights, installed
# packages, and caches (REPOMAP_EXTRA) — a symbol editor doesn't need those exclusions.
_SKIP_NAMES = frozenset(IGNORE_DIRS + REPOMAP_EXTRA)

_MAX_FILE_BYTES = 1_000_000   # skip anything bigger than ~1MB (generated/minified)
_MAX_FILES = 20000
_CHARS_PER_TOK = 4            # rough token estimate without coupling to the tokenizer

# Total wall-clock allowed for the decorative "used by …" annotations in one
# disambiguation listing; candidates past the deadline render un-annotated.
_DISAMBIG_BUDGET_S = 3.0

# An identifier defined in more files than this (__init__, get, main, forward, …) says
# nothing about which file matters, so it's excluded from the rank graph. Without the
# cutoff a generic name fans out definers × referencers: on a 11k-file repo `__init__`
# alone produced 11M edges of a 32M-edge/10GB graph — enough to stall the tool for
# minutes and push a machine already holding model weights into Metal OOM.
_MAX_DEFINERS = 16

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
_CACHE_VERSION = 1          # bump when the entry shape or tags queries change
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

# A definition discovered by tree-sitter. `kind` is the tag suffix (function,
# class, method, constant, ...); `sig` is the collapsed header line(s). `name_row`
# and `name_col` are the 0-based position of the identifier itself (what an LSP
# wants for go-to-def / find-references).
Tag = namedtuple("Tag", "rel path name kind line end_line sig name_row name_col")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOK)


def _build_edges(defines, references):
    """The rank graph's edge weights: {(referencer rel, definer rel): weight}, ONE
    aggregated edge per file pair. The naive form (an edge object per raw reference)
    is definers × references per ident — measured 32M edges / 10 GB on an 11k-file
    repo, which OOMs the machine out from under the model. sqrt damps mega-callers
    so one hub file doesn't drown the ranking, and idents defined in more than
    _MAX_DEFINERS files are excluded (no ranking signal, all of the blowup)."""
    edge_w = defaultdict(float)
    for ident, definers in defines.items():
        refcounts = references.get(ident)
        if not refcounts or len(definers) > _MAX_DEFINERS:
            continue
        for referencer, cnt in refcounts.items():
            w = cnt ** 0.5
            for definer in definers:
                if referencer != definer:
                    edge_w[(referencer, definer)] += w
    return edge_w


def _rank_files(rels, edge_w, seeds):
    """Personalized PageRank over the file graph via rustworkx — a native, maintained
    implementation in place of the hand-rolled power iteration it replaced (validated
    on a real 421k-edge graph: identical top-50, spearman 0.999, 0.79s -> 0.04s)."""
    g = rx.PyDiGraph()
    idx = dict(zip(rels, g.add_nodes_from(list(rels))))
    g.add_edges_from([(idx[u], idx[v], w) for (u, v), w in edge_w.items()])
    pers = {idx[r]: w for r, w in seeds.items()} if seeds else None
    ranks = rx.pagerank(g, alpha=0.85, weight_fn=float, personalization=pers,
                        tol=1.0e-6, max_iter=100)
    return {rel: ranks[i] for rel, i in idx.items()}


class RepoMap:
    """Tree-sitter symbol intelligence rooted at a project directory."""

    def __init__(self, root: str = "."):
        self.root = os.path.abspath(root)
        self._tooling = {}   # lang -> (Parser, Query) | None
        self._cache = {}     # path -> (mtime, [defs], [(refname, rel, line)])
        self._files = None   # memoized completed _code_files() result; None = uncomputed
        self._disk_checked = False  # the on-disk tags cache is loaded at most once
        self._agg = None     # incremental rank-graph tables; see _aggregates()
        self._refsum_cache = {}  # (path, row, col, mtime) -> disambig note

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
                qsrc = tlp.get_tags_query(lang)
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
            for _pat, caps in matches:
                name_nodes = caps.get("name")
                name = (src[name_nodes[0].start_byte:name_nodes[0].end_byte]
                        .decode("utf-8", "replace")) if name_nodes else None
                nrow = name_nodes[0].start_point[0] if name_nodes else 0
                ncol = name_nodes[0].start_point[1] if name_nodes else 0
                for cap, nodes in caps.items():
                    if cap.startswith("definition") and name:
                        kind = cap.split(".", 1)[1] if "." in cap else "def"
                        for dn in nodes:
                            defs.append(Tag(rel, path, name, kind,
                                            dn.start_point[0] + 1, dn.end_point[0] + 1,
                                            self._header(src, dn), nrow, ncol))
                    elif cap.startswith("reference") and name:
                        for rn in nodes:
                            refs.append((name, rel, rn.start_point[0] + 1))
        self._cache[path] = (mtime, defs, refs)
        return defs, refs

    # -- repo map (ranked skeleton) --------------------------------------

    def _aggregates(self, files, should_stop=None):
        """(file_defs, defines, references) for the rank graph, maintained
        incrementally: only files whose cached tags changed since the last call are
        re-folded. Rebuilding from scratch was the warm-path majority (measured
        0.39s per repo_map over 1.2M raw references on an 11k-file repo) and it
        re-ran on every call even when nothing had changed. Interruption is safe:
        contributions fold per file, so a stopped pass just leaves the remaining
        files for the next call."""
        if self._agg is None:
            # contrib: path -> (mtime folded in, [def names], {name: ref count})
            self._agg = ({}, {}, defaultdict(set), defaultdict(Counter))
        contrib, file_defs, defines, references = self._agg
        for f in files:
            if should_stop and should_stop():
                break
            defs, refs = self._extract(f)
            mtime = self._cache.get(f, (None,))[0]
            old = contrib.get(f)
            if old and old[0] == mtime:
                continue
            rel = self._rel(f)
            if old:  # subtract the file's previous contribution before re-folding
                for name in old[1]:
                    s = defines.get(name)
                    if s:
                        s.discard(rel)
                        if not s:
                            del defines[name]
                for name in old[2]:
                    c = references.get(name)
                    if c:
                        c.pop(rel, None)
                        if not c:
                            del references[name]
            def_names = []
            for d in defs:
                if d.name:
                    defines[d.name].add(rel)
                    def_names.append(d.name)
            ref_counts = Counter()
            for name, _rrel, _ln in refs:  # every ref in `refs` is from this file
                ref_counts[name] += 1
            for name, cnt in ref_counts.items():
                references[name][rel] += cnt
            file_defs[rel] = defs
            contrib[f] = (mtime, def_names, ref_counts)
        return file_defs, defines, references

    def repo_map(self, budget_tokens: int = 1500, focus=None, should_stop=None) -> str:
        """A PageRank-ranked, signature-only map of the codebase within a token
        budget. `focus` (list of path substrings) personalizes the ranking toward
        the files the agent is working on, the way aider biases toward chat files."""
        files = self._code_files(should_stop)
        if not files:
            return "[no source files found]"
        self._extract_all(files, should_stop)
        file_defs, defines, references = self._aggregates(files, should_stop)
        if should_stop and should_stop():
            return "[interrupted]"

        # Personalize ranking toward the files the agent is working on (focus) and, by
        # default, likely entrypoints (check.py / main / conftest / app). Tasks usually
        # live in test-reachable code, so a seed on the entrypoint flows rank along its
        # imports to the module under test — surfacing it even in a big, noisy repo.
        seeds = {}
        for rel in file_defs:
            base = os.path.basename(rel).lower()
            if base.startswith(("check", "main", "conftest", "__main__")) or base == "app.py":
                seeds[rel] = 1.0
        if focus:
            for rel in file_defs:
                if any(fp in rel for fp in focus):
                    seeds[rel] = 1.0
        try:
            ranks = _rank_files(file_defs, _build_edges(defines, references), seeds)
        except Exception:  # noqa: BLE001 - rank failure degrades to def-count order
            ranks = {}
        # Files with no edges still deserve a spot; rank them by definition count.
        ordered = sorted(file_defs,
                         key=lambda r: (ranks.get(r, 0.0), len(file_defs[r])),
                         reverse=True)

        out, used = [], 0
        header = (f"repo map ({len(files)} files, ranked by reference centrality; "
                  f"signatures only — use view_symbol/read to see bodies)\n")
        used += _estimate_tokens(header)
        for rel in ordered:
            defs = file_defs[rel]
            if not defs:
                continue
            block = [f"\n{rel}"]
            for d in sorted(defs, key=lambda t: t.line):
                indent = "  " if d.kind in ("method",) else "  "
                block.append(f"{indent}{d.line}: {d.sig}")
            chunk = "\n".join(block)
            cost = _estimate_tokens(chunk)
            if used + cost > budget_tokens and out:
                # Name the most-central omitted files (rank-ordered) so the model can
                # find_symbol/overview into them — but only a handful: dumping every
                # path on a big repo is itself a prefill cost we measured biting back.
                remaining = [r for r in ordered[ordered.index(rel):] if file_defs[r]]
                listed = ", ".join(remaining[:12])
                tail = f" (+{len(remaining) - 12} more)" if len(remaining) > 12 else ""
                out.append(f"\n[{len(remaining)} more files omitted to fit budget="
                           f"{budget_tokens}; raise budget= or find_symbol/overview into "
                           f"one. Top: {listed}{tail}]")
                break
            out.append(chunk)
            used += cost
        return header + "".join(out)

    # -- per-symbol reads (multi-language) -------------------------------

    def _find_defs(self, name, path=None, should_stop=None):
        target = name.replace(".", "/").split("/")[-1]
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
                if d.name == target:
                    hits.append(d)
        return hits

    def _refsum_key(self, h):
        return (h.path, h.name_row, h.name_col, self._cache.get(h.path, (None,))[0])

    def _ref_summary(self, h, timeout=None):
        """A short 'used by …' hint for one definition, so the model can tell same-named
        candidates apart by who calls them (e.g. the validate on the signup path vs the
        widely-used schema validator). Precise via the language server; silent if it's
        unavailable (we don't want to imply precision we don't have). Decorative, so it
        goes through `references_decorative` (never starts a non-Python server) and is
        cached per (definition site, file mtime) — a repeated view_symbol on the same
        ambiguous name used to re-pay every LSP lookup (measured 79s on a repeat)."""
        key = self._refsum_key(h)
        cached = self._refsum_cache.get(key)
        if cached is not None:
            return cached
        try:
            from . import lsp
            locs = lsp.service().references_decorative(h.rel, h.name_row, h.name_col,
                                                       timeout=timeout)
        except Exception:
            locs = None
        if not locs:
            note = ""
        else:
            files = []
            for rel, _ln, _col in locs:
                if rel != h.rel and rel not in files:
                    files.append(rel)
            if not files:
                note = "  — no external refs"
            else:
                more = f" (+{len(files) - 1} more)" if len(files) > 1 else ""
                note = f"  — used by {files[0]}{more}"
        if len(self._refsum_cache) > 512:  # bound a very long session's cache
            self._refsum_cache.clear()
        self._refsum_cache[key] = note
        return note

    def _disambig(self, name, hits):
        # Annotate the first few candidates with who references them, under a shared
        # wall-clock budget: per-candidate LSP lookups are cheap warm, but a cold or
        # slow server must not turn a disambiguation list into a minutes-long stall
        # (measured 86s on an 11k-file mixed-language repo before the budget). Each
        # lookup gets only the budget's remaining time, and candidates the budget
        # never reached are cached blank for this mtime generation — otherwise a
        # repeated lookup re-pays the slow pass one candidate at a time.
        rows = []
        deadline = time.monotonic() + _DISAMBIG_BUDGET_S
        for i, h in enumerate(hits[:50]):
            note = ""
            if i < 8:
                remaining = deadline - time.monotonic()
                if remaining > 0.25:
                    note = self._ref_summary(h, timeout=remaining)
                else:
                    self._refsum_cache.setdefault(self._refsum_key(h), "")
            rows.append(f"  {h.rel}:{h.line}  {h.kind} {h.sig}{note}")
        return (f"[{len(hits)} symbols named '{name}'; pass path= to disambiguate "
                f"(pick by who uses each):]\n" + "\n".join(rows))

    def overview(self, path: str, should_stop=None) -> str:
        if not os.path.isfile(path):
            return f"[no such file: {path}]"
        defs, _ = self._extract(path)
        if not defs:
            if not self.lang_for(path):
                return "[no tree-sitter grammar for this file type; use read]"
            return "[no functions or classes]"
        lines = []
        for d in sorted(defs, key=lambda t: t.line):
            indent = "  " if d.kind == "method" else ""
            lines.append(f"{d.line}: {indent}{d.sig}")
        return "\n".join(lines)

    def find_symbol(self, name: str, should_stop=None) -> str:
        hits = self._find_defs(name, None, should_stop)
        if not hits:
            return f"[no definition found for '{name}']"
        out = [f"{h.rel}:{h.line}  {h.kind}  {h.sig}" for h in hits]
        return "\n".join(out[:100])

    def view_symbol(self, name: str, path=None, should_stop=None) -> str:
        hits = self._find_defs(name, path, should_stop)
        if not hits:
            return f"[symbol not found: {name}]"
        if len(hits) > 1 and path is None:
            return self._disambig(name, hits)
        h = hits[0]
        try:
            with open(h.path, errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            return f"[could not read {h.rel}]"
        a, b = h.line, h.end_line
        width = len(str(b))
        body = "\n".join(f"{i:>{width}}  {lines[i-1]}"
                         for i in range(a, b + 1) if 0 < i <= len(lines))
        return f"{h.rel}:{a}-{b}\n{body}"

    def find_refs(self, name: str, path=None, should_stop=None) -> str:
        """Every USE of a symbol across the project. Precise when a language server
        is available (follows imports, respects scope, won't confuse same-named
        symbols); falls back to tree-sitter name-matching otherwise."""
        # Locate the definition first — gives the exact identifier position the LSP
        # needs, and reuses disambiguation when several symbols share a name.
        hits = self._find_defs(name, path, should_stop)
        if hits and not (len(hits) > 1 and path is None):
            h = hits[0]
            from . import lsp  # lazy: only pay the solidlsp import when refs are requested
            locs = lsp.service().references(h.rel, h.name_row, h.name_col)
            if locs is not None:
                if not locs:
                    return "[no references]"
                return self._format_refs([(rel, ln) for rel, ln, _col in locs],
                                         tag="precise")
        elif len(hits) > 1 and path is None:
            return self._disambig(name, hits)
        # fallback: tree-sitter name match across the project
        return self._find_refs_treesitter(name, should_stop)

    def _find_refs_treesitter(self, name: str, should_stop=None) -> str:
        target = name.replace(".", "/").split("/")[-1]
        out = []
        self._extract_all(self._code_files(should_stop), should_stop)
        for f in self._code_files(should_stop):
            if should_stop and should_stop():
                return "[interrupted by user]"
            _defs, refs = self._extract(f)
            for rname, rel, ln in refs:
                if rname == target:
                    out.append((rel, ln))
        if not out:
            return "[no references]"
        return self._format_refs(out, tag="name-match")

    def _format_refs(self, pairs, tag="") -> str:
        by_file = defaultdict(list)
        for rel, ln in pairs:
            by_file[rel].append(ln)
        lines_out = []
        for rel in sorted(by_file):
            try:
                with open(os.path.join(self.root, rel), errors="replace") as f:
                    src_lines = f.read().splitlines()
            except OSError:
                src_lines = []
            for ln in sorted(set(by_file[rel])):
                code = src_lines[ln - 1].strip() if 0 < ln <= len(src_lines) else ""
                lines_out.append(f"{rel}:{ln}: {code}")
        n = len(lines_out)
        if tag == "name-match":
            # No language server: these are bare name matches and MAY include unrelated
            # same-named symbols. Make the degraded precision visible, not silent.
            head = (f"[{n} reference{'s' if n != 1 else ''} — NAME-MATCH ONLY (no language "
                    f"server); may include unrelated same-named symbols, verify before "
                    f"editing]\n")
        else:
            head = f"[{n} reference{'s' if n != 1 else ''}{', ' + tag if tag else ''}]\n"
        return head + "\n".join(lines_out[:200])


    # -- precise rename (refs-driven, position-safe) ---------------------

    def rename_symbol(self, name: str, new_name: str, path=None, should_stop=None) -> str:
        """Rename a symbol and every precise reference to it in one shot. Sites come
        from the language server's find-all-references (follows imports, respects scope),
        so an unrelated same-named method/function is never touched — and we rewrite
        ONLY the exact identifier token at each (line, col), never a blind text replace.
        Refuses (returns the disambiguation list) when the name is ambiguous, and refuses
        rather than guess when no precise language server is available."""
        hits = self._find_defs(name, path, should_stop)
        if not hits:
            return f"[no definition found for '{name}']"
        if len(hits) > 1 and path is None:
            return (f"[ambiguous: {len(hits)} symbols named '{name}'. Pass path= to choose "
                    f"one before renaming:]\n{self._disambig(name, hits)}")
        h = hits[0]
        target = h.name
        from . import lsp  # lazy: only pay the solidlsp import when a rename is requested
        locs = lsp.service().references(h.rel, h.name_row, h.name_col)
        if locs is None:
            return ("[rename needs the precise language server (find-all-references), which "
                    "is unavailable for this file type here. Use find_refs to locate the "
                    "sites and edit each, or rename with edit.]")
        # Always include the definition's own identifier; dedupe across refs.
        sites = {(h.rel, h.name_row + 1, h.name_col + 1)}
        for rel, ln, col in locs:
            sites.add((rel, ln, col))

        by_file = defaultdict(list)
        for rel, ln, col in sites:
            by_file[rel].append((ln, col))

        changed, total, skipped = [], 0, 0
        for rel in sorted(by_file):
            p = os.path.join(self.root, rel)
            try:
                with open(p, errors="replace") as f:
                    lines = f.read().splitlines(keepends=True)
            except OSError:
                continue
            n_here = 0
            # right-to-left within the file so earlier offsets stay valid after splicing
            for ln, col in sorted(by_file[rel], reverse=True):
                if not (0 < ln <= len(lines)):
                    continue
                c = col - 1
                line = lines[ln - 1]
                if line[c:c + len(target)] != target:
                    skipped += 1  # position didn't land on the identifier — skip safely
                    continue
                lines[ln - 1] = line[:c] + new_name + line[c + len(target):]
                n_here += 1
            if n_here:
                try:
                    with open(p, "w") as f:
                        f.write("".join(lines))
                except OSError:
                    continue
                self._cache.pop(p, None)   # force re-extract on next read
                changed.append((rel, n_here))
                total += n_here

        if not total:
            return f"[no occurrences of '{target}' rewritten]"
        detail = ", ".join(f"{rel} ×{n}" for rel, n in changed)
        note = f" ({skipped} stale position(s) skipped)" if skipped else ""
        return (f"[renamed '{target}' → '{new_name}' at {total} site(s) across "
                f"{len(changed)} file(s){note}: {detail}]")


_SERVICE = None


def service() -> RepoMap:
    """Lazily-bound, cwd-rooted repo map (the project the agent is operating in)."""
    global _SERVICE
    if _SERVICE is None or _SERVICE.root != os.path.abspath(os.getcwd()):
        _SERVICE = RepoMap(os.getcwd())
    return _SERVICE
