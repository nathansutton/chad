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
* `repo_map()` ranks definitions with PageRank over the file→symbol reference graph
  and renders the most central ones as elided signatures within a token budget.
* `overview` / `find_symbol` / `view_symbol` / `find_refs` are the per-symbol read
  tools, now multi-language (they replace the Python-only jedi backend in
  `symbols.py`; jedi remains the editor for `replace_symbol`/`insert_symbol`).

API mirrors `symbols.SymbolService` so `tools.py` can route to either backend.
"""

import glob as _glob
import os
from collections import defaultdict, namedtuple

import tree_sitter_language_pack as tlp
from tree_sitter import Parser, Query, QueryCursor

from .ignore import IGNORE_DIRS, REPOMAP_EXTRA, slash_wrapped

# repomap indexes the whole repo, so it skips the base set PLUS model weights, installed
# packages, and caches (REPOMAP_EXTRA) — a symbol editor doesn't need those exclusions.
_SKIP = slash_wrapped(IGNORE_DIRS + REPOMAP_EXTRA)

_MAX_FILE_BYTES = 1_000_000   # skip anything bigger than ~1MB (generated/minified)
_MAX_FILES = 20000
_CHARS_PER_TOK = 4            # rough token estimate without coupling to the tokenizer

# A definition discovered by tree-sitter. `kind` is the tag suffix (function,
# class, method, constant, ...); `sig` is the collapsed header line(s). `name_row`
# and `name_col` are the 0-based position of the identifier itself (what an LSP
# wants for go-to-def / find-references).
Tag = namedtuple("Tag", "rel path name kind line end_line sig name_row name_col")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOK)


class _Graph:
    """Minimal weighted multigraph: just what _pagerank reads (nodes + weighted
    edges), kept in insertion order so iteration matches the construction loop."""
    def __init__(self):
        self._nodes = []
        self._edges = []  # (u, v, {"weight": w})
    def add_nodes_from(self, ns):
        self._nodes.extend(ns)
    def add_edge(self, u, v, weight=1.0):
        self._edges.append((u, v, {"weight": weight}))
    def nodes(self):
        return self._nodes
    def edges(self, data=False):
        return self._edges if data else [(u, v) for u, v, _ in self._edges]


def _pagerank(g, personalization=None, damping=0.85, iters=50, tol=1.0e-6):
    """Personalized PageRank over a weighted MultiDiGraph, pure-Python power iteration.

    A library pagerank would pull in scipy, which isn't installed here — so the prior
    library call always raised and the repo map silently fell back to ranking by
    definition count. This self-contained version actually ranks by reference centrality
    (and honors the personalization seeds), with no extra dependency."""
    nodes = list(g.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    out = defaultdict(float)
    adj = defaultdict(list)
    for u, v, data in g.edges(data=True):
        w = data.get("weight", 1.0)
        out[u] += w
        adj[u].append((v, w))
    if personalization:
        s = float(sum(personalization.get(u, 0.0) for u in nodes)) or 1.0
        p = {u: personalization.get(u, 0.0) / s for u in nodes}
    else:
        p = {u: 1.0 / n for u in nodes}
    rank = {u: 1.0 / n for u in nodes}
    dangling = [u for u in nodes if out[u] == 0.0]
    for _ in range(iters):
        nxt = {u: (1.0 - damping) * p[u] for u in nodes}
        dmass = damping * sum(rank[u] for u in dangling)
        for u in nodes:
            if out[u] == 0.0:
                continue
            share = damping * rank[u] / out[u]
            for v, w in adj[u]:
                nxt[v] += share * w
        if dmass:
            for u in nodes:
                nxt[u] += dmass * p[u]
        err = sum(abs(nxt[u] - rank[u]) for u in nodes)
        rank = nxt
        if err < tol:
            break
    return rank


class RepoMap:
    """Tree-sitter symbol intelligence rooted at a project directory."""

    def __init__(self, root: str = "."):
        self.root = os.path.abspath(root)
        self._tooling = {}   # lang -> (Parser, Query) | None
        self._cache = {}     # path -> (mtime, [defs], [(refname, rel, line)])
        self._files = None   # memoized completed _code_files() result; None = uncomputed

    # -- tree-sitter plumbing --------------------------------------------

    def _lang_for(self, path):
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
        # list instead of re-globbing the repo mid-turn. A scan interrupted by should_stop
        # is returned but NOT cached — caching a partial walk would hide real files.
        if self._files is not None:
            return self._files
        files = _glob.glob(os.path.join(self.root, "**", "*"), recursive=True)
        out = []
        interrupted = False
        for f in files:
            if should_stop and should_stop():
                interrupted = True
                break
            p = "/" + f.replace(os.sep, "/")
            if any(d in p for d in _SKIP):
                continue
            try:
                if not os.path.isfile(f) or os.path.getsize(f) > _MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            if self._lang_for(f):
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
            return [], []
        cached = self._cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1], cached[2]

        defs, refs = [], []
        tools = self._lang_tools(self._lang_for(path))
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

    def repo_map(self, budget_tokens: int = 1500, focus=None, should_stop=None) -> str:
        """A PageRank-ranked, signature-only map of the codebase within a token
        budget. `focus` (list of path substrings) personalizes the ranking toward
        the files the agent is working on, the way aider biases toward chat files."""
        files = self._code_files(should_stop)
        if not files:
            return "[no source files found]"
        file_defs = {}
        defines = defaultdict(set)      # name -> {rel that define it}
        references = defaultdict(list)  # name -> [rel that reference it]
        for f in files:
            if should_stop and should_stop():
                return "[interrupted]"
            defs, refs = self._extract(f)
            rel = self._rel(f)
            file_defs[rel] = defs
            for d in defs:
                if d.name:
                    defines[d.name].add(rel)
            for name, rrel, _ln in refs:
                references[name].append(rrel)

        # Graph: edge referencer -> definer for every cross-file symbol use.
        g = _Graph()
        g.add_nodes_from(file_defs)
        for ident, definers in defines.items():
            for referencer in references.get(ident, ()):
                for definer in definers:
                    if referencer != definer:
                        g.add_edge(referencer, definer, weight=1.0)

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
        personalization = seeds or None
        try:
            ranks = _pagerank(g, personalization=personalization)
        except Exception:
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
        hits = []
        for f in files:
            if should_stop and should_stop():
                break
            defs, _ = self._extract(f)
            for d in defs:
                if d.name == target:
                    hits.append(d)
        return hits

    def _ref_summary(self, h):
        """A short 'used by …' hint for one definition, so the model can tell same-named
        candidates apart by who calls them (e.g. the validate on the signup path vs the
        widely-used schema validator). Precise via the language server; silent if it's
        unavailable (we don't want to imply precision we don't have)."""
        try:
            from . import lsp
            locs = lsp.service().references(h.rel, h.name_row, h.name_col)
        except Exception:
            locs = None
        if not locs:
            return ""
        files = []
        for rel, _ln, _col in locs:
            if rel != h.rel and rel not in files:
                files.append(rel)
        if not files:
            return "  — no external refs"
        more = f" (+{len(files) - 1} more)" if len(files) > 1 else ""
        return f"  — used by {files[0]}{more}"

    def _disambig(self, name, hits):
        # Annotate the first few candidates with who references them (bounded: the
        # per-candidate LSP lookups are cheap for a handful but we don't want to fire
        # dozens on a very common name).
        rows = []
        for i, h in enumerate(hits[:50]):
            note = self._ref_summary(h) if i < 8 else ""
            rows.append(f"  {h.rel}:{h.line}  {h.kind} {h.sig}{note}")
        return (f"[{len(hits)} symbols named '{name}'; pass path= to disambiguate "
                f"(pick by who uses each):]\n" + "\n".join(rows))

    def overview(self, path: str, should_stop=None) -> str:
        if not os.path.isfile(path):
            return f"[no such file: {path}]"
        defs, _ = self._extract(path)
        if not defs:
            if not self._lang_for(path):
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
