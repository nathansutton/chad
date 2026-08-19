"""Ranked lexical repository retrieval — the engine behind the model-facing `search` tool.

`bash`/`rg` answers "find this exact string". It cannot answer "where is FHIR
validation handled?" without the model first guessing several synonymous regexes,
and every guess that misses costs a round trip. This module gives the model the
other primitive: a persistent BM25 index over the repository's text, queried in
plain English, returning ranked `path:line` locations with a few lines of context.

Deliberate shape:

* **one document per file**, not per line — update/delete semantics stay trivial
  (Tantivy documents are immutable; an update is delete-by-term + re-add + commit)
  and the document count stays in the thousands rather than the millions;
* **content is indexed but not stored** — snippets and line numbers come from
  re-reading only the handful of files that actually rank, so the index stays
  small and a snippet can never disagree with the file on disk;
* **the index lives under ~/.chad/cache/search/**, never in the repository, and is
  a pure derived cache: deleting it costs one rebuild and nothing else;
* **freshness is mtime+size reconciliation**, run lazily on the first search of a
  session — no watcher, no daemon, no cost at chad startup.

`tantivy` is a native (maturin/pyo3) binding, so it is import-guarded for the same
reason `repomap` guards tree-sitter: on a platform with no matching wheel the rest
of the tool surface must keep working. `available()` is false there and the tool
says so instead of raising.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import threading
from dataclasses import dataclass

try:
    import tantivy
except ImportError:  # pragma: no cover — exercised only on wheel-less platforms
    tantivy = None   # type: ignore[assignment]

from . import config
from .ignore import IGNORE_DIRS, REPOMAP_EXTRA
from .repomap import _MAX_FILE_BYTES, lang_for

log = logging.getLogger("chad")

# Same exclusion policy as the other whole-repo walk (repomap): the base set plus
# model weights, installed packages and caches. Single-sourced from ignore.py so a
# new entry is still added in exactly one place.
_SKIP_NAMES = frozenset(IGNORE_DIRS + REPOMAP_EXTRA)

_MAX_FILES = 20000          # same ceiling repomap uses for a whole-repo walk
_STATE_VERSION = 1          # bump when the schema or state shape changes
_CACHE_DIR = os.path.expanduser("~/.chad/cache/search")
_WRITER_HEAP = 32_000_000   # 32MB is ample for file-sized documents; the 128MB
                            # default is real resident memory next to a loaded model

# Result shaping. A search result competes with `rg` output for the same context
# budget, so it must stay small enough that one search beats two greps and a cat.
DEFAULT_LIMIT = 8
MAX_LIMIT = 20
_CONTEXT_LINES = 1          # lines of context either side of the best-matching line
_MAX_LINE_CHARS = 200       # per snippet line, so one minified line can't eat the result

# Question scaffolding carries no retrieval signal but does dilute BM25 scoring
# (every doc matching "where" is noise). Stripped from the query, never from the index.
_STOPWORDS = frozenset("""
a an the and or of in on to for is are was were be been being do does did done
where what which who whom how why when is-it it its this that these those
i we you they there here from by with as at into out up down about
find show tell me my our your please can could should would
""".split())

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")
# Cap the camel-split vocabulary per file: the point is recall on identifier words,
# not a second copy of the file in the index.
_MAX_CAMEL_TOKENS = 2000


@dataclass(frozen=True)
class Hit:
    """One ranked location: where to look, and just enough text to judge it."""
    path: str        # repo-relative
    line: int        # 1-based line the snippet is centered on
    score: float
    lines: list      # [(lineno, text)] — already clipped, ready to render


def available() -> bool:
    """Whether ranked search can run at all (the native binding imported)."""
    return tantivy is not None


# -- query and text normalization ------------------------------------------------

def _camel_split(word: str) -> list:
    return [p for p in _CAMEL_BOUNDARY.split(word) if p]


def query_terms(query: str) -> list:
    """A natural-language question reduced to the words worth matching.

    Tantivy's query language is never exposed to the model — the query is torn down
    to bare alphanumeric words here, so no quoting, `AND`/`OR`, field prefix or stray
    `:` from a pasted `path:line` can change the search's meaning or make it error.
    """
    out, seen = [], set()
    for word in re.findall(r"[A-Za-z0-9_]+", query):
        for part in [word, *_camel_split(word)]:
            t = part.lower()
            if len(t) < 2 or t in _STOPWORDS or t in seen:
                continue
            seen.add(t)
            out.append(t)
    return out


def _camel_tokens(text: str) -> list:
    """Lowercased word-parts of the camelCase identifiers in `text`.

    `parseAuthHeader` is one token to any word tokenizer, so a search for
    "auth header" would miss it entirely in a camelCase codebase. Indexing the
    split parts alongside the raw content fixes that for the cost of a few hundred
    extra terms per file. snake_case needs no help — the default tokenizer already
    splits on `_`.
    """
    out: set = set()
    for word in _WORD.findall(text):
        if not _CAMEL_BOUNDARY.search(word):
            continue
        for part in _camel_split(word):
            if len(part) > 2:
                out.add(part.lower())
                if len(out) >= _MAX_CAMEL_TOKENS:
                    return sorted(out)
    return sorted(out)


def _line_probe(term: str) -> str:
    """The prefix of `term` used to find the best line inside a ranked file.

    The index stems ("validation" and "validate" share a term), but the snippet
    picker works on raw text, so it matches on a truncated prefix to stay in step:
    "validation" -> "validat", "errors" -> "erro". Floored at 4 characters so a
    short word can't match half the file.
    """
    return term[:max(4, len(term) - 3)]


# -- the index ------------------------------------------------------------------

class CodeSearchIndex:
    """A persistent BM25 index over one repository's text files.

    Lifecycle is entirely lazy: nothing is opened until the first `search`, and
    reconciliation reads only files whose mtime/size changed since the last one.
    Every failure path degrades — a corrupt index is rebuilt, a locked writer
    searches the previous commit, an unreadable file is skipped — because a search
    tool that raises costs the agent a turn it can never get back.
    """

    _instances: dict = {}
    _instances_lock = threading.Lock()

    def __init__(self, root: str = "."):
        self.root = os.path.abspath(root)
        self._index = None
        self._lock = threading.Lock()

    @classmethod
    def for_root(cls, root: str = "."):
        """The process-wide instance for `root`, so a session opens the index once."""
        key = os.path.abspath(root)
        with cls._instances_lock:
            inst = cls._instances.get(key)
            if inst is None:
                inst = cls._instances[key] = cls(key)
            return inst

    @classmethod
    def reset(cls) -> None:
        """Drop cached instances (tests, and any caller that moves the cache root)."""
        with cls._instances_lock:
            cls._instances.clear()

    # -- locations --------------------------------------------------------------

    def cache_dir(self) -> str:
        """Per-repository cache directory, keyed by canonical path. Outside the repo
        on purpose: the index is derived state and must never appear in a diff."""
        base = config.env_str("CHAD_SEARCH_DIR", _CACHE_DIR)
        return os.path.join(base, hashlib.sha256(self.root.encode()).hexdigest()[:16])

    def _index_dir(self) -> str:
        return os.path.join(self.cache_dir(), "index")

    def _state_file(self) -> str:
        return os.path.join(self.cache_dir(), "state.json")

    # -- schema -----------------------------------------------------------------

    @staticmethod
    def _schema():
        sb = tantivy.SchemaBuilder()
        # `path` is raw (one un-tokenized term) because it is the document's identity:
        # delete-by-term on update needs an exact match, and it is the only stored field.
        sb.add_text_field("path", stored=True, tokenizer_name="raw")
        # ...and tokenized separately for ranking, so "authentication middleware" can
        # score auth/middleware.py on its path without the path polluting content stats.
        sb.add_text_field("pathtext", stored=False, tokenizer_name="default")
        sb.add_text_field("filename", stored=False, tokenizer_name="default")
        sb.add_text_field("lang", stored=False, tokenizer_name="default")
        # en_stem so "validation"/"validate"/"validating" collide — the single biggest
        # recall win for questions phrased in English rather than in identifiers.
        sb.add_text_field("content", stored=False, tokenizer_name="en_stem")
        return sb.build()

    _FIELDS = ("content", "pathtext", "filename", "lang")
    # A filename hit is a much stronger signal than one term in a 500-line file, and
    # BM25 alone under-weights it; the boosts encode that ordering explicitly.
    _BOOSTS = {"filename": 3.0, "pathtext": 2.0, "lang": 0.5, "content": 1.0}

    def _open(self):
        """Open (or create) the index, rebuilding from scratch if it is unusable."""
        if self._index is not None:
            return self._index
        d = self._index_dir()
        os.makedirs(d, mode=0o700, exist_ok=True)
        try:
            self._index = tantivy.Index(self._schema(), path=d)
        except Exception as e:  # noqa: BLE001 — corrupt/foreign/older index: rebuild it
            log.warning("search index at %s unusable (%s); rebuilding", d, e)
            self._wipe()
            os.makedirs(d, mode=0o700, exist_ok=True)
            self._index = tantivy.Index(self._schema(), path=d)
        return self._index

    def _wipe(self) -> None:
        """Throw the whole cache away, state sidecar included, so the next reconcile is
        a full rebuild. Touches nothing outside the cache directory."""
        shutil.rmtree(self.cache_dir(), ignore_errors=True)

    # -- state sidecar ----------------------------------------------------------
    # relpath -> [mtime_ns, size]. A sidecar rather than indexed fields: it is read
    # in full on every reconcile and written once, which is exactly what a JSON blob
    # is good at, and keeping it out of the index leaves document deletion trivial.

    def _load_state(self) -> dict:
        try:
            with open(self._state_file()) as f:
                data = json.load(f)
            if data.get("v") == _STATE_VERSION and data.get("root") == self.root:
                return data["files"]
        except Exception:  # noqa: BLE001 — absent/corrupt/foreign state: full reindex
            pass
        return {}

    def _save_state(self, files: dict) -> None:
        try:
            os.makedirs(self.cache_dir(), mode=0o700, exist_ok=True)
            blob = json.dumps({"v": _STATE_VERSION, "root": self.root, "files": files})
            tmp = self._state_file() + ".tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(blob)
            os.replace(tmp, self._state_file())
        except Exception:  # noqa: BLE001 — state is an optimization, never a failure
            pass

    # -- discovery --------------------------------------------------------------

    def _discover(self) -> dict:
        """relpath -> (mtime_ns, size) for every candidate file, cheaply.

        Pruning happens in `os.walk`'s dirnames so an ignored tree is never entered,
        and nothing is opened here — only `stat` — so the walk stays proportional to
        the repository's shape rather than its bytes.
        """
        out = {}
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(".") and d not in _SKIP_NAMES]
            for name in filenames:
                if name.startswith("."):
                    continue
                f = os.path.join(dirpath, name)
                try:
                    st = os.stat(f)
                    if not os.path.isfile(f) or st.st_size > _MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                out[os.path.relpath(f, self.root)] = (st.st_mtime_ns, st.st_size)
                if len(out) >= _MAX_FILES:
                    return out
        return out

    def _read_text(self, rel: str):
        """The file's text, or None if it is binary/unreadable. Strict UTF-8 decoding
        is the binary test: cheaper and more honest than sniffing byte histograms."""
        try:
            with open(os.path.join(self.root, rel), "rb") as f:
                raw = f.read(_MAX_FILE_BYTES)
        except OSError:
            return None
        if b"\x00" in raw:
            return None
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return None

    # -- reconciliation ---------------------------------------------------------

    def reconcile(self) -> tuple:
        """Bring the index in step with the working tree. Returns (indexed, removed).

        Unchanged files are never opened — the mtime/size comparison happens against
        the sidecar, so a warm reconcile on an untouched repository does one walk and
        no reads at all.
        """
        index = self._open()
        current = self._discover()
        state = self._load_state()
        changed = [p for p, meta in current.items() if state.get(p) != list(meta)]
        removed = [p for p in state if p not in current]
        if not changed and not removed:
            return 0, 0

        try:
            writer = index.writer(heap_size=_WRITER_HEAP, num_threads=1)
        except Exception as e:  # noqa: BLE001 — another process holds the writer lock
            log.warning("search index busy (%s); searching the last commit", e)
            return 0, 0

        indexed = 0
        for rel in changed + removed:
            writer.delete_documents("path", rel)
        for rel in changed:
            text = self._read_text(rel)
            if text is None:
                # Binary/unreadable, but keep it in the state so it is stat-compared
                # rather than re-read on every future reconcile.
                continue
            writer.add_document(self._document(rel, text))
            indexed += 1
        writer.commit()
        try:
            writer.wait_merging_threads()
        except Exception:  # noqa: BLE001 — merge bookkeeping only; the commit landed
            pass
        del writer
        index.reload()
        self._save_state({p: list(m) for p, m in current.items()})
        return indexed, len(removed)

    def _document(self, rel: str, text: str):
        doc = tantivy.Document()
        doc.add_text("path", rel)
        doc.add_text("pathtext", rel)
        doc.add_text("filename", os.path.basename(rel))
        ext = os.path.splitext(rel)[1].lstrip(".").lower()
        doc.add_text("lang", " ".join(filter(None, (ext, lang_for(rel)))))
        doc.add_text("content", text)
        camel = _camel_tokens(text)
        if camel:
            doc.add_text("content", " ".join(camel))
        return doc

    # -- query ------------------------------------------------------------------

    def search(self, query: str, path: str = None, limit: int = DEFAULT_LIMIT) -> list:
        """Ranked hits for a natural-language query. Never raises on a bad query."""
        limit = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
        terms = query_terms(query)
        if not terms:
            return []
        with self._lock:
            self.reconcile()
            index = self._open()
            searcher = index.searcher()
            # OR by default: a question rarely has every one of its words in the file
            # it is about, and BM25 already ranks the docs that match more of them higher.
            q = index.parse_query(" ".join(terms), default_field_names=list(self._FIELDS),
                                  field_boosts=dict(self._BOOSTS))
            # Over-fetch when a path filter is set — filtering happens on the way out,
            # so the cap has to be applied after, not before.
            raw = searcher.search(q, limit=limit * 10 if path else limit).hits

        prefix = self._norm_prefix(path)
        hits = []
        for score, addr in raw:
            rel = searcher.doc(addr).get_first("path")
            if prefix and not (rel == prefix or rel.startswith(prefix + os.sep)):
                continue
            snip = self._snippet(rel, terms)
            if snip is None:       # deleted since the last commit; the next search re-syncs
                continue
            line, lines = snip
            hits.append(Hit(path=rel, line=line, score=round(float(score), 3), lines=lines))
            if len(hits) >= limit:
                break
        return hits

    def _norm_prefix(self, path):
        """A `path` argument as a repo-relative directory/file prefix, or None."""
        if not path:
            return None
        p = path.strip().strip("/")
        if os.path.isabs(path):
            try:
                p = os.path.relpath(path, self.root)
            except ValueError:
                return None
        return os.path.normpath(p) if p else None

    def _snippet(self, rel: str, terms: list):
        """The best-matching line in `rel` plus context: (lineno, [(lineno, text)]).

        Read from disk rather than from a stored field: the snippet then shows what
        the file says NOW, and a stale index can point at the wrong place but never
        quote text that does not exist.
        """
        text = self._read_text(rel)
        if text is None:
            return None
        lines = text.splitlines()
        if not lines:
            return 1, []
        probes = [_line_probe(t) for t in terms]
        best, best_score = 0, 0
        for i, line in enumerate(lines):
            low = line.lower()
            score = sum(1 for p in probes if p in low)
            if score > best_score:
                best, best_score = i, score
        lo = max(0, best - _CONTEXT_LINES)
        hi = min(len(lines), best + _CONTEXT_LINES + 1)
        return best + 1, [(n + 1, lines[n][:_MAX_LINE_CHARS]) for n in range(lo, hi)]


# -- rendering ------------------------------------------------------------------

def format_hits(hits: list, query: str) -> str:
    """Ranked hits as the compact text the model reads.

    `path:line` first on its own line so it is copy-pasteable into the next bash
    command, then the numbered context. Whole files are never returned: this is a
    navigation primitive, and the next step is the model's to choose.
    """
    if not hits:
        return (f"[no matches for {query!r}]\n"
                "Ranked search found nothing. Try different words, or use bash with "
                "`rg` for an exact/regex search.")
    out = [f"{len(hits)} result{'s' if len(hits) != 1 else ''} for {query!r}:"]
    for h in hits:
        out.append("")
        out.append(f"{h.path}:{h.line}")
        out.extend(f"{n:>5}| {t}" for n, t in h.lines)
    return "\n".join(out)
