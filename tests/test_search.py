"""Characterization battery for ranked lexical search: `search.CodeSearchIndex`
and the `search` tool in `tools.py`. Pure filesystem in temp dirs — no model load,
no network, and the index always lands in a per-test CHAD_SEARCH_DIR so a test run
never touches the developer's ~/.chad/cache.

Three invariants matter here:

1. **Freshness.** A stale index is worse than a slow one: it points the model
   confidently at code that no longer exists. Add/modify/delete must all be
   visible on the next search.
2. **Exclusions.** The index reuses repomap's ignore policy, so model weights,
   virtualenvs and installed packages must never become searchable.
3. **Degradation.** Corrupt state, binary files and malformed queries produce a
   message, never an exception — the tool costs a turn either way, and an
   exception costs the turn AND the answer.
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

from chad import search as sr  # noqa: E402
from chad import tools  # noqa: E402

pytestmark = pytest.mark.skipif(not sr.available(), reason="tantivy not installed")

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


# A synthetic repository whose concepts do NOT line up with its filenames, so a
# query can only be answered by ranking content — the case rg is bad at.
REPO = {
    "src/auth/login.py": (
        'def authenticate(user, password):\n'
        '    """Verify credentials and raise on a failed login attempt."""\n'
        '    if not verify_password(user, password):\n'
        '        raise AuthError("bad credentials")\n'
        '    return issue_session(user)\n'),
    "src/auth/middleware.py": (
        'class AuthMiddleware:\n'
        '    """Attach the authenticated principal to every inbound request."""\n'
        '    def __call__(self, request):\n'
        '        request.user = authenticate(request.token)\n'),
    "src/billing/errors.py": (
        'MAX_RETRIES = 5\n'
        'def charge_with_retry(invoice):\n'
        '    """Retry a declined card charge with exponential backoff."""\n'
        '    for attempt in range(MAX_RETRIES):\n'
        '        try:\n'
        '            return gateway.charge(invoice)\n'
        '        except TransientDeclineError:\n'
        '            sleep(2 ** attempt)\n'),
    "src/fhir/validation.py": (
        'def validate_resource(resource):\n'
        '    """Validate a FHIR resource against its StructureDefinition."""\n'
        '    problems = check_profile(resource)\n'
        '    if problems:\n'
        '        raise FhirValidationError(problems)\n'),
    "tests/test_auth.py": (
        'def test_login_rejects_a_bad_password():\n'
        '    assert authenticate("bob", "wrong") is None\n'),
    "docs/authentication.md": (
        '# Authentication\n'
        'How sessions are issued and how sign-in failures are reported.\n'),
}


def _seed(files=None, root=None):
    """Create a temp repo containing `files` (path->content, nested ok) and point the
    search cache at its own temp dir. Caller MUST call _teardown."""
    d = root or tempfile.mkdtemp(prefix="search_")
    for rel, content in (files if files is not None else REPO).items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
    cache = tempfile.mkdtemp(prefix="searchcache_")
    os.environ["CHAD_SEARCH_DIR"] = cache
    sr.CodeSearchIndex.reset()
    return d, cache


def _teardown(cache, cwd=None):
    os.environ.pop("CHAD_SEARCH_DIR", None)
    sr.CodeSearchIndex.reset()
    shutil.rmtree(cache, ignore_errors=True)
    if cwd:
        os.chdir(cwd)


def _paths(hits):
    return [h.path for h in hits]


# --- query normalization ------------------------------------------------------

def test_query_terms_strips_syntax_and_scaffolding():
    """The model never sees Tantivy's query language, so nothing it can type may be
    read as one: punctuation is dropped, question words carry no retrieval signal."""
    check("question words dropped",
          sr.query_terms("where are the login failures handled?") ==
          ["login", "failures", "handled"],
          sr.query_terms("where are the login failures handled?"))
    check("field syntax is inert",
          sr.query_terms('content:"foo" AND bar~2 OR (baz)') ==
          ["content", "foo", "bar", "baz"],
          sr.query_terms('content:"foo" AND bar~2 OR (baz)'))
    check("camelCase query splits",
          sr.query_terms("AuthMiddleware") == ["authmiddleware", "auth", "middleware"],
          sr.query_terms("AuthMiddleware"))
    check("punctuation-only query is empty", sr.query_terms("?? -- **") == [])


# --- index construction / exclusions ------------------------------------------

def test_indexes_text_and_skips_excluded_trees():
    """Every exclusion the repo already centralizes in ignore.py, plus the size and
    binary boundaries. A model-weights directory in the index is a context disaster."""
    files = dict(REPO)
    for junk in (".git/config", "node_modules/pkg/index.js", ".venv/lib/site.py",
                 "venv/lib/site.py", "__pycache__/x.py", ".cache/blob.txt",
                 "models/weights.txt", "vendor/site-packages/lib.py",
                 "dist/bundle.js", "build/out.txt", ".mypy_cache/m.json"):
        files[junk] = "authentication login session secret\n"
    d, cache = _seed(files)
    try:
        big = os.path.join(d, "src", "huge.py")
        with open(big, "w") as f:
            f.write("# authentication\n" + ("x = 1\n" * 200_000))   # >1MB
        with open(os.path.join(d, "src", "blob.bin"), "wb") as f:
            f.write(b"authentication\x00\x01\x02binary")
        ix = sr.CodeSearchIndex(d)
        indexed, removed = ix.reconcile()
        check("cold reconcile indexed the real files", indexed == len(REPO), indexed)
        check("cold reconcile removed nothing", removed == 0, removed)

        hits = _paths(ix.search("authentication login session secret", limit=20))
        check("only real sources are searchable", set(hits) <= set(REPO), hits)
        for junk in ("node_modules", ".git", "venv", "__pycache__", ".cache",
                     "models", "site-packages", "dist", "build", ".mypy_cache"):
            check(f"{junk} never appears", not any(junk in p for p in hits), hits)
        check("the >1MB file never appears", "src/huge.py" not in hits, hits)
        check("the binary file never appears", "src/blob.bin" not in hits, hits)
    finally:
        _teardown(cache)
        shutil.rmtree(d, ignore_errors=True)


def test_index_lives_outside_the_repository():
    """The index is derived state: it must never show up in the working tree."""
    d, cache = _seed()
    try:
        before = sorted(os.listdir(d))
        ix = sr.CodeSearchIndex(d)
        ix.reconcile()
        ix.search("authentication")
        check("repo root untouched", sorted(os.listdir(d)) == before, os.listdir(d))
        check("index is under the cache dir", ix.cache_dir().startswith(cache),
              ix.cache_dir())
        check("index files exist there", os.path.isdir(os.path.join(ix.cache_dir(), "index")))
    finally:
        _teardown(cache)
        shutil.rmtree(d, ignore_errors=True)


# --- search quality -----------------------------------------------------------

def test_natural_language_queries_rank_the_right_file():
    """Top-N containment, not an exact ordering: several files are legitimately
    relevant to each question and pinning first place would be a brittle test of
    BM25's tie-breaks rather than of retrieval."""
    d, cache = _seed()
    try:
        ix = sr.CodeSearchIndex(d)
        for query, wanted, n in (
            ("where are login authentication failures handled?", "src/auth/login.py", 3),
            ("FHIR validation errors", "src/fhir/validation.py", 2),
            ("billing retry behavior", "src/billing/errors.py", 2),
            ("authentication middleware", "src/auth/middleware.py", 2),
            ("find the tests covering sign-in", "tests/test_auth.py", 3),
        ):
            top = _paths(ix.search(query, limit=n))
            check(f"{query!r} -> {wanted} in top {n}", wanted in top, top)
    finally:
        _teardown(cache)
        shutil.rmtree(d, ignore_errors=True)


def test_results_are_locations_not_files():
    """A search result is a place to look, not a file dump: path:line plus a few
    clipped lines, and never the whole file."""
    d, cache = _seed()
    try:
        ix = sr.CodeSearchIndex(d)
        hits = ix.search("exponential backoff on a declined charge", limit=1)
        check("one hit", len(hits) == 1, hits)
        h = hits[0]
        check("the right file", h.path == "src/billing/errors.py", h.path)
        check("points at the matching line", h.lines[0][0] <= h.line <= h.lines[-1][0], h)
        check("bounded context", len(h.lines) <= 2 * sr._CONTEXT_LINES + 1, h.lines)
        check("not the whole file", len(h.lines) < 8, h.lines)
        out = sr.format_hits(hits, "backoff")
        check("renders path:line", f"{h.path}:{h.line}" in out, out)
        check("no whole-file dump", out.count("\n") < 8, out)
    finally:
        _teardown(cache)
        shutil.rmtree(d, ignore_errors=True)


def test_long_lines_are_clipped():
    """One minified line must not spend the whole result budget."""
    d, cache = _seed({"app/bundle.js": "// widget\nvar widgetRegistry = " + "z" * 5000 + ";\n"})
    try:
        hits = sr.CodeSearchIndex(d).search("widget registry", limit=1)
        check("hit found", len(hits) == 1, hits)
        for _, text in hits[0].lines:
            check("line clipped", len(text) <= sr._MAX_LINE_CHARS, len(text))
    finally:
        _teardown(cache)
        shutil.rmtree(d, ignore_errors=True)


def test_camel_case_identifiers_are_reachable_by_word():
    """`parseAuthHeader` is one token to any word tokenizer; the split parts are
    indexed alongside so a plain-English query can still reach it."""
    d, cache = _seed({"src/http.py": "def parseAuthHeader(req):\n    return req\n",
                      "src/other.py": "def unrelated():\n    return 1\n"})
    try:
        hits = _paths(sr.CodeSearchIndex(d).search("parse header", limit=2))
        check("camel identifier reachable", "src/http.py" in hits, hits)
    finally:
        _teardown(cache)
        shutil.rmtree(d, ignore_errors=True)


def test_path_restricts_results():
    d, cache = _seed()
    try:
        ix = sr.CodeSearchIndex(d)
        hits = _paths(ix.search("authentication", path="src/auth", limit=10))
        check("restricted to the subtree", hits and all(p.startswith("src/auth/") for p in hits),
              hits)
        check("unrestricted sees more",
              len(_paths(ix.search("authentication", limit=10))) > len(hits))
        check("a path matching nothing yields nothing",
              ix.search("authentication", path="src/fhir", limit=10) == [])
    finally:
        _teardown(cache)
        shutil.rmtree(d, ignore_errors=True)


# --- incremental updates ------------------------------------------------------

def test_incremental_reconciliation():
    """Add / modify / delete, and the warm no-op. A stale index is the one failure
    mode that actively misleads the model, so each transition is pinned."""
    d, cache = _seed()
    try:
        ix = sr.CodeSearchIndex(d)
        check("cold indexes everything", ix.reconcile() == (len(REPO), 0))
        check("warm reconcile is a no-op", ix.reconcile() == (0, 0))
        check("unchanged search still works",
              "src/fhir/validation.py" in _paths(ix.search("FHIR validation", limit=2)))

        # modify: the new contents are searchable, the old ones are not.
        p = os.path.join(d, "src", "fhir", "validation.py")
        os.utime(p, (0, 0))                       # force a distinguishable mtime
        with open(p, "w") as f:
            f.write("def check_quantity(observation):\n"
                    '    """Reject an observation whose unit is not UCUM."""\n')
        check("one file reindexed", ix.reconcile() == (1, 0))
        check("new content is searchable",
              "src/fhir/validation.py" in _paths(ix.search("UCUM unit observation", limit=3)))
        check("old content is gone",
              "src/fhir/validation.py" not in _paths(ix.search("StructureDefinition", limit=5)))

        # add
        with open(os.path.join(d, "src", "billing", "refund.py"), "w") as f:
            f.write("def issue_refund(charge):\n    return gateway.refund(charge)\n")
        check("one file added", ix.reconcile() == (1, 0))
        check("added file is searchable",
              "src/billing/refund.py" in _paths(ix.search("issue a refund", limit=3)))

        # delete
        os.remove(os.path.join(d, "src", "auth", "middleware.py"))
        check("one file removed", ix.reconcile() == (0, 1))
        check("deleted file is gone",
              "src/auth/middleware.py" not in _paths(ix.search("authentication middleware",
                                                               limit=10)))
    finally:
        _teardown(cache)
        shutil.rmtree(d, ignore_errors=True)


def test_search_reconciles_lazily():
    """Freshness without a watcher: the first search of a session pays the update."""
    d, cache = _seed()
    try:
        ix = sr.CodeSearchIndex(d)
        check("nothing indexed before the first search",
              not os.path.exists(os.path.join(ix.cache_dir(), "state.json")))
        check("first search sees the repo",
              "src/auth/login.py" in _paths(ix.search("failed login attempt", limit=3)))
        with open(os.path.join(d, "src", "auth", "totp.py"), "w") as f:
            f.write("def verify_totp(code):\n    return check_time_based_code(code)\n")
        check("a file written after the first search is still found",
              "src/auth/totp.py" in _paths(ix.search("time based one time code", limit=3)))
    finally:
        _teardown(cache)
        shutil.rmtree(d, ignore_errors=True)


# --- corruption / rebuild -----------------------------------------------------

def test_corrupt_state_rebuilds():
    """The index is disposable by construction: truncate the sidecar or the index
    itself and the next search rebuilds, without touching a repository file."""
    d, cache = _seed()
    try:
        ix = sr.CodeSearchIndex(d)
        ix.search("authentication")
        before = {rel: open(os.path.join(d, rel)).read() for rel in REPO}

        with open(os.path.join(ix.cache_dir(), "state.json"), "w") as f:
            f.write("{not json at all")
        check("corrupt state rebuilds",
              "src/auth/login.py" in _paths(ix.search("failed login attempt", limit=3)))

        shutil.rmtree(os.path.join(ix.cache_dir(), "index"), ignore_errors=True)
        os.remove(os.path.join(ix.cache_dir(), "state.json"))
        sr.CodeSearchIndex.reset()
        fresh = sr.CodeSearchIndex(d)
        check("a deleted index rebuilds",
              "src/auth/login.py" in _paths(fresh.search("failed login attempt", limit=3)))

        for rel, text in before.items():
            check(f"{rel} untouched by the rebuild",
                  open(os.path.join(d, rel)).read() == text)
    finally:
        _teardown(cache)
        shutil.rmtree(d, ignore_errors=True)


def test_garbage_index_directory_is_replaced():
    d, cache = _seed()
    try:
        ix = sr.CodeSearchIndex(d)
        os.makedirs(ix._index_dir(), mode=0o700, exist_ok=True)
        with open(os.path.join(ix._index_dir(), "meta.json"), "w") as f:
            f.write("this is not a tantivy index")
        check("garbage index is rebuilt, not raised on",
              "src/auth/login.py" in _paths(ix.search("failed login attempt", limit=3)))
    finally:
        _teardown(cache)
        shutil.rmtree(d, ignore_errors=True)


# --- tool contract ------------------------------------------------------------

def test_tool_registration():
    """The tool surface additions: dispatchable, described, and NOT mutating."""
    names = [s["function"]["name"] for s in tools.SCHEMAS]
    check("search has a schema", "search" in names, names)
    check("exactly one search schema", names.count("search") == 1, names)
    check("search is dispatchable", "search" in tools.DISPATCH, sorted(tools.DISPATCH))
    check("search is NOT mutating", tools.is_mutating("search") is False)
    check("search is not auto-edit", "search" not in {"write", "edit", "bash"})
    check("the pre-existing surface is intact",
          {"bash", "write", "edit", "write_todos", "done"} <= set(names), names)

    fn = [s for s in tools.SCHEMAS if s["function"]["name"] == "search"][0]["function"]
    check("query is the only required param", fn["parameters"]["required"] == ["query"])
    check("params are query/path/limit",
          sorted(fn["parameters"]["properties"]) == ["limit", "path", "query"])
    check("the description teaches the split",
          "bash" in fn["description"] and "exact" in fn["description"], fn["description"])
    check("no tantivy jargon leaks to the model",
          "tantivy" not in fn["description"].lower(), fn["description"])


def test_baseline_arm_hides_the_tool():
    """CHAD_NO_SEARCH is the paired benchmark's baseline arm: the model must not see
    the schema at all, or the comparison is not against the current harness."""
    old = os.environ.get("CHAD_NO_SEARCH")
    try:
        os.environ["CHAD_NO_SEARCH"] = "1"
        names = [s["function"]["name"] for s in tools.active_schemas()]
        check("search hidden in the baseline arm", "search" not in names, names)
        check("bash survives the baseline arm", "bash" in names, names)
        os.environ.pop("CHAD_NO_SEARCH")
        check("search present by default",
              "search" in [s["function"]["name"] for s in tools.active_schemas()])
    finally:
        if old is None:
            os.environ.pop("CHAD_NO_SEARCH", None)
        else:
            os.environ["CHAD_NO_SEARCH"] = old


def test_prompt_makes_search_the_opening_move():
    """`search` is written into the prompt as how a turn STARTS, not as an extra
    bullet — one bullet measurably lost to the bash-first instructions around it.
    All four places must say so, and bash must still own exact matching."""
    from chad.prompt import build_system_prompt
    old = os.environ.get("CHAD_NO_SEARCH")
    try:
        os.environ.pop("CHAD_NO_SEARCH", None)
        on = build_system_prompt()
        check("named in the tool list", "- `search`" in on)
        check("named as the first move", "FIRST move" in on)
        check("in the typical turn", "write_todos → search (locate)" in on)
        check("in the locate habit", "Locate code with `search` first" in on)
        check("bash still owns exact matching", "`rg` when you do" in on)
        check("still points at rg for exhaustive matching", "need EVERY match" in on)
    finally:
        if old is None:
            os.environ.pop("CHAD_NO_SEARCH", None)
        else:
            os.environ["CHAD_NO_SEARCH"] = old


def test_ablation_arm_restores_the_previous_prompt():
    """CHAD_NO_SEARCH withholds the schema, so the prompt must not name the tool
    there either. The revert table is separate text from the prompt it reverts, so
    this pins BOTH halves: every `present` string still exists in _BASE_PROMPT (an
    un-matched substitution would silently leave a half-reverted arm), and the result
    mentions nothing the baseline model cannot call."""
    from chad import prompt as pr
    old = os.environ.get("CHAD_NO_SEARCH")
    try:
        for present, _previous in pr._NO_SEARCH_SUBS:
            check("revert table is in step with the prompt", present in pr._BASE_PROMPT,
                  present[:60])
        os.environ["CHAD_NO_SEARCH"] = "1"
        off = pr.build_system_prompt()
        check("baseline arm never names the tool", "`search`" not in off, off[:200])
        check("baseline arm restores the bash-first line",
              "- `bash` — your primary tool: locate, read, run, verify." in off)
        check("baseline arm restores the ripgrep locate habit",
              "- Locate code with ripgrep:" in off)
        check("baseline arm restores the bash-first turn",
              "write_todos → bash (locate, then read the region)" in off)
    finally:
        if old is None:
            os.environ.pop("CHAD_NO_SEARCH", None)
        else:
            os.environ["CHAD_NO_SEARCH"] = old


def test_tool_dispatch_and_degradation():
    """Every failure mode returns a message. `search` costs a turn on the way out;
    raising would cost the turn AND the answer."""
    d, cache = _seed()
    cwd = os.getcwd()
    try:
        os.chdir(d)
        sr.CodeSearchIndex.reset()
        out = tools.DISPATCH["search"]({"query": "where are login failures handled?"})
        check("dispatch returns ranked text", "src/auth/login.py:" in out, out)
        check("result is compact", len(out) < 2000, len(out))

        check("empty query is a message, not a raise",
              "[search: empty query]" in tools.DISPATCH["search"]({"query": "   "}))
        check("punctuation-only query is a message",
              "[no matches" in tools.DISPATCH["search"]({"query": "?? ** --"}))
        check("a no-match query points at rg",
              "rg" in tools.tool_search("zzzqqq nonexistent xylophone"))

        check("limit is honored",
              tools.tool_search("authentication", limit=1).count("\n\n") == 1)
        check("limit is capped at MAX_LIMIT",
              len(sr.CodeSearchIndex.for_root(".").search("authentication",
                                                          limit=9999)) <= sr.MAX_LIMIT)
        check("path is honored",
              "src/billing" not in tools.tool_search("authentication", path="src/auth"))
    finally:
        os.chdir(cwd)
        _teardown(cache)
        shutil.rmtree(d, ignore_errors=True)


def test_tool_never_runs_a_shell_command():
    """`search` is a read-only retrieval primitive; a query that looks like a shell
    command is just words."""
    d, cache = _seed()
    cwd = os.getcwd()
    marker = os.path.join(d, "PWNED")
    try:
        os.chdir(d)
        sr.CodeSearchIndex.reset()
        tools.tool_search(f"; touch {marker}; echo authentication")
        tools.tool_search(f"$(touch {marker}) `touch {marker}` authentication")
        check("no shell command ran", not os.path.exists(marker))
    finally:
        os.chdir(cwd)
        _teardown(cache)
        shutil.rmtree(d, ignore_errors=True)
